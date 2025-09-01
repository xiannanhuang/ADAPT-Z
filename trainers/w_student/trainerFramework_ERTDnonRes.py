import importlib
from trainers.w_student.trainerBaseFramework import TrainerBaseFramework
from utils.tools import Struct
import torch
from torch import nn
from argparse import Namespace
from typing import Dict
from data.data_loader import DatasetCustom
from torch.utils.data import DataLoader 

import warnings
warnings.filterwarnings('ignore')


class Exp_TS2VecSupervised(TrainerBaseFramework):
    """
    Experiment class for supervised training with TS2Vec architecture and student-teacher framework.
    Inherits from TrainerBaseFramework.
    """

    def __init__(
        self,
        args: Namespace,
        main_model: nn.Module,
        student_model: nn.Module,
    ):
        """
        Initialize the experiment.

        Args:
            args: Arguments/configuration object.
            main_model: Callable that returns the main model instance.
            student_model: Callable that returns the student model instance.
        """
        assert student_model is not None, 'Student model is required for this framework'
        super().__init__(args, main_model, student_model)

        # Check if main and student models have the same architecture
        self.same_arch: bool = True if args.main_model == args.student_model else False

        # Instantiate the student model
        self.student_model = student_model(
            Struct(args).dict2attr('student_model'), args.seq_len
        ).to(self.device)

        # Dynamically import the student model's forward function
        self.studentFFN = getattr(
            importlib.import_module(
                f'trainers.forward.trainerForward_{args.student_model["model"]}'
            ),
            'TrainerForward'
        )(args, self.student_model, self.device)

    def _process_one_batch(
        self,
        dataset_object: DatasetCustom,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_x_mark: torch.Tensor,
        batch_y_mark: torch.Tensor,
        mode: str = 'train'
    ) -> Dict[str, torch.Tensor]:
        """
        Process a single batch of data.

        Args:
            dataset_object: The dataset object.
            batch_x: Input features tensor.
            batch_y: Target tensor.
            batch_x_mark: Input time marker tensor.
            batch_y_mark: Target time marker tensor.
            mode: Mode string ('train', 'test', etc.).

        Returns:
            Dictionary containing model outputs and predictions.
        """
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y[:, -self.args.pred_len:, self.f_dim:].float().to(self.device)

        # Forward pass through main model
        main_outputs = self.mainFFN.ffn(
            dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode
        )

        # Forward pass through student model if required
        student_model_exists = mode == 'test' or not self.same_arch
        if student_model_exists:
            student_outputs = self.studentFFN.ffn(
                dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode
            )

        # Prepare output dictionary
        outputs = main_outputs
        outputs.update({
            'predT': main_outputs['pred'],  # Teacher prediction
            'predS': student_outputs['pred'] if student_model_exists else main_outputs['pred'],  # Student prediction
            'pred': student_outputs['pred'] if student_model_exists else main_outputs['pred'],   # Final prediction
        })
        outputs.update({'true': batch_y})  # Ground truth

        return outputs

    def train_step(
        self,
        train_data: DatasetCustom,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_x_mark: torch.Tensor,
        batch_y_mark: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            train_data: Training dataset object.
            batch_x: Input features tensor.
            batch_y: Target tensor.
            batch_x_mark: Input time marker tensor.
            batch_y_mark: Target time marker tensor.

        Returns:
            Dictionary with loss value.
        """
        outputs = self._process_one_batch(
            train_data, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train'
        )
        lossT = self.backward_lossT_train(outputs)

        # If architectures differ, perform additional backward pass for student
        if not self.same_arch:
            self.backward_lossFT(outputs, outputs)

        # Step all optimizers
        for opt in self.opt['train']:
            opt.step()
        self.store_grad()
        self.update_ocp(outputs, outputs)

        return {'loss': lossT.item()}

    def test(
        self,
        test_data: DatasetCustom,
        test_loader: DataLoader,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate the model on the test set.

        Args:
            test_data: Test dataset object.
            test_loader: DataLoader for test data.

        Returns:
            Test results as returned by the base class.
        """
        # If architectures are the same, copy weights from main to student
        if self.same_arch:
            self.student_model.load_state_dict(self.main_model.state_dict())

        return super().test(test_data, test_loader)
