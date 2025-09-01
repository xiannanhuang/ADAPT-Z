import torch
from trainers.w_student.trainerBaseFramework import TrainerBaseFramework
from utils.tools import Struct
import importlib
import warnings
from typing import Any, Dict
from argparse import Namespace
from dataloader import DatasetCustom
import torch.nn as nn

warnings.filterwarnings('ignore')


class Exp_TS2VecSupervised(TrainerBaseFramework):
    """
    Trainer framework for supervised TS2Vec with a student model.
    Combines predictions from a main model and a student model for time series forecasting.
    """

    def __init__(self, args: Namespace, main_model: nn.Module, student_model: nn.Module):
        """
        Initialize the Exp_TS2VecSupervised trainer.

        Args:
            args: Configuration arguments.
            main_model: The main model class or instance.
            student_model: The student model class (must not be None).
        """
        assert student_model is not None, 'Student model is required for this framework'
        super().__init__(args, main_model, student_model)

        # Instantiate the student model with appropriate arguments
        self.student_model = student_model(
            Struct(args).dict2attr('student_model'),
            args.seq_len + args.pred_len
        ).to(self.device)

        # Dynamically import and instantiate the student model's forward function
        self.studentFFN = getattr(
            importlib.import_module(
                f'trainers.forward.trainerForward_{args.student_model["model"]}'
            ),
            'TrainerForward'
        )(args, self.student_model, self.device)

    def get_final_output(
        self,
        main_pred: torch.Tensor,
        student_pred: torch.Tensor,
        batch_xy: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine the predictions from the main and student models.

        Args:
            main_pred: Prediction from the main model.
            student_pred: Prediction from the student model.
            batch_xy: Concatenated input and future prediction.

        Returns:
            Combined prediction tensor.
        """
        match_dim = main_pred.shape == student_pred.shape

        if match_dim:
            # If shapes match, sum predictions
            return main_pred + student_pred

        if student_pred.shape[1] > main_pred.shape[1]:
            # If student predicts more steps, use batch_xy + student_pred
            return batch_xy + student_pred

        # Otherwise, sum future of main_pred and student_pred
        return self.mainFFN.get_future(main_pred) + student_pred

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
        Process a single batch through both main and student models.

        Args:
            dataset_object: Dataset object for the batch.
            batch_x: Input sequence tensor.
            batch_y: Target sequence tensor.
            batch_x_mark: Input time marker tensor.
            batch_y_mark: Target time marker tensor.
            mode: Mode string ('train', 'val', 'test').

        Returns:
            Dictionary of outputs including predictions.
        """
        # Forward pass through main model
        main_outputs = self.mainFFN.ffn(
            dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode
        )
        outputs = main_outputs
        outputs.update({
            'predT': main_outputs['pred'],
            'pred': torch.clone(main_outputs['pred']).detach(),
        })

        # Prepare input for student model by concatenating input and main model's future prediction
        batch_xy = torch.cat(
            [batch_x.to(self.device), self.mainFFN.get_future(outputs['pred'])],
            dim=1
        )
        batch_xy_mark = torch.cat(
            [batch_x_mark.to(self.device), batch_y_mark.to(self.device)],
            dim=1
        )

        # Forward pass through student model
        student_outputs = self.studentFFN.ffn(
            dataset_object, batch_xy, batch_y, batch_xy_mark, batch_y_mark, mode
        )

        # Update outputs with student prediction and combined prediction
        outputs.update({
            'predS': student_outputs['pred'],
            'pred': self.get_final_output(outputs['pred'], student_outputs['pred'], batch_xy),
        })
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
            batch_x: Input sequence tensor.
            batch_y: Target sequence tensor.
            batch_x_mark: Input time marker tensor.
            batch_y_mark: Target time marker tensor.

        Returns:
            Dictionary with loss value.
        """
        outputs = self._process_one_batch(
            train_data, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train'
        )

        # Compute losses for teacher and feature transfer
        lossT = self.backward_lossT_train(outputs)
        lossFT = self.backward_lossFT(outputs, outputs)

        # Update model parameters
        for opt in self.opt['train']:
            opt.step()
        self.store_grad()
        self.update_ocp(outputs, outputs)
        for opt in self.opt['train']:
            opt.zero_grad()

        return {'loss': lossT.item()}