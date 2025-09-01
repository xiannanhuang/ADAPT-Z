import torch
from trainers.trainerBaseERTD import TrainerBaseERTD
import warnings
from typing import Any, Dict, Optional
from data.data_loader import DatasetCustom
from argparse import Namespace
from torch import nn

warnings.filterwarnings('ignore')


class Exp_TS2VecSupervised(TrainerBaseERTD):
    """
    Inherits from TrainerBaseERTD.
    """

    def __init__(self, args: Namespace, main_model: nn.Module, student_model: Optional[nn.Module] = None):
        """
        Initialize the trainer.

        Args:
            args: Arguments/configuration object.
            main_model: The main model to be trained.
            student_model: Should be None for this framework.
        """
        super().__init__(args, main_model, student_model)
        assert args.student_model['model'] is None, (
            'Student model exists. This framework is for training without student model.'
        )

    def backward_lossT_train(self, prev: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute and backpropagate the loss for the training target.

        Args:
            prev: Dictionary containing predictions and other batch info.

        Returns:
            The computed loss tensor.
        """
        lossT = self.criterion(prev['predT'], self.mainFFN.gt4update(prev))
        lossT.backward()
        return lossT

    def backward_lossT_test(self, prev: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute and backpropagate the loss for the test target.

        Args:
            prev: Dictionary containing predictions and other batch info.

        Returns:
            The computed loss tensor.
        """
        lossT = self.criterion(prev['predT'], self.mainFFN.gt4update(prev))
        lossT.backward()
        return lossT

    def backward_lossFT(self, prev: Dict[str, torch.Tensor], curr: Dict[str, torch.Tensor]) -> None:
        """
        Placeholder for feature-target loss backpropagation.
        Not used in this framework.
        """
        return

    def backward_tdlossFT(self, prev: Dict[str, torch.Tensor], curr: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute and backpropagate the temporal difference (TD) loss for feature-target pairs.

        Args:
            prev: Previous batch outputs.
            curr: Current batch outputs.

        Returns:
            The computed TD loss tensor.
        """
        td_truth = []
        for i, k in enumerate(self.indices):
            # Concatenate true values up to k and pseudo targets from current batch
            td_truth.append(
                torch.cat([
                    prev['true'][i:i+1, :k],
                    self.mainFFN.td4update_pseudo(curr, k)
                ], dim=1)
            )
        td_truth = torch.cat(td_truth, dim=0)
        lossFT = self.MSEReductNone(prev['pred'], td_truth) * self.discountedW
        lossFT = lossFT.mean()
        lossFT.backward()
        return lossFT

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
        Process a single batch through the main feed-forward network.

        Args:
            dataset_object: Dataset object for the batch.
            batch_x: Input features.
            batch_y: Target values.
            batch_x_mark: Input time markers.
            batch_y_mark: Target time markers.
            mode: Mode string ('train' or 'test').

        Returns:
            Dictionary of outputs including predictions.
        """
        outputs = self.mainFFN.ffn(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode)
        outputs.update({'predT': outputs['pred']})
        return outputs

    def train_step(
        self,
        dataset_object: DatasetCustom,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_x_mark: torch.Tensor,
        batch_y_mark: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            dataset_object: Dataset object for the batch.
            batch_x: Input features.
            batch_y: Target values.
            batch_x_mark: Input time markers.
            batch_y_mark: Target time markers.

        Returns:
            Dictionary containing the loss value.
        """
        outputs = self._process_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train')
        loss = self.criterion(outputs['pred'], self.mainFFN.gt4update(outputs))
        loss.backward()

        for opt in self.opt['train']:
            opt.step()
        self.store_grad()
        self.update_ocp(outputs, outputs)

        return {'loss': loss.item()}

    def test_step(
        self,
        dataset_object: DatasetCustom,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_x_mark: torch.Tensor,
        batch_y_mark: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a single test/validation step.

        Args:
            dataset_object: Dataset object for the batch.
            batch_x: Input features.
            batch_y: Target values.
            batch_x_mark: Input time markers.
            batch_y_mark: Target time markers.

        Returns:
            Dictionary of cleaned outputs.
        """
        self.er(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark)
        self.td(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark)

        with torch.no_grad():
            curr = self._process_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='test')

        curr = self.clean(curr)
        return curr