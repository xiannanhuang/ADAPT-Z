import torch
from trainers.trainerBase import TrainerBase
import warnings
from typing import Any, Dict, Tuple
from data.data_loader import DatasetCustom
from argparse import Namespace
from torch import nn

warnings.filterwarnings('ignore')

class Exp_TS2VecSupervised(TrainerBase):
    """
    Inherits from TrainerBase and implements training and testing steps.
    """

    def __init__(self, args: Namespace, main_model: nn.Module, student_model: Optional[nn.Module] = None):
        """
        Initialize the trainer.

        Args:
            args: Arguments/configuration object.
            main_model: The main model to be trained.
            student_model: Should be None. Asserted in code.
        """
        super().__init__(args, main_model)
        assert args.student_model['model'] is None, (
            'Student model exists. This framework is for training without student model.'
        )
        self.laststeps: list = []  # Stores recent batches for online adaptation

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
        Forward pass for one batch.

        Args:
            dataset_object: Dataset object for processing.
            batch_x: Input features.
            batch_y: Ground truth labels.
            batch_x_mark: Input time markers.
            batch_y_mark: Output time markers.
            mode: 'train' or 'test'.

        Returns:
            Model outputs as a dictionary.
        """
        outputs = self.mainFFN.ffn(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode)
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
            dataset_object: Dataset object for processing.
            batch_x: Input features.
            batch_y: Ground truth labels.
            batch_x_mark: Input time markers.
            batch_y_mark: Output time markers.

        Returns:
            Dictionary with loss value.
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
        Perform a single test step, with online adaptation if enough previous steps exist.

        Args:
            dataset_object: Dataset object for processing.
            batch_x: Input features.
            batch_y: Ground truth labels.
            batch_x_mark: Input time markers.
            batch_y_mark: Output time markers.
            model: Not used, kept for compatibility.

        Returns:
            Cleaned model outputs as a dictionary.
        """
        # Online adaptation if enough previous steps are stored
        if len(self.laststeps) >= self.args.pred_len:
            px, py, px_mark, py_mark = self.laststeps.pop(0)
            prev = self._process_one_batch(dataset_object, px, py, px_mark, py_mark, mode='test')
            loss = self.criterion(prev['pred'], self.mainFFN.gt4update(prev))
            loss.backward()

            for opt in self.opt['test']['online']:
                opt.step()
            self.store_grad()
            self.update_ocp(prev, prev)
            for opt in self.opt['test']['online']:
                opt.zero_grad()

        # Store current batch for future online adaptation
        self.laststeps.append((batch_x, batch_y, batch_x_mark, batch_y_mark))

        # Forward pass without gradient calculation
        with torch.no_grad():
            curr = self._process_one_batch(
                dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='test'
            )
        curr = self.clean(curr)

        return curr