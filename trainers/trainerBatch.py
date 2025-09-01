from trainers.trainerBase import TrainerBase
import warnings
import torch
from argparse import Namespace
from typing import Any, Dict, Optional
from torch import nn
from torch.utils.data import DataLoader
from dataloader import DatasetCustom 
warnings.filterwarnings('ignore')

class Exp_TS2VecSupervised(TrainerBase):
    """
    Trainer class for supervised TS2Vec experiments.
    Inherits from TrainerBase and implements training and testing steps.
    """

    def __init__(self, args: Namespace, main_model: nn.Module, student_model: nn.Module = None) -> None:
        """
        Initialize the Exp_TS2VecSupervised trainer.

        Args:
            args: Arguments or configuration for the trainer.
            main_model: The main model to be trained.
            student_model: Not used in this framework (must be None).
        """
        assert student_model is None, 'Student model is not required for this framework'
        super().__init__(args, main_model)
    
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
            dataset_object: The dataset object providing context or utilities.
            batch_x: Input features tensor.
            batch_y: Target tensor.
            batch_x_mark: Additional input markers.
            batch_y_mark: Additional target markers.
            mode: Mode of operation ('train' or 'test').

        Returns:
            Dictionary of outputs from the model.
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
            dataset_object: The dataset object.
            batch_x: Input features tensor.
            batch_y: Target tensor.
            batch_x_mark: Additional input markers.
            batch_y_mark: Additional target markers.

        Returns:
            Dictionary containing the loss value.
        """
        outputs = self._process_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train')
        # Compute loss between predictions and ground truth
        loss = self.criterion(outputs['pred'], self.mainFFN.gt4update(outputs))
        loss.backward()
        # Update model parameters
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
        Perform a single evaluation (test) step.

        Args:
            dataset_object: The dataset object.
            batch_x: Input features tensor.
            batch_y: Target tensor.
            batch_x_mark: Additional input markers.
            batch_y_mark: Additional target markers.
            model: Optional model to use (defaults to main model).

        Returns:
            Cleaned dictionary of outputs from the model.
        """
        with torch.no_grad():
            curr = self._process_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='test')
        curr = self.clean(curr)
        return curr
