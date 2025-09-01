import torch
from trainers.trainerBaseERTD import TrainerBaseERTD
from argparse import Namespace
import warnings
warnings.filterwarnings('ignore')
from dataloader import DatasetCustom 

from typing import  Dict

class TrainerBaseFramework(TrainerBaseERTD):
    """
    Base framework for training and testing with a main model and a student model.
    Extends TrainerBaseERTD and provides methods for loss computation, backpropagation,
    optimizer selection, and test step execution.
    """

    def __init__(self, args: Namespace, main_model: torch.nn.Module, student_model: torch.nn.Module):
        """
        Initialize the TrainerBaseFramework.

        Args:
            args: Configuration arguments.
            main_model: The main model used for training.
            student_model: The student model used for training.
        """
        super().__init__(args, main_model, student_model)

    def backward_lossT_train(self, prev: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute and backpropagate the loss for the main model during training.

        Args:
            prev: Dictionary containing previous batch outputs.

        Returns:
            The computed loss tensor.
        """
        lossT = self.criterion(prev['predT'], self.mainFFN.gt4update(prev))            
        lossT.backward()
        return lossT

    def backward_lossT_test(self, prev: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute and backpropagate the loss for the main model during testing.

        Args:
            prev: Dictionary containing previous batch outputs.

        Returns:
            The computed loss tensor.
        """
        lossT = self.criterion(prev['predT'], self.mainFFN.gt4update(prev))            
        lossT.backward()
        return lossT
    
    def backward_lossFT(self, prev: Dict[str, torch.Tensor], curr: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute and backpropagate the loss for the student model.

        Args:
            prev: Dictionary containing previous batch outputs.
            curr: Dictionary containing current batch outputs.

        Returns:
            The computed loss tensor.
        """
        lossFT = self.criterion(prev['pred'], self.studentFFN.gt4update(curr))
        lossFT.backward()
        return lossFT
    
    def backward_tdlossFT(self, prev: Dict[str, torch.Tensor], curr: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute and backpropagate the temporal difference (TD) loss for the student model.

        Args:
            prev: Dictionary containing previous batch outputs.
            curr: Dictionary containing current batch outputs.

        Returns:
            The computed TD loss tensor.
        """
        td_truth = []
        for i, k in enumerate(self.indices):
            # If FITS is student model, the following code cannot work.
            td_truth.append(
                torch.cat([prev['true'][i:i+1, :k], 
                           self.mainFFN.td4update_pseudo(curr, k)], 
                          dim=1)
            )
        td_truth = torch.cat(td_truth, dim=0)
        
        lossFT = self.MSEReductNone(prev['pred'], td_truth) * self.discountedW
        lossFT = lossFT.mean()
        lossFT.backward()
        return lossFT

    def _select_optimizer(self) -> Dict[str, torch.optim.Optimizer]:
        """
        Select and configure optimizers for both main and student models.

        Returns:
            A dictionary containing optimizers for training and testing.
        """
        opts = super()._select_optimizer()
        opts['train'].append(self._get_optim(self.args.opt_student['train'], self.student_model.parameters()))
        opts['test']['batch'].append(self._get_optim(self.args.opt_student['test']['batch'], self.student_model.parameters()))
        opts['test']['online'].append(self._get_optim(self.args.opt_student['test']['online'], self.student_model.parameters()))
        return opts

    def test_step(
        self,
        dataset_object: DatasetCustom,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_x_mark: torch.Tensor,
        batch_y_mark: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a test step, including optional experience replay (ER) and temporal difference (TD) updates.

        Args:
            dataset_object: The dataset object.
            batch_x: Input batch data.
            batch_y: Target batch data.
            batch_x_mark: Input batch markers.
            batch_y_mark: Target batch markers.

        Returns:
            A dictionary containing the processed batch outputs.
        """
        if self.args.er:
            self.er(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark)
        
        if self.args.td:
            self.td(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark)

        with torch.no_grad():    
            curr = self._process_one_batch(
                dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='test'
            )
            curr.update({'pred': self.studentFFN.get_future(curr['pred'])})
        curr = self.clean(curr)            
        
        return curr
