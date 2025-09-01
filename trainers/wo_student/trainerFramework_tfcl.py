import torch
from trainers.wo_student.trainerFramework_plain import Exp_TS2VecSupervised as TrainerPlain
from utils.buffer import BufferFIFO as Buffer
from collections import deque
import numpy as np
import warnings
from typing import Any, Optional, Tuple, List, Dict
from data.data_loader import DatasetCustom
from argparse import Namespace
from torch import nn

warnings.filterwarnings('ignore')


class Exp_TS2VecSupervised(TrainerPlain):
    """
    Trainer class for TS2Vec supervised learning without a student model.
    Implements online continual learning with memory buffers and MAS regularization.
    """

    def __init__(
        self,
        args: Namespace,
        main_model: torch.nn.Module,
        student_model: Optional[torch.nn.Module]
    ):
        """
        Initialize the trainer.

        Args:
            args: Arguments/configuration object.
            main_model: The main model to be trained.
            student_model: Should be None for this framework.
        """
        super().__init__(args, main_model, student_model)
        assert args.student_model['model'] is None, 'Student model exists. This framework is for training without student model.'

        # Buffers for recent and hard samples
        self.laststeps: List[Tuple[torch.Tensor, ...]] = []
        self.recentBuffer: Buffer = Buffer(args.recent_buffer_size)
        self.hardBuffer: Buffer = Buffer(args.hard_buffer_size)

        # MAS (Memory Aware Synapses) regularization parameters
        self.mas_weight: float = args.mas_weight
        self.gradient_steps: int = args.gradient_steps

        # Buffer for tracking recent losses
        self.lossBuffer: deque = deque(maxlen=args.loss_buffer_size)

        # Statistics for loss distribution
        self.old_mu: float = 0
        self.old_std: float = 0

        # MAS variables
        self.star_variables: Optional[List[torch.Tensor]] = None
        self.omegas: Optional[List[torch.Tensor]] = None
        self.count_update: int = 0

        # MSE loss without reduction
        self.mse_NR = torch.nn.MSELoss(reduction='none')

    def test_step(
        self,
        dataset_object: DatasetCustom,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_x_mark: torch.Tensor,
        batch_y_mark: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a test step, including online continual learning updates.

        Args:
            dataset_object: Dataset object for batch processing.
            batch_x: Input batch data.
            batch_y: Target batch data.
            batch_x_mark: Input batch time markers.
            batch_y_mark: Target batch time markers.
            model: Optional model to use (default: None).

        Returns:
            Dictionary with processed batch results.
        """
        recent_loss: Optional[torch.Tensor] = None
        hard_loss: Optional[torch.Tensor] = None
        hx = hy = hx_mark = hy_mark = None

        # If enough steps have been collected, perform online updates
        if len(self.laststeps) >= self.args.pred_len:
            px, py, px_mark, py_mark = self.laststeps.pop(0)
            prev = self._process_one_batch(dataset_object, px, py, px_mark, py_mark, mode='test')
            loss = self.criterion(prev['pred'], self.mainFFN.gt4update(prev))

            for _ in range(self.gradient_steps):
                # Use recent buffer if full
                if self.recentBuffer.is_full():
                    bx, by, bx_mark, by_mark = self.recentBuffer.get_all_data()
                    b_new = self._process_one_batch(dataset_object, bx, by, bx_mark, by_mark, mode='test')
                    recent_loss = self.mse_NR(b_new['pred'], self.mainFFN.gt4update(b_new))
                    loss += recent_loss.sum()

                # Use hard buffer if full
                if self.hardBuffer.is_full():
                    hx, hy, hx_mark, hy_mark = self.hardBuffer.get_all_data()
                    h_new = self._process_one_batch(dataset_object, hx, hy, hx_mark, hy_mark, mode='test')
                    hard_loss = self.mse_NR(h_new['pred'], self.mainFFN.gt4update(h_new))
                    loss += hard_loss.sum()

                # MAS regularization
                mas_loss = 0
                if self.omegas is not None and self.star_variables is not None:
                    for omega, star_var, param in zip(self.omegas, self.star_variables, self.mainFFN.model.parameters()):
                        mas_loss += self.mas_weight / 2. * torch.sum(omega) * (param - star_var).pow(2).sum()
                loss += mas_loss

                # Backpropagation and optimizer step
                loss.backward()
                self.lossBuffer.append(loss.item())
                for opt in self.opt['test']['online']:
                    opt.step()
                self.store_grad()
                self.update_ocp(prev, prev)
                for opt in self.opt['test']['online']:
                    opt.zero_grad()

            # Update MAS statistics if hard buffer is full and enough loss history
            if self.hardBuffer.is_full() and len(self.lossBuffer) >= self.args.loss_buffer_size:
                new_mu = np.mean(self.lossBuffer)
                new_std = np.std(self.lossBuffer)

                if new_mu > self.old_mu + self.old_std:
                    self.count_update += 1
                    self.old_mu = new_mu
                    self.old_std = new_std

                    self.mainFFN.model.zero_grad()
                    h_new = self._process_one_batch(dataset_object, hx, hy, hx_mark, hy_mark, mode='test')
                    torch.norm(h_new['pred'], p=2).backward()  # Compute gradients for MAS
                    grad = [p.grad.data for p in self.mainFFN.model.parameters() if p.grad is not None]

                    # Update omega and star variables for MAS
                    self.omegas = [
                        1 / self.count_update * g + (1 - 1 / self.count_update) * omega
                        for omega, g in zip(self.omegas, grad)
                    ] if self.omegas is not None else grad
                    self.star_variables = [p.data for p in self.mainFFN.model.parameters()]

            # Add current batch to recent buffer
            self.recentBuffer.add_data(batch_x=px, batch_y=py, batch_x_mark=px_mark, batch_y_mark=py_mark)

            # Update hard buffer with hardest samples
            if recent_loss is not None:
                cat = lambda x, y: torch.cat([x, y], dim=0) if y is not None else x
                hx, hy, hx_mark, hy_mark = map(cat, [bx, by, bx_mark, by_mark], [hx, hy, hx_mark, hy_mark])

                individual_loss = cat(recent_loss, hard_loss) if hard_loss is not None else recent_loss
                individual_loss = torch.mean(individual_loss.data, dim=[1, 2])

                # Select hardest samples by loss
                indices = torch.Tensor([
                    x for x, _ in sorted(enumerate(individual_loss), key=lambda a: a[1], reverse=True)
                ]).int()
                indices = indices[:self.args.hard_buffer_size]
                self.hardBuffer.add_data(
                    batch_x=hx[indices],
                    batch_y=hy[indices],
                    batch_x_mark=hx_mark[indices],
                    batch_y_mark=hy_mark[indices]
                )

        # Store current batch for future online updates
        self.laststeps.append((batch_x, batch_y, batch_x_mark, batch_y_mark))

        # Run model in evaluation mode (no grad)
        with torch.no_grad():
            curr = self._process_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='test')
        curr = self.clean(curr)

        return curr