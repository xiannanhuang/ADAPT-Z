import torch
from trainers.wo_student.trainerFramework_plain import Exp_TS2VecSupervised as TrainerPlain
from utils.buffer import Buffer
import warnings
from typing import Any, Optional, Tuple, Dict
from argparse import Namespace
from data.data_loader import DatasetCustom

warnings.filterwarnings('ignore')


class Exp_TS2VecSupervised(TrainerPlain):
    """
    This class extends the plain trainer and adds a buffer for experience replay,
    as well as online test-time adaptation.
    """

    def __init__(self, args: Namespace, main_model: torch.nn.Module, student_model: Optional[torch.nn.Module]):
        """
        Initialize the trainer.

        Args:
            args: Arguments/configuration object.
            main_model: The main model to be trained.
            student_model: Should be None. Asserted in code.
        """
        super().__init__(args, main_model, student_model)
        assert args.student_model['model'] is None, 'Student model exists. This framework is for training without student model.'
        self.laststeps: list = []  # Stores recent batches for online adaptation
        self.buffer: Buffer = Buffer(args.buffer_size, mode=args.mode)  # Experience replay buffer

    def test_step(
        self,
        dataset_object: DatasetCustom,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_x_mark: torch.Tensor,
        batch_y_mark: torch.Tensor,
        model: Optional[torch.nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a test step with online adaptation and buffer replay.

        Args:
            dataset_object: Dataset object for batch processing.
            batch_x: Input batch data.
            batch_y: Target batch data.
            batch_x_mark: Input batch time markers.
            batch_y_mark: Target batch time markers.
            model: Optional model to use (default: None).

        Returns:
            Dictionary containing processed batch outputs.
        """

        # If enough previous steps are stored, perform online adaptation
        if len(self.laststeps) >= self.args.pred_len:
            # Pop the oldest batch from laststeps
            px, py, px_mark, py_mark = self.laststeps.pop(0)
            # Forward pass on previous batch
            prev = self._process_one_batch(dataset_object, px, py, px_mark, py_mark, mode='test')
            # Compute loss between prediction and ground truth
            loss = self.criterion(prev['pred'], self.mainFFN.gt4update(prev))

            # If buffer is not empty, perform experience replay
            if not self.buffer.is_empty():
                bx, by, bx_mark, by_mark, b_old_pred = self.buffer.get_data(self.args.batchReplaySize)
                b_new = self._process_one_batch(dataset_object, bx, by, bx_mark, by_mark, mode='test')
                # Add replay losses
                loss += 0.2 * self.criterion(b_new['pred'], self.mainFFN.gt4update(b_new))
                loss += 0.2 * self.criterion(b_old_pred, b_new['pred'])

            # Backpropagate and update online optimizer
            loss.backward()
            for opt in self.opt['test']['online']:
                opt.step()
            self.store_grad()
            self.update_ocp(prev, prev)
            for opt in self.opt['test']['online']:
                opt.zero_grad()

            # Add the processed batch to the buffer
            self.buffer.add_data(
                batch_x=px,
                batch_y=py,
                batch_x_mark=px_mark,
                batch_y_mark=py_mark,
                prev_out=prev['pred'].data
            )

        # Store current batch for future adaptation
        self.laststeps.append((batch_x, batch_y, batch_x_mark, batch_y_mark))

        # Forward pass on current batch without gradient
        with torch.no_grad():
            curr = self._process_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='test')
        curr = self.clean(curr)

        return curr