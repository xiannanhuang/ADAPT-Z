from trainers.trainerBase import TrainerBase
import warnings
import torch
from torch import nn
from argparse import Namespace
import numpy as np
from typing import Any, List, Optional, Callable
from dataloader import DatasetCustom 

warnings.filterwarnings("ignore")


class TrainerBaseERTD(TrainerBase):
    """
    Base trainer class for Experience Replay and Temporal Difference (ERTD) learning.
    Implements replay buffer management and temporal difference updates for time series models.
    """

    def __init__(
        self,
        args: Namespace,
        main_model: nn.Module,
        student_model: nn.Module,
    ):
        """
        Initialize the TrainerBaseERTD.

        Args:
            args: Argument object containing configuration.
            main_model: The main neural network model.
            student_model: The student model (not used directly here).
        """
        super().__init__(args, main_model)

        # Buffers for storing recent sequences for experience replay and TD learning
        self.laststepA: List[Optional[torch.Tensor]] = [None, None, None, None]
        self.laststepB: List[Optional[torch.Tensor]] = [None, None, None, None]

        # Replay buffer parameters
        self.replayBufferSize: int = args.replayBufferSize if not args.test_run else 20
        self.batchReplayBufferSize: int = (
            args.batchReplaySize if not args.test_run else min(10, args.batchReplaySize)
        )
        self.num_ERepochs: int = args.num_ERepochs
        self.freq_ERupdate: int = args.freq_ERupdate
        self.count: int = 0
        self.replayBuffer: List[Optional[torch.Tensor]] = [None, None, None, None]

        # Loss function with no reduction (returns per-element loss)
        self.MSEReductNone: nn.MSELoss = nn.MSELoss(reduction="none")

        # Indices for TD steps (k-step TD)
        self.indices: List[int] = [k for k in self.args.td_k if k <= self.args.pred_len]
        self.discountedW: torch.Tensor = []

        # Precompute discounted weights for each TD step
        for idx in self.indices:
            self.discountedW.append(
                torch.cat(
                    [
                        torch.ones(idx - 1),
                        torch.Tensor([self.args.discounted ** i for i in range(self.args.pred_len - idx + 1)]),
                    ]
                )[None, :, None].to(self.device)
            )
        self.discountedW = torch.cat(self.discountedW, dim=0)

    def backward_lossT_train(self, prev: torch.Tensor) -> None:
        """
        Backward pass for training loss (to be implemented in subclass).
        """
        raise NotImplementedError

    def backward_lossT_test(self, prev: torch.Tensor) -> None:
        """
        Backward pass for test loss (to be implemented in subclass).
        """
        raise NotImplementedError

    def backward_lossFT(self, prev: torch.Tensor, curr: torch.Tensor) -> None:
        """
        Backward pass for fine-tuning loss (to be implemented in subclass).
        """
        raise NotImplementedError

    def backward_tdlossFT(self, prev: torch.Tensor, curr: torch.Tensor) -> None:
        """
        Backward pass for TD loss (to be implemented in subclass).
        """
        raise NotImplementedError

    def er(
        self,
        dataset_object: DatasetCustom,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_x_mark: torch.Tensor,
        batch_y_mark: torch.Tensor,
    ) -> None:
        """
        Experience Replay (ER) mechanism. Maintains a replay buffer (laststepA) and periodically
        samples from it to perform additional updates.

        Args:
            dataset_object: Dataset object for batch processing.
            batch_x: Input features for the current batch.
            batch_y: Target values for the current batch.
            batch_x_mark: Additional input markers for the current batch.
            batch_y_mark: Additional target markers for the current batch.
        """
        # Helper function to slice tensors for replay buffer
        g: Callable[[torch.Tensor, int, int], torch.Tensor] = lambda x, index, i: (
            x[:, index : index + self.args.pred_len]
            if i % 2
            else x[:, index : index + self.args.seq_len]
        )

        # If laststepA (buffer) is initialized and has enough data
        if (
            self.laststepA[0] is not None
            and (self.laststepA[0].shape[1] - self.args.seq_len + 1) >= self.args.pred_len
        ):
            # Save the first element of laststepA for each buffer
            itemA = [g(x, 0, i) for i, x in enumerate(self.laststepA)]

            # Remove the first element from laststepA
            self.laststepA = [x[:, 1:] for x in self.laststepA]

            # Enqueue the first element into the replay buffer
            self.replayBuffer = [
                (torch.cat([bufferT, newT[:, -1:]], dim=1) if bufferT is not None else newT)
                for bufferT, newT in zip(self.replayBuffer, itemA)
            ]

            # Dequeue the oldest element if buffer is too large
            if self.replayBuffer[0].shape[1] - self.args.seq_len + 1 > self.replayBufferSize:
                self.replayBuffer = [bufferT[:, 1:] for bufferT in self.replayBuffer]

            # If enough sequences in the buffer, sample and update
            if (N_buffer := self.replayBuffer[0].shape[1] - self.args.seq_len + 1) >= self.batchReplayBufferSize:
                if self.count % self.freq_ERupdate == 0:
                    for _ in range(self.num_ERepochs):
                        # Randomly sample indices for replay
                        randomSampleIndex = np.random.choice(
                            N_buffer, self.batchReplayBufferSize, replace=False
                        )

                        # Prepare sampled batches
                        pA_x, pA_y, pA_x_mark, pA_y_mark = [
                            torch.cat(
                                [g(x, index, j) for index in randomSampleIndex], dim=0
                            )
                            for j, x in enumerate(self.replayBuffer)
                        ]

                        # Process batch and perform backward passes
                        prevA = self._process_one_batch(
                            dataset_object,
                            pA_x,
                            pA_y,
                            pA_x_mark,
                            pA_y_mark,
                            mode="test",
                        )

                        self.backward_lossT_test(prevA)
                        self.backward_lossFT(prevA, prevA)

                        for opt in self.opt["test"]["batch"]:
                            opt.step()
                        self.store_grad()
                        self.update_ocp(prevA, prevA)
                        for opt in self.opt["test"]["batch"]:
                            opt.zero_grad()
                        self.clean(prevA)

                self.count = (self.count + 1) % self.freq_ERupdate

        # Add the latest data to laststepA
        curr_data = [batch_x, batch_y, batch_x_mark, batch_y_mark]
        self.laststepA = [
            torch.cat([lastT, currT[:, -1:]], dim=1) if lastT is not None else currT
            for lastT, currT in zip(self.laststepA, curr_data)
        ]

    def td(
        self,
        dataset_object: DatasetCustom,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_x_mark: torch.Tensor,
        batch_y_mark: torch.Tensor,
    ) -> None:
        """
        Temporal Difference (TD) learning mechanism. Maintains a buffer (laststepB) for TD updates
        and performs online updates using TD.

        Args:
            dataset_object: Dataset object for batch processing.
            batch_x: Input features for the current batch.
            batch_y: Target values for the current batch.
            batch_x_mark: Additional input markers for the current batch.
            batch_y_mark: Additional target markers for the current batch.
        """
        b, t, d = batch_y.shape

        # Helper function to slice tensors for TD buffer
        g: Callable[[torch.Tensor, int, int], torch.Tensor] = lambda x, index, i: (
            torch.cat(
                (x[:, -index:], torch.zeros((b, t - index, x.shape[2]), device=x.device)),
                dim=1,
            )
            if (i % 2 == 1)
            else x[:, -(index + self.args.seq_len - 1) : -(index - 1) if index != 1 else None, :]
        )

        # If laststepB (buffer) is initialized and has enough data for TD
        if self.laststepB[0] is not None and (
            self.laststepB[0].shape[1] - self.args.seq_len + 1 >= max(self.indices)
        ):
            # Prepare TD batches for each k in indices
            pB_x, pB_y, pB_x_mark, pB_y_mark = [
                torch.cat([g(x, index, j) for index in self.indices], dim=0)
                for j, x in enumerate(self.laststepB)
            ]

            # Previous prediction
            prevB = self._process_one_batch(
                dataset_object, pB_x, pB_y, pB_x_mark, pB_y_mark, mode="test"
            )

            # Current prediction (no grad)
            with torch.no_grad():
                curr = self._process_one_batch(
                    dataset_object,
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_y_mark,
                    mode="test",
                )
            self.backward_tdlossFT(prevB, curr)

            for opt in self.opt["test"]["online"]:
                opt.step()
            self.store_grad()
            self.update_ocp(prevB, curr)
            for opt in self.opt["test"]["online"]:
                opt.zero_grad()

            self.clean(curr)
            self.clean(prevB)

            # Remove the oldest element from laststepB
            self.laststepB = [x[:, 1:] for x in self.laststepB]

        # Append the latest element to laststepB
        if self.laststepB[0] is None:
            self.laststepB = [
                batch_x,
                batch_y[:, 0:1],
                batch_x_mark,
                batch_y_mark[:, 0:1],
            ]
            return

        self.laststepB = [
            torch.cat([lastT, currT[:, 0:1]], dim=1)
            for lastT, currT in zip(
                self.laststepB,
                [
                    batch_x[:, -1:],
                    batch_y[:, 0:1],
                    batch_x_mark[:, -1:],
                    batch_y_mark[:, 0:1],
                ],
            )
        ]