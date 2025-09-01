import os
import time
import warnings
import importlib
from typing import Any, Dict, List, Optional, Tuple, Sequence
from argparse import Namespace
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from dataloader import DatasetCustom 
from torch.utils.data import DataLoader

from utils.tools import EarlyStopping, AverageMeter, DataLogger, Struct, save_model
from utils.metrics import metric
from .forward.trainerBaseForward import TrainerBaseForward as TrainerFFNBase

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class TrainerBase:
    """
    Base class for training and testing models.

    Attributes:
        args: Configuration arguments for the trainer.
        main_model: The main model to be trained.
        student_model: Optional student model for knowledge distillation.
        device: Device to run the computations on (e.g., 'cpu' or 'cuda').
        criterion: Loss function used for training.
        meters: Dictionary to track metrics for training, validation, and testing.
        datalog: Logger to store predictions, ground truths, and metrics.
        scaler: Gradient scaler for mixed precision training.
    """

    def __init__(self, args: Namespace, main_model: nn.Module) -> None:
        """
        Initializes the TrainerBase.

        Args:
            args: Configuration arguments for the trainer.
            main_model: The main model to be trained.
        """
        self.args = args
        self.online = args.online_learning
        self.device = args.device

        # Initialize the main model with given arguments and move to device
        self.main_model = main_model(
            Struct(args).dict2attr("main_model"), args.seq_len
        ).to(self.device)
        # if args.main_model["model"]=='TimesNet':
        #     path= fr'checkpoints2\{args.main_model["model"]}_{args.data}_{args.seq_len}_{args.pred_len}_{args.seed}.pth'
        #     self.main_model.load_state_dict(torch.load(path))
        #     print('load model from ', path)
        # Dynamically import the forward function for the main model
        self.mainFFN: TrainerFFNBase = getattr(
            importlib.import_module(
                f'trainers.forward.trainerForward_{args.main_model["model"]}'
            ),
            "TrainerForward",
        )(args, self.main_model, self.device)

        # Placeholders for student model and its forward function (for distillation)
        self.student_model: Optional[nn.Module] = None
        self.studentFFN: Optional[TrainerFFNBase] = None

        # Mixed precision scaler if AMP is enabled
        self.scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
        # Feature dimension index based on feature type
        self.f_dim = -1 if self.args.features == "MS" else 0

        # Select loss function
        self.criterion = self._select_criterion()

        # Initialize meters for tracking losses and metrics
        self.meters = {
            "train": {"loss": AverageMeter()},
            "valid": {"loss": AverageMeter()},
            "test": {
                "mae": AverageMeter(),
                "mse": AverageMeter(),
                "rmse": AverageMeter(),
                "mape": AverageMeter(),
                "mspe": AverageMeter(),
            },
        }

        # Logger for predictions, ground truths, and metrics
        self.datalog = DataLogger(["pred", "true", "mse", "mae", "embeddings"])

    def _get_optim(self, arguments: Dict[str, Any], params: Any) -> optim.Optimizer:
        """
        Selects and initializes an optimizer.

        Args:
            arguments: Optimizer configuration arguments.
            params: Parameters to optimize.

        Returns:
            Initialized optimizer.
        """
        opt_str = arguments["opt"].lower()
        # Map optimizer string to torch optimizer class
        Optimizer = {
            "adam": optim.Adam,
            "adamw": optim.AdamW,
            "sgd": optim.SGD,
        }[opt_str]

        d = arguments.copy()
        d.pop("opt")
        return Optimizer(params, **d)

    def _select_optimizer(self) -> Dict[str, Any]:
        """
        Selects optimizers for training and testing.

        Returns:
            Dictionary containing optimizers for training and testing.
        """
        opts = {
            "train": [
                self._get_optim(
                    self.args.opt_main["train"], self.main_model.parameters()
                )
            ],
            "test": {
                "batch": [
                    self._get_optim(
                        self.args.opt_main["test"]["batch"],
                        self.main_model.parameters(),
                    )
                ],
                "online": [
                    self._get_optim(
                        self.args.opt_main["test"]["online"],
                        self.main_model.parameters(),
                    )
                ],
            },
        }
        return opts

    def _select_criterion(self) -> nn.Module:
        """
        Selects the loss function.

        Returns:
            Loss function.
        """
        # Default to mean squared error loss
        return nn.MSELoss()

    def update_ocp(self, prev: Dict[str, torch.Tensor], curr: Dict[str, torch.Tensor]) -> None:
        """
        Updates the online change point detection.

        Args:
            prev: Previous state.
            curr: Current state.
        """
        # Update OCP for main and student forward functions
        self.mainFFN.update_ocp(prev, curr)
        if self.studentFFN is not None:
            self.studentFFN.update_ocp(prev, curr)

    def store_grad(self) -> None:
        """
        Stores gradients for the main and student models.
        """
        # Store gradients for main and student forward functions
        self.mainFFN.store_grad()
        if self.studentFFN is not None:
            self.studentFFN.store_grad()

    def display_msg(self, epoch: int, train_steps: int, modes: List[str]) -> str:
        """
        Displays a message with training progress.

        Args:
            epoch: Current epoch.
            train_steps: Current training step.
            modes: List of modes (e.g., 'train', 'valid').

        Returns:
            Formatted message string.
        """
        msg = f"Epoch: {epoch + 1}, Steps: {train_steps + 1} "
        # Append metrics for each mode to the message
        for mode in modes:
            for k, v in self.meters[mode].items():
                msg += f"| {mode} {k}: {v.avg:.4f} "
        return msg

    def _process_one_batch(
        self,
        dataset_object: DatasetCustom,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_x_mark: torch.Tensor,
        batch_y_mark: torch.Tensor,
        mode: str = "train",
    ) -> Dict[str, torch.Tensor]:
        """
        Processes a single batch of data.

        Args:
            dataset_object: Dataset object.
            batch_x: Input features.
            batch_y: Ground truth labels.
            batch_x_mark: Input time markers.
            batch_y_mark: Output time markers.
            mode: Mode of operation ('train', 'valid', or 'test').

        Returns:
            Dictionary containing model outputs.
        """
        # To be implemented in subclasses
        raise NotImplementedError

    def clean(self, outputs: Dict[str, Sequence]) -> Dict[str, Sequence]:
        """
        Cleans the outputs by detaching and moving tensors to CPU.

        Args:
            outputs: Dictionary of model outputs.

        Returns:
            Cleaned outputs.
        """
        # Detach tensors and move to CPU for further processing or logging
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                outputs[key] = value.detach().cpu().numpy()
        return outputs

    def train_step(
        self,
        train_data: DatasetCustom,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_x_mark: torch.Tensor,
        batch_y_mark: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Performs a single training step.

        Args:
            train_data: Training dataset object.
            batch_x: Input features.
            batch_y: Ground truth labels.
            batch_x_mark: Input time markers.
            batch_y_mark: Output time markers.

        Returns:
            Dictionary containing the loss value.
        """
        # Forward pass for one batch
        outputs = self._process_one_batch(
            train_data, batch_x, batch_y, batch_x_mark, batch_y_mark
        )

        # Compute loss
        loss = self.criterion(outputs["pred"], outputs["true"])
        if self.args.use_amp:
            # Mixed precision backward and optimizer step
            self.scaler.scale(loss).backward()
            for opt in self.opt["train"]:
                self.scaler.step(opt)
            self.scaler.update()
        else:
            # Standard backward and optimizer step
            loss.backward()
            for opt in self.opt["train"]:
                opt.step()

        # Store gradients and update OCP
        self.store_grad()
        self.update_ocp(outputs, outputs)

        return {"loss": loss.item()}

    def valid_step(
        self,
        vali_data: DatasetCustom,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_x_mark: torch.Tensor,
        batch_y_mark: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Performs a single validation step.

        Args:
            vali_data: Validation dataset object.
            batch_x: Input features.
            batch_y: Ground truth labels.
            batch_x_mark: Input time markers.
            batch_y_mark: Output time markers.

        Returns:
            Dictionary containing the loss value.
        """
        outputs = self._process_one_batch(
            vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, mode="vali"
        )
        # Compute loss between predicted and true future values
        loss = self.criterion(
            self.mainFFN.get_future(outputs["pred"]).detach().cpu(),
            self.mainFFN.get_future(outputs["true"]).detach().cpu(),
        )
        return {"loss": loss.item()}

    def test_step(
        self,
        dataset_object: DatasetCustom,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_x_mark: torch.Tensor,
        batch_y_mark: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Processes a single test batch.

        Args:
            dataset_object: Test dataset object.
            batch_x: Input features.
            batch_y: Ground truth labels.
            batch_x_mark: Input time markers.
            batch_y_mark: Output time markers.

        Returns:
            Dictionary containing model outputs.
        """
        return self._process_one_batch(
            dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode="test"
        )

    def online_freezing(self) -> None:
        """
        Freezes model parameters for online learning scenarios.

        If online mode is 'regressor', only the encoder is frozen.
        If online mode is 'none', all parameters are frozen.
        """
        if self.online == "regressor":
            for p in self.main_model.encoder.parameters():
                p.requires_grad = False
        elif self.online == "none":
            for p in self.main_model.parameters():
                p.requires_grad = False

    def train(
        self,
        train_data: DatasetCustom,
        train_loader: DataLoader,
        vali_data: DatasetCustom,
        vali_loader: DataLoader,
    ) -> nn.Module:
        """
        Trains the main model using the provided data loaders.

        Args:
            train_data: Training dataset object.
            train_loader: DataLoader for training data.
            vali_data: Validation dataset object.
            vali_loader: DataLoader for validation data.

        Returns:
            The trained main model.
        """
        encountered_nan = False
        self.opt = self._select_optimizer()

        path = os.path.join(self.args.setting, "checkpoints")
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.train_epochs):
            iter_count = 0

            self.main_model.train()
            if self.student_model is not None:
                self.student_model.train()

            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                iter_count += 1
                # Initialize augmenter if present
                if hasattr(self, "augmenter"):
                    if self.augmenter is None:
                        self.get_augmenter(batch_x)

                # Zero gradients for all optimizers
                for opt in self.opt["train"]:
                    opt.zero_grad()

                # Perform a training step
                losses = self.train_step(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark
                )
                # Check for NaN loss
                if losses["loss"] != losses["loss"]:
                    encountered_nan = True

                # Update training meters
                for k, v in losses.items():
                    self.meters["train"][k].update(v)

                # Print progress every 100 iterations
                if (i + 1) % 100 == 0:
                    print(self.display_msg(epoch, i, ["train"]))

                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                if encountered_nan:
                    break
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            # Validation phase
            self.main_model.eval()
            if self.student_model is not None:
                self.student_model.eval()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                losses = self.valid_step(
                    vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark
                )
                for k, v in losses.items():
                    self.meters["valid"][k].update(v)
            self.main_model.train()
            if self.student_model is not None:
                self.student_model.train()

            # Prepare checkpoint content
            ckpt_content = {
                "main_model": self.main_model.state_dict(),
                "student_model": (
                    self.student_model.state_dict()
                    if self.student_model is not None
                    else None
                ),
            }
            
            # Early stopping check
            early_stopping(
                self.meters["valid"][k].avg,
                ckpt_content,
                path,
            )

            if early_stopping.early_stop:
                print("Early stopping")
                break

            # Logging progress
            print(self.display_msg(epoch, train_steps, ["train", "valid"]))

            if self.args.test_run or encountered_nan:
                break

        # Load the best model found during training
        self.load(early_stopping.best_path)

        return self.main_model

    def test(
        self,
        test_data: DatasetCustom,
        test_loader: DataLoader,
    ) -> Tuple[
        List[float],
        float,
        float,
        float,
        float,
    ]:
        """
        Evaluates the model on the test set.

        Args:
            test_data: Test dataset object.
            test_loader: DataLoader for test data.
            trial: Optional trial object for hyperparameter tuning.

        Returns:
            Tuple containing test metrics, MAE log, MSE log, predictions, and ground truths.
        """
        encountered_nan = False

        if self.online == "none":
            self.main_model.eval()

        self.online_freezing()
        start = time.time()
        pbar = tqdm(test_loader)
        start_cpu = 0

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pbar):

            outputs = self.test_step(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark
            )
            if outputs is None:
                continue
            
            # Compute metrics for the batch
            metrics = metric(
                self.mainFFN.get_future(outputs["pred"]),
                self.mainFFN.get_future(outputs["true"]),
            )
            if metrics["mse"] != metrics["mse"]:
                encountered_nan = True

            # Update test meters
            for k, v in metrics.items():
                self.meters["test"][k].update(v)

            # Log metrics
            self.datalog.update({k: metrics[k] for k in metrics.keys()})

            # Special handling for Traffic dataset
            if self.args.data == "Traffic":
                outputs["pred"] = outputs["pred"][:, :, :60]
                outputs["true"] = outputs["true"][:, :, :60]
            self.datalog.update({k: outputs[k] for k in outputs.keys()})

            # Update progress bar
            pbar.set_postfix(
                {"point": metrics["mse"], "cumavg": self.meters["test"]["mse"].avg}
            )
            if self.args.test_run and i > (self.args.pred_len + 50):
                break
            
            if encountered_nan:
                break

        print("test shape:", self.datalog["pred"].shape, self.datalog["true"].shape)
        exp_time = time.time() - start
        
        # Print final test metrics
        print(
            f"mse:{self.meters['test']['mse'].avg}, mae:{self.meters['test']['mae'].avg}, time:{exp_time}"
        )
        
        # Save the test results
        self.save(name=f"s{i}_test_{self.meters['test']['mse'].avg:.4f}.pth")

        return (
            [
                self.meters["test"]["mae"].avg,
                self.meters["test"]["mse"].avg,
                self.meters["test"]["rmse"].avg,
                self.meters["test"]["mape"].avg,
                self.meters["test"]["mspe"].avg,
                exp_time,
            ],
            self.datalog["mae"],
            self.datalog["mse"],
            self.datalog["pred"],
            self.datalog["true"],
        )

    def save(self, name: str = "checkpoint.pth") -> None:
        """
        Saves the current model state to disk.

        Args:
            name: Name of the checkpoint file.
        """
        path = os.path.join(self.args.setting, "checkpoints", name)
        ckpt_content = {
            "main_model": self.main_model.state_dict(),
            "student_model": (
                self.student_model.state_dict()
                if self.student_model is not None
                else None
            ),
        }
        save_model(ckpt_content, path)

    def load(self, path: str) -> None:
        """
        Loads model state from a checkpoint file.

        Args:
            path: Path to the checkpoint file.
        """
        ckpt_content = torch.load(path, map_location=self.device)
        self.main_model.load_state_dict(ckpt_content["main_model"])
        if self.student_model is not None:
            self.student_model.load_state_dict(ckpt_content["student_model"])