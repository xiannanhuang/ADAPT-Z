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

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class TrainerBase:
    """
    Base class for training and testing models.

    Attributes:
        args: Configuration arguments for the trainer.
        model: The model to be trained.
        device: Device to run the computations on (e.g., 'cpu' or 'cuda').
        criterion: Loss function used for training.
        meters: Dictionary to track metrics for training, validation, and testing.
        datalog: Logger to store predictions, ground truths, and metrics.
        scaler: Gradient scaler for mixed precision training.
    """

    def __init__(self, args: Namespace, model: nn.Module) -> None:
        """
        Initializes the TrainerBase.

        Args:
            args: Configuration arguments for the trainer.
            model: The model to be trained.
        """
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        
        # Mixed precision scaler if AMP is enabled
        self.scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
        
        # Select loss function
        self.criterion = nn.MSELoss()

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
        self.datalog = DataLogger(["pred", "true", "mse", "mae"])

    def _select_optimizer(self) -> optim.Optimizer:
        """
        Selects optimizer for training.

        Returns:
            Initialized optimizer.
        """
        if self.args.optimizer.lower() == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.learning_rate
            )
        elif self.args.optimizer.lower() == "adamw":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate
            )
        else:  # Default to SGD
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.learning_rate,
                momentum=0.9
            )
        return optimizer

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
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        mode: str = "train"
    ) -> Dict[str, torch.Tensor]:
        """
        Processes a single batch of data.

        Args:
            batch_x: Input features.
            batch_y: Ground truth labels.
            mode: Mode of operation ('train', 'valid', or 'test').

        Returns:
            Dictionary containing model outputs and true labels.
        """
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        
        # For sequence data, reshape if necessary
        if batch_x.dim() == 2:
            batch_x = batch_x.unsqueeze(2)
        if batch_y.dim() == 2:
            batch_y = batch_y.unsqueeze(2)
            
        # Forward pass
        # x: [Batch, Input length, Channel]
        if self.args.norm_style=='DishTS':
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1)
            outputs = self.model(batch_x, dec_inp)
        else:
            outputs = self.model(batch_x)
        
        return {
            "pred": outputs["pred"] if isinstance(outputs, dict) else outputs,
            "true": batch_y
        }

    def train_step(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Performs a single training step.

        Args:
            batch_x: Input features.
            batch_y: Ground truth labels.

        Returns:
            Dictionary containing the loss value.
        """
        self.optimizer.zero_grad()
        outputs = self._process_one_batch(batch_x, batch_y, "train")
        
        # Compute loss
        loss = self.criterion(outputs["pred"], outputs["true"])
        if self.args.norm_style=='FAN' and self.args.pred_len>1:
            loss2=self.model.nm.loss(outputs["true"])
            loss=loss+loss2
        
        # Backward pass
        if self.args.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return {"loss": loss.item()}

    def valid_step(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Performs a single validation step.

        Args:
            batch_x: Input features.
            batch_y: Ground truth labels.

        Returns:
            Dictionary containing the loss value.
        """
        with torch.no_grad():
            outputs = self._process_one_batch(batch_x, batch_y, "valid")
            loss = self.criterion(outputs["pred"], outputs["true"])
        return {"loss": loss.item()}

    def test_step(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Processes a single test batch.

        Args:
            batch_x: Input features.
            batch_y: Ground truth labels.

        Returns:
            Dictionary containing model outputs and true labels.
        """
        with torch.no_grad():
            outputs = self._process_one_batch(batch_x, batch_y, "test")
        return outputs

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> nn.Module:
        """
        Trains the model using the provided data loaders.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.

        Returns:
            The trained model.
        """
        self.optimizer = self._select_optimizer()
        encountered_nan = False

        path = os.path.join(self.args.checkpoints)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.train_epochs):
            iter_count = 0

            # Training phase
            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, _, _) in enumerate(train_loader):
                iter_count += 1

                # Perform training step
                losses = self.train_step(batch_x, batch_y)
                
                # Check for NaN loss
                if torch.isnan(torch.tensor(losses["loss"])):
                    encountered_nan = True
                    print(f"NaN loss encountered at epoch {epoch}, step {i}")
                    break

                # Update training meters
                for k, v in losses.items():
                    self.meters["train"][k].update(v)

                # Print progress every 100 iterations
                if (i + 1) % 100 == 0:
                    print(self.display_msg(epoch, i, ["train"]))

                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.2f}s")

            # Validation phase
            self.model.eval()
            for i, (batch_x, batch_y, _, _) in enumerate(val_loader):
                losses = self.valid_step(batch_x, batch_y)
                for k, v in losses.items():
                    self.meters["valid"][k].update(v)
                    
            # Back to training mode
            self.model.train()

            # Early stopping check
            early_stopping(self.meters["valid"]["loss"].avg, self.model.state_dict(), path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            # Logging progress
            print(self.display_msg(epoch, train_steps, ["train", "valid"]))

            if self.args.test_run and i > (self.args.pred_len + 50):
                break

        # Load the best model found during training
        self.load(early_stopping.best_path)
        print(f"Loaded best model from {early_stopping.best_path}")
        print(f"Best model restored with validation loss: {early_stopping.best_score:.4f}")

        return self.model

    def test(
        self,
        test_loader: DataLoader,
    ) -> Tuple[
        List[float],
        List[float],
        List[float],
        List[float],
        float
    ]:
        """
        Evaluates the model on the test set.

        Args:
            test_loader: DataLoader for test data.

        Returns:
            Tuple containing test metrics, MAE log, MSE log, predictions, and ground truths.
        """
        self.model.eval()
        start_time = time.time()
        
        # 创建临时字典存储指标
        test_metrics = {
            "mae": [],
            "mse": [],
            "rmse": [],
            "mape": [],
            "mspe": []
        }
        
        # 重置数据记录器
        self.datalog = DataLogger(["pred", "true", "mse", "mae"])

        for i, (batch_x, batch_y, _, _) in enumerate(test_loader):
            outputs = self.test_step(batch_x, batch_y)
            if self.args.data=='nyctaxi':
                outputs['pred']=test_loader.dataset.scaler.inverse_transform(outputs['pred'])
                outputs['true']=test_loader.dataset.scaler.inverse_transform(outputs['true'])
            
            # 计算批处理指标
            metrics = metric(outputs["pred"].cpu().numpy(), outputs["true"].cpu().numpy())
            
            # 更新测试仪表
            for k, v in metrics.items():
                if k in self.meters["test"]:
                    self.meters["test"][k].update(v)
            
            # 存储指标用于后续返回
            for k in test_metrics:
                if k in metrics:
                    test_metrics[k].append(metrics[k])
            
            # 记录预测和真实值
            self.datalog.update({
                "pred": outputs["pred"],
                "true": outputs["true"]
            })
            
            # 记录指标
            self.datalog.update({
                "mse": metrics.get("mse", None),
                "mae": metrics.get("mae", None)
            })

            # 测试运行提前终止
            if self.args.test_run and i > 10:
                print("Test run: Stopping testing early")
                break

        test_time = time.time() - start_time
        
        # 打印最终测试指标
        print("Test Results:")
        for metric_name, meter in self.meters["test"].items():
            print(f"{metric_name.upper()}: {meter.avg:.4f}")
        print(f"Test time: {test_time:.2f} seconds")
        
        # 保存测试结果
        save_path = os.path.join(self.args.checkpoints, f"test_results.pth")
        torch.save({
            "predictions": self.datalog["pred"],
            "targets": self.datalog["true"],
            "metrics": {k: v.avg for k, v in self.meters["test"].items()}
        }, save_path)
        print(f"Test results saved to {save_path}")

        return (
            [v.avg for v in self.meters["test"].values()],
            test_metrics["mae"],
            test_metrics["mse"],
            self.datalog["pred"],
            self.datalog["true"]
        )
        

    def save(self, name: str = "checkpoint.pth") -> None:
        """
        Saves the current model state to disk.

        Args:
            name: Name of the checkpoint file.
        """
        path = os.path.join(self.args.checkpoints, name)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """
        Loads model state from a checkpoint file.

        Args:
            path: Path to the checkpoint file.
        """
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"Model loaded from {path}")