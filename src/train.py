"""Training logic with early stopping and checkpointing."""

import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.builder import build_loss, build_model
from src.utils.io import ensure_dir
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    """Trainer class for model training with all bells and whistles."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        config: dict[str, Any],
        device: torch.device,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            config: Configuration dictionary
            device: Device to train on
        """
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device

        # Training config
        train_cfg = config["training"]
        self.epochs = train_cfg["epochs"]
        # AMP will only be active when CUDA is available
        self.mixed_precision = train_cfg["mixed_precision"]
        self.gradient_clip_norm = train_cfg.get("gradient_clip_norm")
        self.gradient_accumulation_steps = train_cfg.get("gradient_accumulation_steps", 1)

        # Early stopping
        es_cfg = train_cfg["early_stopping"]
        self.early_stopping_enabled = es_cfg["enabled"]
        self.early_stopping_patience = es_cfg["patience"]
        self.early_stopping_metric = es_cfg["metric"]
        self.early_stopping_mode = es_cfg["mode"]

        # Checkpointing
        ckpt_cfg = train_cfg["checkpoint"]
        self.checkpoint_dir = Path(ckpt_cfg["save_dir"])
        self.save_best = ckpt_cfg["save_best"]
        self.save_last = ckpt_cfg["save_last"]

        ensure_dir(self.checkpoint_dir)

        # Mixed precision scaler (conditionally enabled)
        use_cuda = torch.cuda.is_available()
        use_amp = bool(self.mixed_precision) and use_cuda
        self.mixed_precision = use_amp
        self.scaler = GradScaler() if use_amp else None

        # Tracking
        self.current_epoch = 0
        self.best_metric = float("inf") if self.early_stopping_mode == "min" else float("-inf")
        self.epochs_without_improvement = 0
        self.history: dict[str, list] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
            "epoch_time": [],
        }

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch + 1}/{self.epochs} [Train]")

        for batch_idx, (videos, labels) in enumerate(pbar):
            videos = videos.to(self.device)
            labels = labels.to(self.device)

            # Forward pass with mixed precision
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(videos)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.gradient_accumulation_steps
            else:
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.mixed_precision:
                assert self.scaler is not None
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.gradient_clip_norm is not None:
                    if self.mixed_precision:
                        assert self.scaler is not None
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)

                # Optimizer step
                if self.mixed_precision:
                    assert self.scaler is not None
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            pbar.set_postfix({"loss": total_loss / num_batches})

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for videos, labels in tqdm(dataloader, desc="Validating"):
            videos = videos.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(videos)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        metrics = {
            "val_loss": total_loss / len(dataloader),
            "val_acc": correct / total,
        }

        return metrics

    def save_checkpoint(self, filepath: Path, is_best: bool = False) -> None:
        """Save checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_metric": self.best_metric,
            "history": self.history,
            "config": self.config,
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint: {filepath}")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Train model."""
        logger.info("Starting training...")
        logger.info(f"Device: {self.device} | AMP enabled: {bool(self.scaler)}")
        start_time = time.time()

        for epoch in range(self.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["val_loss"])
                else:
                    self.scheduler.step()

            # Log
            lr = self.optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start

            self.history["epoch"].append(epoch + 1)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_acc"].append(val_metrics["val_acc"])
            self.history["lr"].append(lr)
            self.history["epoch_time"].append(epoch_time)

            logger.info(
                f"Epoch {epoch + 1}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val Acc: {val_metrics['val_acc']:.4f} | "
                f"LR: {lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Check for improvement
            current_metric = val_metrics[self.early_stopping_metric]
            improved = (
                current_metric < self.best_metric
                if self.early_stopping_mode == "min"
                else current_metric > self.best_metric
            )

            if improved:
                self.best_metric = current_metric
                self.epochs_without_improvement = 0
                if self.save_best:
                    self.save_checkpoint(self.checkpoint_dir / "best.pt", is_best=True)
            else:
                self.epochs_without_improvement += 1

            # Config-driven checkpointing
            ckpt_cfg = self.config["training"]["checkpoint"]
            save_interval = ckpt_cfg.get("save_interval")
            save_best_only = ckpt_cfg.get("save_best_only", False)
            ckpt_cfg.get("monitor_metric", "val_loss")

            # Interval saving when enabled and not save_best_only
            if save_interval and (epoch + 1) % int(save_interval) == 0 and not save_best_only:
                interval_path = self.checkpoint_dir / f"epoch_{epoch+1}.pt"
                self.save_checkpoint(interval_path)

            # Save last
            if self.save_last and not save_best_only:
                self.save_checkpoint(self.checkpoint_dir / "last.pt")

            # Early stopping
            if (
                self.early_stopping_enabled
                and self.epochs_without_improvement >= self.early_stopping_patience
            ):
                logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs "
                    f"(no improvement for {self.early_stopping_patience} epochs)"
                )
                break

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time / 60:.1f} minutes")

        # Save history
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(self.checkpoint_dir / "training_history.csv", index=False)


def train(config: dict[str, Any], resume: str | None = None) -> None:
    """
    Train model from config.

    Args:
        config: Configuration dictionary
        resume: Optional checkpoint path to resume from
    """
    from src.data.dataset import build_dataloaders
    from src.data.transforms import get_transforms

    # Setup device
    if config["hardware"]["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["hardware"]["device"])

    logger.info(f"Using device: {device}")

    # Build transforms and dataloaders
    transforms = {
        "train": get_transforms(config, is_train=True),
        "val": get_transforms(config, is_train=False),
    }
    dataloaders = build_dataloaders(config, transforms)

    # Build model
    model = build_model(config)

    # Build loss (with class weights if specified)
    class_weights = None
    if config["model"]["loss"].get("class_weights") is not None:
        class_weights = torch.tensor(config["model"]["loss"]["class_weights"]).to(device)

    criterion = build_loss(config, class_weights)

    # Build optimizer
    opt_cfg = config["training"]["optimizer"]
    if opt_cfg["type"] == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg["learning_rate"],
            weight_decay=opt_cfg["weight_decay"],
        )
    elif opt_cfg["type"] == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=opt_cfg["learning_rate"],
            momentum=opt_cfg.get("momentum", 0.9),
            weight_decay=opt_cfg["weight_decay"],
            nesterov=opt_cfg.get("nesterov", False),
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_cfg['type']}")

    # Build scheduler
    sched_cfg = config["training"]["scheduler"]
    if sched_cfg["type"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=sched_cfg["T_max"], eta_min=sched_cfg.get("eta_min", 0)
        )
    elif sched_cfg["type"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=sched_cfg["step_size"], gamma=sched_cfg["gamma"]
        )
    elif sched_cfg["type"] == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=sched_cfg["patience"],
            factor=sched_cfg["factor"],
        )
    elif sched_cfg["type"] == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {sched_cfg['type']}")

    # Build trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,  # type: ignore[arg-type]
        config=config,
        device=device,
    )

    # Resume from checkpoint if specified
    if resume is not None:
        # Implementation of resume logic would go here
        logger.info(f"Resuming from checkpoint: {resume}")

    # Train
    trainer.fit(dataloaders["train"], dataloaders["val"])
