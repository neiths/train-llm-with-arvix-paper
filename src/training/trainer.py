"""
LLM training module with PyTorch.

Supports mixed precision training, gradient accumulation, and distributed training.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config.settings import settings
from src.logging_config import get_logger
from src.training.model import TransformerConfig, TransformerLM
from src.training.wandb_integration import WandBLogger

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    
    # Training
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    
    # Mixed precision
    use_amp: bool = True
    
    # Checkpointing
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    
    # Paths
    output_dir: Path = Path("checkpoints")
    
    @classmethod
    def from_settings(cls) -> "TrainingConfig":
        """Create config from settings."""
        return cls(
            learning_rate=settings.training.learning_rate,
            num_epochs=settings.training.num_epochs,
            batch_size=settings.training.batch_size,
            gradient_accumulation_steps=settings.training.gradient_accumulation_steps,
        )


class TokenDataset(Dataset):
    """Dataset for tokenized training data."""
    
    def __init__(
        self,
        data_path: Path,
        max_length: int = 2048,
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to Parquet data
            max_length: Maximum sequence length
        """
        import pyarrow.parquet as pq
        
        self.max_length = max_length
        
        # Load data
        table = pq.read_table(data_path)
        self.data = table.to_pandas()
        
        logger.info(
            "Loaded dataset",
            num_samples=len(self.data),
            path=str(data_path),
        )
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]
        
        token_ids = row["token_ids"]
        attention_mask = row.get("attention_mask", [1] * len(token_ids))
        
        # Pad or truncate
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        elif len(token_ids) < self.max_length:
            pad_length = self.max_length - len(token_ids)
            token_ids = token_ids + [0] * pad_length
            attention_mask = attention_mask + [0] * pad_length
        
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(token_ids, dtype=torch.long),
        }


class LLMTrainer:
    """
    Trainer for LLM fine-tuning and pre-training.
    
    Features:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Learning rate scheduling with warmup
    - Checkpoint saving and resumption
    - W&B integration
    """
    
    def __init__(
        self,
        model: TransformerLM,
        train_config: TrainingConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        wandb_logger: Optional[WandBLogger] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_config: Training configuration
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            wandb_logger: Optional W&B logger
        """
        self.model = model
        self.config = train_config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.wandb_logger = wandb_logger
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = None  # Will be created in train()
        
        # Setup AMP
        self.scaler = GradScaler() if self.config.use_amp and self.device.type == "cuda" else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            "Trainer initialized",
            device=str(self.device),
            num_parameters=model.num_parameters(),
            use_amp=self.config.use_amp,
        )
    
    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer with weight decay."""
        # Separate parameters that should/shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        return AdamW(param_groups, lr=self.config.learning_rate)
    
    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler with warmup."""
        warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps - warmup_steps,
            eta_min=self.config.learning_rate * 0.1,
        )
        
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
    
    def train(self) -> dict:
        """
        Run training loop.
        
        Returns:
            dict: Training statistics
        """
        # Create data loader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        
        # Calculate total steps
        steps_per_epoch = len(train_loader) // self.config.gradient_accumulation_steps
        total_steps = steps_per_epoch * self.config.num_epochs
        
        # Create scheduler
        self.scheduler = self._create_scheduler(total_steps)
        
        logger.info(
            "Starting training",
            num_epochs=self.config.num_epochs,
            steps_per_epoch=steps_per_epoch,
            total_steps=total_steps,
        )
        
        training_stats = {
            "started_at": datetime.utcnow().isoformat(),
            "epochs": [],
        }
        
        self.model.train()
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_stats = self._train_epoch(train_loader)
            training_stats["epochs"].append(epoch_stats)
            
            # Evaluate
            if self.eval_dataset:
                eval_stats = self.evaluate()
                epoch_stats["eval"] = eval_stats
                
                # Save best model
                if eval_stats["loss"] < self.best_loss:
                    self.best_loss = eval_stats["loss"]
                    self.save_checkpoint("best_model")
            
            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch}")
        
        training_stats["completed_at"] = datetime.utcnow().isoformat()
        training_stats["final_loss"] = training_stats["epochs"][-1]["loss"]
        
        # Save final model
        self.save_checkpoint("final_model")
        
        logger.info("Training complete", **training_stats)
        return training_stats
    
    def _train_epoch(self, train_loader: DataLoader) -> dict:
        """Train for one epoch."""
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass with AMP
            if self.scaler:
                with autocast():
                    outputs = self.model(**batch)
                    loss = outputs["loss"] / self.config.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(**batch)
                loss = outputs["loss"] / self.config.gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                
                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / num_batches
                    lr = self.scheduler.get_last_lr()[0]
                    
                    progress_bar.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        lr=f"{lr:.2e}",
                    )
                    
                    if self.wandb_logger:
                        self.wandb_logger.log({
                            "train/loss": avg_loss,
                            "train/learning_rate": lr,
                            "train/epoch": self.epoch,
                            "train/global_step": self.global_step,
                        })
                
                # Periodic checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}")
        
        avg_loss = total_loss / num_batches
        return {
            "epoch": self.epoch,
            "loss": avg_loss,
            "global_step": self.global_step,
        }
    
    @torch.no_grad()
    def evaluate(self) -> dict:
        """Evaluate the model."""
        if not self.eval_dataset:
            return {}
        
        self.model.eval()
        
        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
        )
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(eval_loader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            if self.config.use_amp:
                with autocast():
                    outputs = self.model(**batch)
            else:
                outputs = self.model(**batch)
            
            total_loss += outputs["loss"].item()
            num_batches += 1
        
        self.model.train()
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        eval_stats = {
            "loss": avg_loss,
            "perplexity": perplexity,
        }
        
        if self.wandb_logger:
            self.wandb_logger.log({
                "eval/loss": avg_loss,
                "eval/perplexity": perplexity,
            })
        
        logger.info("Evaluation complete", **eval_stats)
        return eval_stats
    
    def save_checkpoint(self, name: str) -> Path:
        """Save a training checkpoint."""
        checkpoint_path = self.config.output_dir / name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = checkpoint_path / "model.pt"
        torch.save(self.model.state_dict(), model_path)
        
        # Save optimizer and scheduler
        training_state = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
        }
        torch.save(training_state, checkpoint_path / "training_state.pt")
        
        # Save config
        import json
        with open(checkpoint_path / "config.json", "w") as f:
            json.dump({
                "model_config": self.model.config.__dict__,
                "training_config": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in self.config.__dict__.items()
                },
            }, f, indent=2)
        
        logger.info("Saved checkpoint", path=str(checkpoint_path))
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load a training checkpoint."""
        # Load model
        model_state = torch.load(checkpoint_path / "model.pt", map_location=self.device)
        self.model.load_state_dict(model_state)
        
        # Load training state
        training_state = torch.load(checkpoint_path / "training_state.pt", map_location=self.device)
        self.optimizer.load_state_dict(training_state["optimizer"])
        
        if training_state["scheduler"] and self.scheduler:
            self.scheduler.load_state_dict(training_state["scheduler"])
        
        if training_state["scaler"] and self.scaler:
            self.scaler.load_state_dict(training_state["scaler"])
        
        self.global_step = training_state["global_step"]
        self.epoch = training_state["epoch"]
        self.best_loss = training_state["best_loss"]
        
        logger.info(
            "Loaded checkpoint",
            path=str(checkpoint_path),
            global_step=self.global_step,
            epoch=self.epoch,
        )
