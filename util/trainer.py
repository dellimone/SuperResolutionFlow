import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.utils import save_image
from tqdm import tqdm
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

from models.flowMatchingSR import FlowMatchingSR
from models.normalizingFlowSR import NormalizingFlowSR
from util.config import NormalizingFlowSRConfig, FlowMatchingSRConfig
from util.dataset import SuperResolutionDataset


class BaseSRTrainer(ABC):
    """
    Base class for Super-Resolution model training
    """

    def __init__(self, config, model_class, model_kwargs: Dict[str, Any]):
        """
        Initialize the base trainer.

        Args:
            config: Training configuration object
            model_class: Model class to instantiate
            model_kwargs: Keyword arguments for model initialization
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training on device: {self.device}")

        # Initialize model
        self.model = model_class(**model_kwargs).to(self.device)

        # Initialize optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        # Setup data loaders
        self.train_loader, self.val_loader = self._setup_data_loaders()

        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')

        # Resume from checkpoint if specified
        if self.config.resume_from is not None:
            self._load_checkpoint(config.resume_from)

        # Print model info
        self._print_model_info()

    def _print_model_info(self):
        """Print model and dataset information."""
        param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"Model parameters: {param_count:.2f}M")
        print(f"Training on {len(self.train_loader.dataset)} images, "
              f"validating on {len(self.val_loader.dataset)} images")

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """
        Setup optimizer based on config.

        Returns:
            Configured optimizer

        Raises:
            ValueError: If optimizer type is not supported
        """
        optimizer_type =self.config.optimizer_type

        if optimizer_type == 'AdamW':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=getattr(self.config, 'weight_decay', 1e-4),
                betas=getattr(self.config, 'betas', (0.9, 0.999)),
                eps=getattr(self.config, 'eps', 1e-8)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    def _setup_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Setup learning rate scheduler based on config.

        Returns:
            Configured learning rate scheduler

        Raises:
            ValueError: If scheduler type is not supported
        """
        scheduler_type = self.config.lr_scheduler_type

        if scheduler_type == "onecycle":
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=self.config.num_epochs,
                pct_start=0.1,
                anneal_strategy='cos'
            )
        else:
            raise ValueError(f"Unknown learning rate scheduler: {scheduler_type}")

    def _setup_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Setup training and validation data loaders.

        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Load full dataset with augmentation for training
        full_dataset = SuperResolutionDataset(
            root_dir=self.config.root_dir,
            hr_size=self.config.hr_size,
            scale_factor=self.config.scale_factor,
            augment=True
        )

        # Split dataset for training/validation
        train_split = getattr(self.config, 'train_split', 0.8)
        train_size = int(train_split * len(full_dataset))
        val_size = len(full_dataset) - train_size

        train_dataset, val_indices = random_split(
            full_dataset, [train_size, val_size])

        # Create validation dataset without augmentation
        val_dataset_no_aug = SuperResolutionDataset(
            root_dir=self.config.root_dir,
            hr_size=self.config.hr_size,
            scale_factor=self.config.scale_factor,
            augment=False
        )
        val_dataset = Subset(val_dataset_no_aug, val_indices.indices)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            drop_last=True  # Ensure consistent batch sizes
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        return train_loader, val_loader

    @abstractmethod
    def compute_loss(self, hr: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        """
        Compute the training loss for the specific model type.

        Args:
            hr: High-resolution ground truth images
            lr: Low-resolution input images

        Returns:
            Computed loss tensor
        """
        pass

    @abstractmethod
    def _sample(self, lr: torch.Tensor) -> torch.Tensor:
        pass

    def evaluate_model(self, data_loader: DataLoader) -> Tuple[float, Dict]:
        """
        Evaluate normalizing flow model on given dataset split.
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, (hr, lr) in enumerate(data_loader):
                hr, lr = hr.to(self.device), lr.to(self.device)
                loss = self.compute_loss(hr, lr)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def _save_output(self, train_dataloader: DataLoader, val_dataloader: DataLoader, epoch: int) -> None:
        for split, dataloader in zip(['train', 'val'], [train_dataloader, val_dataloader]):
            # Create output dir if not exist
            eval_dir = os.path.join(self.config.output_dir, f"eval_{split}_epoch{epoch + 1}")
            os.makedirs(eval_dir, exist_ok=True)
            for i, (hr, lr) in enumerate(dataloader):
                if i > self.config.num_samples:
                    break
                save_image(lr, os.path.join(eval_dir, f"sample_{i}_LR.png"))
                save_image(hr, os.path.join(eval_dir, f"sample_{i}_HR.png"))
                save_image(self._sample(lr), os.path.join(eval_dir, f"sample_{i}_sample.png"))

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        if is_best:
            path_name = f"best_model.pth"
        else:
            path_name = f"model_at_epoch{epoch + 1}.pth"

        checkpoint_path = os.path.join(self.config.checkpoint_dir, path_name)

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config)
        }
        torch.save(checkpoint, checkpoint_path)

    def _load_checkpoint(self, path: str):
        """
        Load model checkpoint and update training state.

        Args:
            path: Path to checkpoint file
        """
        if not os.path.exists(path):
            print(f"Warning: Checkpoint file {path} not found.")
            return

        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

            print(f"Resumed from epoch {self.start_epoch}, best val loss: {self.best_val_loss:.4f}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")

    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs} [Train]")

        for batch_idx, (hr, lr) in enumerate(pbar):
            hr, lr = hr.to(self.device), lr.to(self.device)

            self.optimizer.zero_grad()
            loss = self.compute_loss(hr, lr)
            loss.backward()

            if self.config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                'Avg': f'{total_loss / num_batches:.4f}'
            })
        return total_loss / num_batches if num_batches > 0 else float('inf')

    def train(self):
        """Main training loop."""
        print("Starting training...")

        for epoch in range(self.start_epoch, self.config.num_epochs):
            avg_train_loss = self.train_epoch(epoch)
            self.scheduler.step()
            val_loss = self.evaluate_model(self.val_loader)

            print(
                f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )

            # Save best model
            if val_loss < self.best_val_loss:
                print(f"New best model validation loss = {val_loss:.4f}")
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, is_best=True)

            # Periodic checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(epoch)

            # Save outputs
            if (epoch + 1) % self.config.eval_every == 0:
                self._save_output(self.train_loader, self.val_loader, epoch)

        print("Training completed successfully!")
        return self.best_val_loss


class NormalizingFlowTrainer(BaseSRTrainer):
    """
    Trainer for Normalizing Flow Super-Resolution models.
    """

    def __init__(self, config: NormalizingFlowSRConfig):
        """
        Initialize the normalizing flow trainer.
        """
        model_kwargs = {
            'hr_channels': config.hr_channels,
            'lr_channels': config.lr_channels,
            'scale_factor': config.scale_factor,
            'num_flows': config.num_flows,
            'cond_net_base_channels': config.cond_net_base_channels,
            'cond_net_res_block_num': config.cond_net_res_block_num,
            'cond_net_res_weight': config.cond_net_res_weight,
        }
        super().__init__(config, NormalizingFlowSR, model_kwargs)


    def compute_loss(self, hr: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood loss for normalizing flows between lr and hr.
        """
        log_likelihood = self.model.log_likelihood(hr, lr)
        return -log_likelihood.mean()

    def _sample(self,lr: torch.Tensor) -> Dict:
        """
        Generate samples normalizing flows.
        """
        return self.model.sample(lr)


class FlowMatchingTrainer(BaseSRTrainer):
    """
    Trainer for Flow Matching Super-Resolution.
    """

    def __init__(self, config: FlowMatchingSRConfig):
        """
        Initialize the flow matching trainer.
        """
        model_kwargs = {
            'hr_channels': config.hr_channels,
            'lr_channels': config.lr_channels,
            'scale_factor': config.scale_factor,
            'cond_net_base_channels': config.cond_net_base_channels,
            'cond_net_res_block_num': config.cond_net_res_block_num,
            'cond_net_res_weight': config.cond_net_res_weight,
            'unet_base_channels': config.unet_base_channels,
            'unet_channel_mult': config.unet_channel_mult,
            'unet_attention': config.unet_attention,
            'unet_num_res_block': config.unet_num_res_block,
            'alpha': config.alpha,
            'beta': config.beta,
            'l1w': config.l1w,
            'l2w': config.l2w
        }
        super().__init__(config, FlowMatchingSR, model_kwargs)

    def compute_loss(self, hr: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        """
        Compute flow matching loss using hr and lr.
        """
        predicted_velocity, target_velocity, _ = self.model(hr, lr, training=True)
        return self.model.flow_matching_loss(predicted_velocity, target_velocity)


    def _sample(self,lr: torch.Tensor) -> Dict:
        return self.model.sample(lr, self.config.integration_step)



