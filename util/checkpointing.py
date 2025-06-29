import os
import torch

from typing import Optional, List, Tuple

from models.baseFlow import BaseFlow


def load_checkpoint(model: BaseFlow,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler._LRScheduler,
                    checkpoint_path: str) -> Tuple[int, float]:
    """
    Load model checkpoint
    Args:
        model: model to load weights into
        optimizer: optimizer to load state into
        scheduler: scheduler to load state into
        checkpoint_path: path to checkpoint file
    Returns:
        Tuple of (start_epoch, best_loss)
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} not found")
        return 0, float('inf')

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint.get('best_loss', float('inf'))

    print(f"Resumed from epoch {start_epoch} with best loss {best_loss:.4f}")
    return start_epoch, best_loss


def save_checkpoint(model: BaseFlow,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler._LRScheduler,
                    epoch: int,
                    loss: float,
                    checkpoint_path: str,
                    best_loss: float):
    """Save model checkpoint"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'best_loss': best_loss
    }, checkpoint_path)
