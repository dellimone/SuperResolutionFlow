import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple


class BaseFlow(nn.Module, ABC):
    """Abstract base class for all flow-based super-resolution models"""

    def __init__(self,
                 hr_channels: int = 3,
                 lr_channels: int = 3,
                 scale_factor: int = 2):
        super().__init__()
        self.hr_channels = hr_channels
        self.lr_channels = lr_channels
        self.scale_factor = scale_factor

    @abstractmethod
    def forward(self, hr_img: torch.Tensor, lr_img: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass - implementation depends on flow type"""
        pass

    @abstractmethod
    def sample(self, lr_img: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate high-resolution samples from low-resolution input"""
        pass


    def get_hr_shape(self, lr_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute HR shape from LR shape"""
        *batch_dims, c, h, w = lr_shape
        return (*batch_dims, self.hr_channels, h * self.scale_factor, w * self.scale_factor)

