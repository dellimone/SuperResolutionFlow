import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block"""

    def __init__(self, channels: int, dropout: float = 0.0, use_group_norm: bool = True):
        super().__init__()

        norm_layer = nn.GroupNorm(8, channels) if use_group_norm else nn.BatchNorm2d(channels)
        activation = nn.SiLU()

        self.block = nn.Sequential(
            norm_layer,
            activation,
            nn.Conv2d(channels, channels, 3, padding=1),
            activation,
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.GroupNorm(8, channels) if use_group_norm else nn.BatchNorm2d(channels),
            activation,
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ConditioningNetwork(nn.Module):
    """Unified conditioning network for generating initial upsampled image"""

    def __init__(self,
                 lr_channels: int = 3,
                 hr_channels: int = 3,
                 scale_factor: int = 2,
                 hidden_dim: int = 64,
                 num_residual_blocks: int = 6,
                 residual_weight: float = 0.1):
        super().__init__()

        assert scale_factor in (2, 4), f"Scale factor must be 2 or 4, got {scale_factor}"
        self.scale_factor = scale_factor
        self.residual_weight = residual_weight

        # Initial feature extraction
        self.initial_conv = nn.Conv2d(lr_channels, hidden_dim, 3, padding=1)

        # Residual blocks for feature extraction
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)
        ])

        # Progressive upsampling
        self.upsampler = self._build_upsampler(hidden_dim, hr_channels, scale_factor)

    def _build_upsampler(self, hidden_dim: int, out_channels: int, scale_factor: int) -> nn.Module:
        """Build upsampling layers based on scale factor"""
        layers = []

        if scale_factor == 2:
            layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(hidden_dim // 2, out_channels, 3, padding=1)
            ])
        elif scale_factor == 4:
            layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim * 4, 3, padding=1),
                nn.PixelShuffle(2),  # First 2x
                nn.SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim * 4, 3, padding=1),
                nn.PixelShuffle(2),  # Second 2x
                nn.SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(hidden_dim // 2, out_channels, 3, padding=1)
            ])

        return nn.Sequential(*layers)

    def forward(self, lr_img: torch.Tensor) -> torch.Tensor:
        # Generate bicubic baseline
        hr_bicubic = F.interpolate(
            lr_img, scale_factor=self.scale_factor,
            mode='bicubic', align_corners=False
        )

        # Extract features
        x = self.initial_conv(lr_img)

        for block in self.residual_blocks:
            x = block(x)

        x = self.upsampler(x)

        # Combine with bicubic baseline
        return torch.tanh(x) * self.residual_weight + hr_bicubic