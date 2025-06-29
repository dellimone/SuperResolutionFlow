import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
import math
from typing import Tuple, List, Optional

from .baseFlow import BaseFlow
from .conditioningNetwork import ConditioningNetwork


# UNET

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -embeddings)
        embeddings = t[:, None].float() * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=1)

        if self.dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=1)

        return embeddings


class TimeResidualBlock(nn.Module):
    """Residual block with time embedding injection"""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float = 0.0):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        self.skip_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)

        # Inject time embedding
        t_proj = self.time_proj(t_emb)
        while len(t_proj.shape) < len(h.shape):
            t_proj = t_proj.unsqueeze(-1)
        h = h + t_proj

        h = self.conv2(h)
        return h + self.skip_conv(x)


class SelfAttentionBlock(nn.Module):
    """Self-attention for capturing long-range dependencies"""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for attention
        q = q.view(B, C, H * W).transpose(1, 2)
        k = k.view(B, C, H * W)
        v = v.view(B, C, H * W).transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(C)
        attn = torch.softmax(torch.bmm(q, k) * scale, dim=-1)
        h = torch.bmm(attn, v)

        h = h.transpose(1, 2).view(B, C, H, W)
        h = self.proj(h)

        return x + h


class FlowMatchingUNet(nn.Module):
    """U-Net architecture for flow matching"""

    def __init__(self,
                 channels: int = 3,
                 conditioning_channels: int = 3,
                 time_dim: int = 128,
                 base_channels: int = 128,
                 channel_multipliers: List[int] = [1, 2, 2, 4],
                 use_attention: List[bool] = [False, False, True, True],
                 num_res_blocks: int = 2,
                 dropout: float = 0.0):
        super().__init__()

        self.time_dim = time_dim

        # Time embedding
        self.time_embedding = TimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim * 4)
        )

        # Input projection
        self.input_conv = nn.Conv2d(channels + conditioning_channels, base_channels, 3, padding=1)

        # Build encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downsample = nn.ModuleList()

        ch = base_channels
        input_ch = ch

        for level, mult in enumerate(channel_multipliers):
            output_ch = base_channels * mult

            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(TimeResidualBlock(input_ch, output_ch, time_dim * 4, dropout))
                input_ch = output_ch

                if use_attention[level]:
                    level_blocks.append(SelfAttentionBlock(output_ch))

            self.encoder_blocks.append(level_blocks)

            if level < len(channel_multipliers) - 1:
                self.encoder_downsample.append(nn.Conv2d(output_ch, output_ch, 3, stride=2, padding=1))
            else:
                self.encoder_downsample.append(nn.Identity())

        # Middle blocks
        mid_ch = base_channels * channel_multipliers[-1]
        self.middle_blocks = nn.ModuleList([
            TimeResidualBlock(mid_ch, mid_ch, time_dim * 4, dropout),
            SelfAttentionBlock(mid_ch),
            TimeResidualBlock(mid_ch, mid_ch, time_dim * 4, dropout)
        ])

        # Build decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_upsample = nn.ModuleList()

        for level, mult in enumerate(reversed(channel_multipliers)):
            output_ch = base_channels * mult

            if level > 0:
                self.decoder_upsample.append(nn.ConvTranspose2d(input_ch, input_ch, 4, stride=2, padding=1))
            else:
                self.decoder_upsample.append(nn.Identity())

            skip_ch = base_channels * mult

            level_blocks = nn.ModuleList()
            for i in range(num_res_blocks + 1):
                if i == 0:
                    level_blocks.append(TimeResidualBlock(input_ch + skip_ch, output_ch, time_dim * 4, dropout))
                else:
                    level_blocks.append(TimeResidualBlock(output_ch, output_ch, time_dim * 4, dropout))

                use_attn_level = len(channel_multipliers) - 1 - level
                if use_attention[use_attn_level]:
                    level_blocks.append(SelfAttentionBlock(output_ch))

            self.decoder_blocks.append(level_blocks)
            input_ch = output_ch

        # Output projection
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, channels, 3, padding=1)
        )

        # Zero initialization for stability
        nn.init.zeros_(self.output_conv[-1].weight)
        nn.init.zeros_(self.output_conv[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        # Process time embedding
        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)

        # Combine input with conditioning
        h = torch.cat([x, conditioning], dim=1)
        h = self.input_conv(h)

        # Encoder with skip connections
        skip_connections = []
        for level_blocks, downsample in zip(self.encoder_blocks, self.encoder_downsample):
            for block in level_blocks:
                if isinstance(block, TimeResidualBlock):
                    h = block(h, t_emb)
                else:
                    h = block(h)

            skip_connections.append(h)
            h = downsample(h)

        # Middle processing
        for block in self.middle_blocks:
            if isinstance(block, TimeResidualBlock):
                h = block(h, t_emb)
            else:
                h = block(h)

        # Decoder with skip connections
        for level_blocks, upsample in zip(self.decoder_blocks, self.decoder_upsample):
            h = upsample(h)
            skip = skip_connections.pop()
            h = torch.cat([h, skip], dim=1)

            for block in level_blocks:
                if isinstance(block, TimeResidualBlock):
                    h = block(h, t_emb)
                else:
                    h = block(h)

        return self.output_conv(h)

# Flow matching

class RectifiedFlow:
    """Rectified Flow for optimal transport between distributions"""

    @staticmethod
    def compute_path(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute straight line path between source and target"""
        t = t.view(-1, 1, 1, 1)
        return (1 - t) * x0 + t * x1

    @staticmethod
    def compute_velocity(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute target velocity field for straight line transport"""
        return x1 - x0


class FlowMatchingSR(BaseFlow):
    """Flow Matching Super-Resolution model"""

    def __init__(self,
                 hr_channels: int = 3,
                 lr_channels: int = 3,
                 scale_factor: int = 2,
                 cond_net_base_channels: int = 64,
                 cond_net_res_block_num: int = 6,
                 cond_net_res_weight: float = 0.1,
                 unet_base_channels: int = 128,
                 unet_channel_mult: List[int] =[1, 2, 2, 4],
                 unet_attention: List[bool] =[False, False, True, True],
                 unet_num_res_block=2,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 l1w: float = 1.0,
                 l2w: float = 0.1,
                 ):
        super().__init__(hr_channels, lr_channels, scale_factor)

        self.alpha = alpha
        self.beta = beta

        self.l1w = l1w
        self.l2w = l2w

        # Conditioning network
        self.conditioning = ConditioningNetwork(
            lr_channels=lr_channels,
            hr_channels=hr_channels,
            scale_factor=scale_factor,
            hidden_dim=cond_net_base_channels,
            num_residual_blocks=cond_net_res_block_num,
            residual_weight=cond_net_res_weight,)

        # Flow matching U-Net
        self.flow_net = FlowMatchingUNet(
            channels=hr_channels,
            conditioning_channels=hr_channels,
            base_channels=unet_base_channels,
            channel_multipliers=unet_channel_mult,
            use_attention = unet_attention,
            num_res_blocks=unet_num_res_block,
            dropout=0.0
        )

        self.rectified_flow = RectifiedFlow()

    def forward(self, hr_img: torch.Tensor, lr_img: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, ...]:
        """Forward pass for training or inference"""
        batch_size = hr_img.shape[0]
        device = hr_img.device

        # Generate conditioning
        conditioning = self.conditioning(lr_img)

        if training:
            # Sample time from Beta distribution
            alpha = torch.tensor(self.alpha, device=device)
            beta = torch.tensor(self.beta, device=device)
            t = Beta(alpha, beta).sample((batch_size,))

            # Sample noise and compute path
            x0 = torch.randn_like(hr_img)
            xt = self.rectified_flow.compute_path(x0, hr_img, t)
            target_velocity = self.rectified_flow.compute_velocity(x0, hr_img, t)

            # Predict velocity
            predicted_velocity = self.flow_net(xt, t, conditioning)

            return predicted_velocity, target_velocity, conditioning
        else:
            return self.sample(lr_img)

    def sample(self, lr_img: torch.Tensor, num_steps: int = 50) -> torch.Tensor:
        """Generate high-resolution images"""
        batch_size = lr_img.shape[0]
        device = lr_img.device

        # Get conditioning
        conditioning = self.conditioning(lr_img)

        # Initialize with noise
        x = torch.randn(*self.get_hr_shape(lr_img.shape), device=device)

        dt = 1.0 / num_steps

        with torch.no_grad():
            for i in range(num_steps):
                t = torch.full((batch_size,), i / num_steps, device=device)
                velocity = self.flow_net(x, t, conditioning)
                x = x + dt * velocity

        return torch.clamp(x, 0, 1)

    def loss(self, hr_img: torch.Tensor, lr_img: torch.Tensor, training):
        predicted_velocity, target_velocity, conditioning = self(hr_img, lr_img, training=training)
        return self.flow_matching_loss(predicted_velocity, target_velocity)

    def flow_matching_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute flow matching loss combining L1 and L2 terms
        Args:
            pred: predicted velocity field
            target: target velocity field
        Returns:
            Combined loss value
        """
        l1_loss = F.l1_loss(pred, target)
        l2_loss = F.mse_loss(pred, target)
        return self.l1w * l1_loss + self.l2w * l2_loss