import torch
import torch.nn as nn
import math
from typing import Tuple, Union

from .baseFlow import BaseFlow
from .conditioningNetwork import ConditioningNetwork


class CouplingLayer(nn.Module):
    """Affine coupling layer with improved stability"""

    def __init__(self, channels: int, hidden_dim: int = 128):
        super().__init__()
        self.channels = channels
        self.half_channels = channels // 2

        self.net = nn.Sequential(
            nn.Conv2d(self.half_channels, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, (self.channels - self.half_channels) * 2, 3, padding=1)
        )

        # Stable initialization
        with torch.no_grad():
            self.net[-1].weight.normal_(0, 0.05)
            self.net[-1].bias.zero_()

    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = x[:, :self.half_channels], x[:, self.half_channels:]

        st = self.net(x1)
        s, t = st.chunk(2, dim=1)

        if not reverse:
            x2 = x2 * torch.exp(s) + t
            log_det = s.sum(dim=[1, 2, 3])
        else:
            x2 = (x2 - t) * torch.exp(-s)
            log_det = -s.sum(dim=[1, 2, 3])

        return torch.cat([x1, x2], dim=1), log_det


class ActNorm(nn.Module):
    """Activation normalization layer"""

    def __init__(self, channels: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.initialized = False

    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.initialized and self.training and not reverse:
            with torch.no_grad():
                flat_x = x.permute(0, 2, 3, 1).contiguous().view(-1, x.shape[1])
                mean = flat_x.mean(0).view(1, -1, 1, 1)
                std = flat_x.std(0).view(1, -1, 1, 1)

                self.bias.data.copy_(-mean)
                self.scale.data.copy_(1.0 / (std + 1e-6))
            self.initialized = True

        if not reverse:
            x = x * self.scale + self.bias
            log_det = self.scale.abs().log().sum() * x.shape[2] * x.shape[3]
        else:
            x = (x - self.bias) / self.scale
            log_det = -self.scale.abs().log().sum() * x.shape[2] * x.shape[3]

        return x, log_det.expand(x.shape[0])


class Squeeze2d(nn.Module):
    """Squeeze operation for multi-scale processing"""

    def __init__(self, factor: int = 2):
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if not reverse:
            B, C, H, W = x.shape
            x = x.view(B, C, H // self.factor, self.factor, W // self.factor, self.factor)
            x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
            x = x.view(B, C * self.factor * self.factor, H // self.factor, W // self.factor)
        else:
            B, C, H, W = x.shape
            x = x.view(B, C // (self.factor * self.factor), self.factor, self.factor, H, W)
            x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
            x = x.view(B, C // (self.factor * self.factor), H * self.factor, W * self.factor)

        return x, torch.zeros(x.shape[0], device=x.device)


class FlowStep(nn.Module):
    """Single step in normalizing flow"""

    def __init__(self, channels: int):
        super().__init__()
        self.actnorm = ActNorm(channels)
        self.coupling = CouplingLayer(channels)
        self.permutation = nn.Parameter(torch.eye(channels)[torch.randperm(channels)])

    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(x.size(0), device=x.device)

        if not reverse:
            x, ld = self.actnorm(x)
            log_det += ld

            x = torch.einsum('bchw,cd->bdhw', x, self.permutation)

            x, ld = self.coupling(x)
            log_det += ld
        else:
            x, ld = self.coupling(x, reverse=True)
            log_det += ld

            x = torch.einsum('bchw,dc->bdhw', x, self.permutation)

            x, ld = self.actnorm(x, reverse=True)
            log_det += ld

        return x, log_det


class NormalizingFlowSR(BaseFlow):
    """Normalizing Flow Super-Resolution model"""

    def __init__(self,
                 hr_channels: int = 3,
                 lr_channels: int = 3,
                 scale_factor: int = 2,
                 cond_net_base_channels: int = 64,
                 cond_net_res_block_num: int = 6,
                 cond_net_res_weight: float = 0.1,
                 num_flows: int = 12):
        super().__init__(hr_channels, lr_channels, scale_factor)

        # Conditioning network
        self.conditioning = ConditioningNetwork(
            lr_channels=lr_channels,
            hr_channels=hr_channels,
            scale_factor=scale_factor,
            hidden_dim=cond_net_base_channels,
            num_residual_blocks=cond_net_res_block_num,
            residual_weight=cond_net_res_weight,)

        # Multi-scale architecture
        self.squeeze = Squeeze2d(factor=2)
        squeezed_channels = hr_channels * 4

        # Flow steps
        self.flows = nn.ModuleList([
            FlowStep(squeezed_channels) for _ in range(num_flows)
        ])

        # Learnable prior
        self.prior_conv = nn.Sequential(
            nn.Conv2d(squeezed_channels, cond_net_base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cond_net_base_channels, squeezed_channels * 2, 3, padding=1)
        )

    def _get_prior_params(self, shape: Tuple[int, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get parameters for spatial prior"""
        B, C, H, W = shape
        device = self.prior_conv[0].weight.device

        h_coord = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(B, C, H, W)
        w_coord = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).expand(B, C, H, W)

        coords = torch.cat([h_coord, w_coord], dim=1)
        params = self.prior_conv(coords[:, :C])
        mean, log_std = params.chunk(2, dim=1)

        return mean, log_std

    def forward(self, hr_img: torch.Tensor, lr_img: torch.Tensor, reverse: bool = False) -> Union[
        Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Forward pass"""
        return self.decode(hr_img, lr_img) if reverse else self.encode(hr_img, lr_img)

    def encode(self, hr_img: torch.Tensor, lr_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode HR image to latent space"""
        conditioning = self.conditioning(lr_img)
        x = hr_img - conditioning

        x, _ = self.squeeze(x)

        log_det_total = torch.zeros(x.shape[0], device=x.device)
        for flow in self.flows:
            x, log_det = flow(x)
            log_det_total += log_det

        return x, log_det_total

    def decode(self, z: torch.Tensor, lr_img: torch.Tensor) -> torch.Tensor:
        """Decode latent to HR image"""
        x = z

        for flow in reversed(self.flows):
            x, _ = flow(x, reverse=True)

        x, _ = self.squeeze(x, reverse=True)

        conditioning = self.conditioning(lr_img)
        return x + conditioning

    def sample(self, lr_img):
        batch_size = lr_img.size(0)

        # Get squeezed dimensions
        hr_h, hr_w = lr_img.size(2) * self.scale_factor, lr_img.size(3) * self.scale_factor
        squeezed_h, squeezed_w = hr_h // 2, hr_w // 2
        squeezed_channels = self.hr_channels * 4

        with torch.no_grad():
            mean, log_std = self._get_prior_params((batch_size, squeezed_channels, squeezed_h, squeezed_w))
            eps = torch.randn_like(mean)
            z = mean + eps * torch.exp(log_std)
            sample = self.decode(z, lr_img)
            sample = torch.clamp(sample, 0, 1)

        return sample

    def log_likelihood(self, hr_img, lr_img):
        z, log_det = self.encode(hr_img, lr_img)

        # Log probability under learned prior
        mean, log_std = self._get_prior_params(z.shape)

        log_prior = -0.5 * torch.sum(
            (z - mean) ** 2 / (torch.exp(2 * log_std) + 1e-8) +
            2 * log_std + math.log(2 * math.pi),
            dim=[1, 2, 3]
        )

        return log_prior + log_det