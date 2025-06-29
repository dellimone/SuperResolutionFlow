import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod


@dataclass
class BaseSRConfig(ABC):
    """Base configuration class for Super-Resolution training with common parameters."""

    # Data parameters
    root_dir: str = "data/set14"
    hr_size: int = 128
    scale_factor: int = 2
    hr_channels: int = 3
    lr_channels: int = 3
    train_split: float = 0.8  # Fraction of data for training (set to 1.0 for no validation)

    # Training parameters
    batch_size: int = 8
    num_epochs: int = 200
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4

    # Optimization parameters
    optimizer_type: str = 'AdamW'
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "onecycle"
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8

    # Conditioning Network
    cond_net_base_channels: int = 64
    cond_net_res_block_num: int =  6
    cond_net_res_weight: float = 0.2

    # System parameters
    num_workers: int = 2

    # Evaluation and saving parameters
    eval_every: int = 25
    save_every: int = 50
    num_samples: int = 5

    # Paths
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"
    resume_from: Optional[str] = None

    # Additional optimizer parameters that can be overridden
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization setup."""
        # Set default checkpoint and output directories based on model type
        if self.checkpoint_dir == "checkpoints":
            self.checkpoint_dir = f"checkpoints/{self.get_model_name()}"
        if self.output_dir == "outputs":
            self.output_dir = f"outputs/{self.get_model_name()}"

        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Validate configuration
        self.validate_config()

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model name for directory naming."""
        pass

    def validate_config(self):
        """Validate configuration parameters."""
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.num_epochs > 0, "Number of epochs must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert 0 < self.train_split <= 1.0, "Train split must be between 0 and 1"
        assert self.scale_factor in (2, 4), "Scale factor must be greater than 2 or 4"
        assert self.hr_size > 0, "HR size must be positive"
        assert self.hr_channels > 0 and self.lr_channels > 0, "Channels must be positive"
        assert self.lr_scheduler_type in ["onecycle", "cosine"], "Invalid scheduler type"
        assert self.optimizer_type in ["AdamW", "Adam", "SGD"], "Invalid optimizer type"

@dataclass
class NormalizingFlowSRConfig(BaseSRConfig):
    """Configuration for Normalizing Flow Super-Resolution training."""

    # Model-specific parameters
    num_flows: int = 12
    hidden_dim: int = 128
    temperatures: List[float] = field(default_factory=lambda: [1.0])

    def get_model_name(self) -> str:
        return "normalizing-flow"

    def validate_config(self):
        """Additional validation for Normalizing Flow."""
        super().validate_config()
        assert self.num_flows > 0, "Number of flows must be positive"
        assert self.hidden_dim > 0, "Hidden dimension must be positive"

@dataclass
class FlowMatchingSRConfig(BaseSRConfig):
    """Configuration for Flow Matching Super-Resolution training."""
    # Unet parameters
    unet_base_channels: int = 64
    unet_channel_mult: List[int] = field(default_factory=lambda: [1, 2, 2, 4])
    unet_attention: List[bool] = field(default_factory=lambda: [False, False, True, True])
    unet_num_res_block = 2

    # Flow Matching specific parameters
    integration_step: int = 50
    alpha: float = 1.0
    beta: float = 1.0
    l1w: float = 1.0
    l2w: float = 0.1
    def get_model_name(self) -> str:
        return "flow-matching"

    def validate_config(self):
        """Additional validation for Flow Matching."""
        super().validate_config()
        assert self.unet_base_channels > 0, "Base channels must be positive"
        assert self.alpha > 0 and self.beta > 0, "Alpha and beta must be positive"
        assert self.integration_step > 0, "Integration step must be positive"



