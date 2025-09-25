# SuperResolutionFlow

A PyTorch implementation of flow-based super-resolution models, featuring both Normalizing Flows and Flow Matching approaches for high-quality image upsampling.

## Models

### Flow Matching Super-Resolution

- **Architecture**: U-Net with time embedding and self-attention
- **Method**: Rectified flow for optimal transport between noise and high-resolution images
- **Training**: Beta distribution time sampling with velocity matching loss

### Normalizing Flow Super-Resolution

- **Architecture**: Coupling layers with affine transformations
- **Method**: Invertible transformations with learned spatial priors
- **Training**: Maximum likelihood estimation using change of variables

## Features

- **Conditioning Network**: Shared residual network that generates initial upsampled images using bicubic interpolation + learned residuals
- **Multi-scale Support**: 2x and 4x upsampling factors
- **Training Infrastructure**: Unified training framework with checkpointing, validation, and sample generation
- **Data Augmentation**: Built-in augmentation pipeline for robust training

## Project Structure

```
SuperResolutionFlow/
├── models/
│   ├── baseFlow.py              # Abstract base class
│   ├── conditioningNetwork.py   # Shared conditioning network
│   ├── flowMatchingSR.py        # Flow matching implementation
│   └── normalizingFlowSR.py     # Normalizing flow implementation
├── util/
│   ├── config.py                # Configuration classes
│   ├── dataset.py               # Dataset loader
│   ├── trainer.py               # Training infrastructure
│   └── checkpointing.py         # Checkpoint utilities
└── README.md
```

