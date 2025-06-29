import os
from typing import Tuple

from PIL import Image
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from torchvision import transforms


class SuperResolutionDataset(Dataset):
    """Dataset for real super-resolution training"""

    def __init__(self, root_dir: str, hr_size: int = 128, scale_factor: int = 2, augment: bool = True):
        self.hr_dir = root_dir
        self.hr_images = sorted([
            os.path.join(self.hr_dir, fname)
            for fname in os.listdir(self.hr_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.scale_factor = scale_factor
        self.hr_size = hr_size
        self.augment = augment

        print(f"Found {len(self.hr_images)} images in {root_dir}")

        # Define transforms based on augmentation setting
        if augment:
            self.hr_transform = transforms.Compose([
                transforms.Resize((hr_size + hr_size // 4, hr_size + hr_size // 4)),
                transforms.RandomCrop((hr_size, hr_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02),
                transforms.ToTensor()
            ])
        else:
            self.hr_transform = transforms.Compose([
                transforms.Resize((hr_size, hr_size)),
                transforms.ToTensor()
            ])

    def __len__(self) -> int:
        return len(self.hr_images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load and transform high-resolution image
        hr_img = Image.open(self.hr_images[idx]).convert("RGB")
        hr = self.hr_transform(hr_img)

        # Generate corresponding low-resolution image
        lr = F.interpolate(
            hr.unsqueeze(0),
            scale_factor=1 / self.scale_factor,
            mode='area',
            recompute_scale_factor=False
        ).squeeze(0)

        # Add realistic blur to low-resolution image
        lr = transforms.GaussianBlur(kernel_size=3, sigma=0.5)(lr)

        return hr, lr