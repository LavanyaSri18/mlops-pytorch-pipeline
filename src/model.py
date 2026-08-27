"""Model definitions for CIFAR-10 classification."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18


def get_model(architecture: str = "resnet18", num_classes: int = 10) -> nn.Module:
    """Create a classifier configured for 32x32 CIFAR-10 images."""
    if architecture.lower() != "resnet18":
        raise ValueError(f"Unsupported architecture: {architecture}. Supported: resnet18")

    model = resnet18(weights=None, num_classes=num_classes)
    # CIFAR-10 images are 32x32, so use a smaller stem than ImageNet ResNet.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
