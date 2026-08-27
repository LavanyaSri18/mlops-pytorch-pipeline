"""CIFAR-10 dataset and transform helpers."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_transforms(train: bool = True) -> transforms.Compose:
    """Return augmentation for training or deterministic validation preprocessing."""
    operations: list[object] = []
    if train:
        operations.extend((transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)))
    operations.extend((transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)))
    return transforms.Compose(operations)


def get_dataloaders(
    data_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Download CIFAR-10 when needed and return training and validation loaders."""
    root = str(data_dir)
    train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=get_transforms(True))
    validation_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=get_transforms(False))
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    )
