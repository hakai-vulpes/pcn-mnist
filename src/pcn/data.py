"""MNIST data loading, normalization and one-hot encoding.

Conventions:
- Images are flattened to vectors of length 784.
- Pixels are standardized (zero mean, unit variance using the MNIST training
  statistics) so the inputs sit comfortably in the range of ``tanh``.
- Labels are returned both as integer class indices and as +/-1 one-hot
  targets, scaled to the range of ``tanh`` for clamping at the top layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

# MNIST global statistics (standard values used across the literature).
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

NUM_CLASSES = 10
IMAGE_DIM = 784


def onehot_targets(
    labels: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    low: float = -1.0,
    high: float = 1.0,
) -> torch.Tensor:
    """Encode integer labels as one-hot targets scaled to ``[low, high]``.

    For a tanh top layer we clamp the correct class to ``high`` (+1) and the
    rest to ``low`` (-1), which lies inside tanh's saturating range.
    """
    onehot = torch.full((labels.shape[0], num_classes), low, dtype=torch.float32)
    onehot[torch.arange(labels.shape[0]), labels] = high
    return onehot


def _flatten_standardize(images: torch.Tensor) -> torch.Tensor:
    """Flatten raw uint8 ``[N, 1, 28, 28]`` to ``[N, 784]`` and standardize.

    The raw MNIST tensors are uint8 in ``[0, 255]``; the standard MNIST_MEAN /
    MNIST_STD constants are calibrated for pixels scaled to ``[0, 1]``, so we
    divide by 255 *first*. (Skipping this scaling leaves pixels ~250x too large
    and swamps the prediction errors -- the classic PCN normalization pitfall.)
    """
    images = images.view(images.shape[0], -1).float() / 255.0
    return (images - MNIST_MEAN) / MNIST_STD


@dataclass
class MNISTData:
    """In-memory MNIST tensors split into train / val / test."""

    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor

    def loaders(
        self, batch_size: int = 128, generator: torch.Generator | None = None
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        train_ds = TensorDataset(self.x_train, self.y_train)
        val_ds = TensorDataset(self.x_val, self.y_val)
        test_ds = TensorDataset(self.x_test, self.y_test)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            drop_last=False,
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader


def load_mnist(
    root: str = "data", val_size: int = 5000, seed: int = 0, download: bool = True
) -> MNISTData:
    """Load MNIST into memory, standardized and flattened.

    A fixed validation split is carved off the 60k training set.
    """
    transform = transforms.ToTensor()
    train_full = datasets.MNIST(
        root=root, train=True, download=download, transform=transform
    )
    test_full = datasets.MNIST(
        root=root, train=False, download=download, transform=transform
    )

    x_train_full = _flatten_standardize(train_full.data.unsqueeze(1))
    y_train_full = train_full.targets.clone()
    x_test = _flatten_standardize(test_full.data.unsqueeze(1))
    y_test = test_full.targets.clone()

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x_train_full.shape[0], generator=g)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    return MNISTData(
        x_train=x_train_full[train_idx],
        y_train=y_train_full[train_idx],
        x_val=x_train_full[val_idx],
        y_val=y_train_full[val_idx],
        x_test=x_test,
        y_test=y_test,
    )


def subset_loader(
    data: MNISTData, n: int, batch_size: int = 128, seed: int = 0
) -> DataLoader:
    """A small fixed subset of the training set (for the overfit test)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(data.x_train.shape[0], generator=g)[:n]
    ds = Subset(TensorDataset(data.x_train, data.y_train), idx.tolist())
    return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)
