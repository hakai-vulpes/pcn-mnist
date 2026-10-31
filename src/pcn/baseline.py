"""Same-architecture backprop MLP baseline.

This is the head-to-head comparison for the PCN: identical layer dimensions and
nonlinearity, but trained the ordinary way with ``loss.backward()`` + Adam. The
PCN should land within ~1% of this model's test accuracy.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from pcn.functional import ACTIVATIONS


class MLP(nn.Module):
    """Plain feedforward MLP matching a PCN's dims/nonlinearity.

    A PCN with dims ``[d0, d1, ..., dL]`` predicts top-down; the discriminative
    MLP runs bottom-up ``d0 -> d1 -> ... -> dL`` with the same hidden
    nonlinearity and a linear output layer (logits).
    """

    def __init__(self, dims: list[int], activation: str = "tanh") -> None:
        super().__init__()
        if activation not in ACTIVATIONS:
            raise KeyError(f"unknown activation {activation!r}")
        self.act_name = activation
        act_layer = {"tanh": nn.Tanh, "relu": nn.ReLU, "identity": nn.Identity}[
            activation
        ]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # no activation on the logits layer
                layers.append(act_layer())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp_epoch(
    model: MLP,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device = "cpu",
    loss: str = "ce",
    target_low: float = -1.0,
    target_high: float = 1.0,
) -> float:
    """One backprop epoch. ``loss`` is ``"ce"`` (cross-entropy) or ``"mse"``
    (squared error to a one-hot, the same objective the PCN minimizes -- this
    makes the MLP an apples-to-apples backprop baseline for inference learning)."""
    from pcn.data import onehot_targets

    model.train()
    ce = nn.CrossEntropyLoss()
    total, n = 0.0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(images)
        if loss == "mse":
            tgt = onehot_targets(labels, out.shape[1], target_low, target_high).to(
                device
            )
            batch_loss = nn.functional.mse_loss(out, tgt)
        else:
            batch_loss = ce(out, labels)
        batch_loss.backward()
        optimizer.step()
        total += float(batch_loss) * labels.shape[0]
        n += labels.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def mlp_accuracy(
    model: MLP, loader: Iterable, device: str | torch.device = "cpu"
) -> float:
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += int((preds == labels).sum())
        total += labels.shape[0]
    return correct / max(total, 1)
