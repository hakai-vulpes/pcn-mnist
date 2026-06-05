"""Training loops for the PCN (supervised and unsupervised/generative).

Each step is: clamp the observed layers, run inference to settle the free value
neurons, then apply ONE local Hebbian weight update on the settled errors. The
update is applied via a ``torch.optim`` optimizer whose ``.grad`` fields we fill
manually (``model.set_manual_grads``) -- autograd is never used for credit
assignment.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from pcn.data import onehot_targets
from pcn.functional import energy
from pcn.model import PCN


def _step(
    model: PCN,
    xs: list[torch.Tensor],
    clamped: set[int],
    optimizer: torch.optim.Optimizer,
    T: int | None,
    gamma: float | None,
) -> float:
    """One inference-then-learn step. Returns the settled energy per example."""
    with torch.no_grad():  # credit assignment is manual; never build an autograd graph
        settled, _ = model.infer(xs, clamped=clamped, T=T, gamma=gamma)
        _, errors = model.predict(settled)
        optimizer.zero_grad(set_to_none=True)
        model.set_manual_grads(settled, errors)
        optimizer.step()
        return float(energy(errors)) / settled[0].shape[0]


def train_supervised_epoch(
    model: PCN,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device = "cpu",
    T: int | None = None,
    gamma: float | None = None,
    target_low: float = -1.0,
    target_high: float = 1.0,
) -> float:
    """One epoch of supervised PCN training (clamp image AND one-hot label)."""
    model.train()
    total, n_batches = 0.0, 0
    for images, labels in loader:
        images = images.to(device)
        targets = onehot_targets(labels, model.dims[-1], target_low, target_high).to(
            device
        )
        xs = model.init_states(
            images.shape[0], x_bottom=images, x_top=targets, feedforward=True
        )
        total += _step(
            model, xs, clamped={0, model.L}, optimizer=optimizer, T=T, gamma=gamma
        )
        n_batches += 1
    return total / max(n_batches, 1)


def train_unsupervised_epoch(
    model: PCN,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device = "cpu",
    T: int | None = None,
    gamma: float | None = None,
) -> float:
    """One epoch of generative PCN training (clamp image only, free top latent)."""
    model.train()
    total, n_batches = 0.0, 0
    for images, _ in loader:
        images = images.to(device)
        xs = model.init_states(images.shape[0], x_bottom=images, feedforward=True)
        total += _step(model, xs, clamped={0}, optimizer=optimizer, T=T, gamma=gamma)
        n_batches += 1
    return total / max(n_batches, 1)
