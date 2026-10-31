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
    """Standard PC: relax to a fixed point, then ONE local weight update.

    Returns the settled energy per example.
    """
    with torch.no_grad():  # credit assignment is manual; never build an autograd graph
        settled, _ = model.infer(xs, clamped=clamped, T=T, gamma=gamma)
        _, errors = model.predict(settled)
        optimizer.zero_grad(set_to_none=True)
        model.set_manual_grads(settled, errors)
        optimizer.step()
        return float(energy(errors)) / settled[0].shape[0]


def _step_incremental(
    model: PCN,
    xs: list[torch.Tensor],
    clamped: set[int],
    optimizer: torch.optim.Optimizer,
    T: int | None,
    gamma: float | None,
) -> float:
    """Incremental PC (Salvatori et al. 2022): a weight update at EVERY inference
    step, interleaved with the value-neuron relaxation.

    Standard PC relaxes the value neurons to equilibrium before updating weights;
    at that fixed point the hidden layers have "explained away" much of the
    output error, so the weight update only weakly approximates backprop. iPC
    instead takes a small weight step after each relaxation step, accumulating
    updates along the whole trajectory -- a markedly better backprop
    approximation and a stronger optimizer in practice.
    """
    T = model.T if T is None else T
    gamma = model.gamma if gamma is None else gamma
    with torch.no_grad():
        xs = [x.clone() for x in xs]
        last_energy = 0.0
        for _ in range(T):
            # Relax free value neurons one step.
            _, errors = model.predict(xs)
            grads = model.grad_x(xs, errors)
            for l in range(model.L + 1):
                if l not in clamped:
                    xs[l] = xs[l] - gamma * grads[l]
            # ... then take one local weight step on the fresh errors.
            _, errors = model.predict(xs)
            optimizer.zero_grad(set_to_none=True)
            model.set_manual_grads(xs, errors)
            optimizer.step()
            last_energy = float(energy(errors)) / xs[0].shape[0]
        return last_energy


def train_supervised_epoch(
    model: PCN,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device = "cpu",
    T: int | None = None,
    gamma: float | None = None,
    target_low: float = -1.0,
    target_high: float = 1.0,
    incremental: bool = False,
) -> float:
    """One epoch of supervised PCN training (clamp image AND one-hot label).

    With ``incremental=True`` uses iPC (a weight update per inference step);
    otherwise standard PC (one update after inference converges).
    """
    step = _step_incremental if incremental else _step
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
        total += step(
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
