"""Nonlinearities (f and f') and the PCN energy F.

The energy minimized by both inference and learning is the sum of squared
prediction errors::

    F = sum_{l=0}^{L-1} 0.5 * ||eps[l]||^2

where ``eps[l] = x[l] - mu[l]`` and ``mu[l] = W[l] @ f(x[l+1]) + b[l]``.

Each activation function exposes the value ``f(x)`` and its elementwise
derivative ``f'(x)`` so the inference update can use them directly (the manual
update rule needs ``f'(x[l])`` explicitly -- it is not obtained via autograd).
"""

from __future__ import annotations

from collections.abc import Callable

import torch

# An activation is a pair (f, f') of elementwise functions.
Activation = tuple[
    Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]
]


def tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


def tanh_grad(x: torch.Tensor) -> torch.Tensor:
    """d/dx tanh(x) = 1 - tanh(x)^2."""
    t = torch.tanh(x)
    return 1.0 - t * t


def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)


def relu_grad(x: torch.Tensor) -> torch.Tensor:
    return (x > 0).to(x.dtype)


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


def identity_grad(x: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(x)


ACTIVATIONS: dict[str, Activation] = {
    "tanh": (tanh, tanh_grad),
    "relu": (relu, relu_grad),
    "identity": (identity, identity_grad),
}


def get_activation(name: str) -> Activation:
    if name not in ACTIVATIONS:
        raise KeyError(
            f"unknown activation {name!r}; choose from {sorted(ACTIVATIONS)}"
        )
    return ACTIVATIONS[name]


def energy(errors: list[torch.Tensor]) -> torch.Tensor:
    """F = sum_l 0.5 * ||eps[l]||^2, summed over batch and units.

    ``errors`` is the list of per-layer error tensors ``eps[l]``, each of shape
    ``[B, dim_l]``. Returns a scalar (the total energy over the batch).
    """
    total = errors[0].new_zeros(())
    for eps in errors:
        total = total + 0.5 * (eps * eps).sum()
    return total
