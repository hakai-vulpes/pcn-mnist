"""The Predictive Coding Network.

Indexing convention (fixed everywhere): layers ``l = 0 .. L`` where layer 0 is
the data/image end and layer L is the abstract/label end. The layer *above*
predicts the layer *below*::

    mu[l]  = W[l] @ f(x[l+1]) + b[l]        for l = 0 .. L-1   (column-vector form)
    eps[l] = x[l] - mu[l]

In batched row-vector form (states are ``[B, dim]``) this is::

    mu[l]  = f(x[l+1]) @ W[l].T + b[l]
    eps[l] = x[l] - mu[l]

so ``W[l]`` has shape ``[dim_l, dim_{l+1}]`` and maps layer ``l+1 -> l``.

The energy is ``F = sum_l 0.5 ||eps[l]||^2``. Both inference (relaxing free value
neurons) and learning (updating weights) are gradient *descent* on F, done with
**manually derived** local updates -- never ``loss.backward()``.

Manual gradients (verified against autograd in tests/test_gradients.py):

    dF/dx[l] = eps[l]                                  (own error,  l = 0..L-1)
             - f'(x[l]) * (eps[l-1] @ W[l-1])          (error below, l = 1..L)
    dF/dW[l] = - eps[l].T @ f(x[l+1])                  (l = 0..L-1)
    dF/db[l] = - sum_batch eps[l]                      (l = 0..L-1)
"""

from __future__ import annotations

import torch
from torch import nn

from pcn.functional import get_activation


class PCN(nn.Module):
    def __init__(
        self,
        dims: list[int],
        activation: str = "tanh",
        gamma: float = 0.1,
        T: int = 20,
        init_std: float = 0.05,
        init: str = "xavier",
        seed: int | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        if len(dims) < 2:
            raise ValueError("need at least 2 layers (input + output)")
        self.dims = list(dims)
        self.L = len(dims) - 1  # top layer index; there are L weight matrices
        self.gamma = gamma
        self.T = T
        self.activation_name = activation
        self.f, self.f_grad = get_activation(activation)
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )

        if seed is not None:
            gen = torch.Generator().manual_seed(seed)
        else:
            gen = None

        # W[l]: [dim_l, dim_{l+1}], b[l]: [dim_l], for l = 0..L-1.
        Ws, bs = [], []
        for l in range(self.L):
            out_dim, in_dim = dims[l], dims[l + 1]
            W = torch.empty(out_dim, in_dim)
            if init == "xavier":
                bound = (6.0 / (in_dim + out_dim)) ** 0.5
                if gen is not None:
                    W.uniform_(-bound, bound, generator=gen)
                else:
                    W.uniform_(-bound, bound)
            else:  # "normal"
                if gen is not None:
                    W.normal_(0.0, init_std, generator=gen)
                else:
                    W.normal_(0.0, init_std)
            Ws.append(nn.Parameter(W))
            bs.append(nn.Parameter(torch.zeros(out_dim)))
        self.W = nn.ParameterList(Ws)
        self.b = nn.ParameterList(bs)
        self.to(self.device)

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict(
        self,
        xs: list[torch.Tensor],
        Ws: list[torch.Tensor] | None = None,
        bs: list[torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Compute ``mu[l]`` and ``eps[l]`` for ``l = 0 .. L-1``.

        ``Ws``/``bs`` allow passing external (e.g. autograd-tracked) parameters;
        otherwise the model's own parameters are used.
        """
        Ws = self.W if Ws is None else Ws
        bs = self.b if bs is None else bs
        mus, errors = [], []
        for l in range(self.L):
            mu = self.f(xs[l + 1]) @ Ws[l].T + bs[l]
            mus.append(mu)
            errors.append(xs[l] - mu)
        return mus, errors

    # ------------------------------------------------------------------ #
    # Manual gradients (the heart of the model)
    # ------------------------------------------------------------------ #
    def grad_x(
        self, xs: list[torch.Tensor], errors: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """dF/dx[l] for every layer ``l = 0 .. L`` (matches autograd of F)."""
        grads = []
        for l in range(self.L + 1):
            g = torch.zeros_like(xs[l])
            if l < self.L:  # own error term (no error neuron at the top layer L)
                g = g + errors[l]
            if l >= 1:  # bottom-up drive from the unexplained error below
                g = g - self.f_grad(xs[l]) * (errors[l - 1] @ self.W[l - 1])
            grads.append(g)
        return grads

    def grad_w(
        self, xs: list[torch.Tensor], errors: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """dF/dW[l], dF/db[l] for ``l = 0 .. L-1`` (summed over batch; matches autograd)."""
        gW, gb = [], []
        for l in range(self.L):
            gW.append(-errors[l].T @ self.f(xs[l + 1]))
            gb.append(-errors[l].sum(dim=0))
        return gW, gb

    # ------------------------------------------------------------------ #
    # State initialization
    # ------------------------------------------------------------------ #
    def init_states(
        self,
        batch_size: int,
        x_bottom: torch.Tensor | None = None,
        x_top: torch.Tensor | None = None,
        feedforward: bool = True,
    ) -> list[torch.Tensor]:
        """Initialize value states for every layer.

        With ``feedforward=True`` and a clamped bottom (image), hidden layers are
        seeded by a feedforward pass so inference starts near a fixed point and
        converges in roughly half the steps. The feedforward direction inverts
        the generative mapping approximately by ``x[l+1] = f(x[l]) @ W[l]`` (the
        transpose / encoder direction), which is a cheap, good-enough warm start.

        If a top clamp is supplied it overrides the feedforward value at layer L.
        """
        xs: list[torch.Tensor] = [None] * (self.L + 1)  # type: ignore[list-item]
        dev = self.W[0].device
        dtype = self.W[0].dtype

        if x_bottom is not None:
            xs[0] = x_bottom.to(dev, dtype)
        else:
            xs[0] = torch.zeros(batch_size, self.dims[0], device=dev, dtype=dtype)

        if feedforward and x_bottom is not None:
            for l in range(self.L):
                xs[l + 1] = self.f(xs[l]) @ self.W[l]
        else:
            for l in range(1, self.L + 1):
                xs[l] = torch.zeros(batch_size, self.dims[l], device=dev, dtype=dtype)

        if x_top is not None:
            xs[self.L] = x_top.to(dev, dtype)
        return xs

    # ------------------------------------------------------------------ #
    # Inference (relaxation of free value neurons)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def infer(
        self,
        xs: list[torch.Tensor],
        clamped: set[int],
        T: int | None = None,
        gamma: float | None = None,
        record_energy: bool = False,
    ) -> tuple[list[torch.Tensor], list[float]]:
        """Relax the unclamped value neurons by gradient descent on F.

        ``clamped`` is the set of layer indices held fixed (observed). Returns the
        settled states and, optionally, the energy after each step (for plots).
        """
        from pcn.functional import energy

        T = self.T if T is None else T
        gamma = self.gamma if gamma is None else gamma
        xs = [x.clone() for x in xs]
        energies: list[float] = []

        if record_energy:
            _, errors = self.predict(xs)
            energies.append(float(energy(errors)))

        for _ in range(T):
            _, errors = self.predict(xs)
            grads = self.grad_x(xs, errors)
            for l in range(self.L + 1):
                if l not in clamped:
                    xs[l] = xs[l] - gamma * grads[l]
            if record_energy:
                _, errors = self.predict(xs)
                energies.append(float(energy(errors)))
        return xs, energies

    # ------------------------------------------------------------------ #
    # Learning (single Hebbian weight update after inference settles)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def set_manual_grads(
        self, xs: list[torch.Tensor], errors: list[torch.Tensor]
    ) -> None:
        """Write dF/dW, dF/db (averaged over batch) into ``.grad`` of each param.

        After this an external ``torch.optim`` optimizer can ``.step()`` on the
        manual gradient -- no autograd graph is involved.
        """
        batch = xs[0].shape[0]
        gW, gb = self.grad_w(xs, errors)
        for l in range(self.L):
            self.W[l].grad = gW[l] / batch
            self.b[l].grad = gb[l] / batch

    @torch.no_grad()
    def learn_sgd(
        self, xs: list[torch.Tensor], errors: list[torch.Tensor], alpha: float
    ) -> None:
        """Plain local Hebbian SGD update (gradient descent on F)."""
        batch = xs[0].shape[0]
        gW, gb = self.grad_w(xs, errors)
        for l in range(self.L):
            self.W[l] -= alpha * gW[l] / batch
            self.b[l] -= alpha * gb[l] / batch
