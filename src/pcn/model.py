"""The Predictive Coding Network.

Indexing convention (fixed everywhere): layers ``l = 0 .. L`` where layer 0 is
the data/image end and layer L is the abstract/label end. There is one weight
matrix ``W[i]`` and bias ``b[i]`` per adjacent pair of layers (``i = 0 .. L-1``).

What differs between tasks is the *prediction direction* -- which neighbour
predicts which. We support both, because the network is discriminative *and*
generative (Bogacz 2017; Whittington & Bogacz 2017):

* ``direction="generative"`` -- the layer *above* predicts the layer *below*
  (top-down). Connection ``i`` predicts the lower layer ``i`` from the upper
  layer ``i+1``. ``W[i]`` has shape ``[dim_i, dim_{i+1}]``. The energy is then
  dominated by reconstructing the 784-d image, so this is the right orientation
  for unsupervised representation learning, reconstruction, and class-conditioned
  generation (Phase 5).

* ``direction="discriminative"`` -- the layer *below* predicts the layer *above*
  (bottom-up). Connection ``i`` predicts the upper layer ``i+1`` from the lower
  layer ``i``. ``W[i]`` has shape ``[dim_{i+1}, dim_i]``. Now the energy is
  driven by the *output* (label) error, so inference learning approximates
  backprop and reaches MLP-level accuracy. This is the orientation for the
  supervised classifier (Phase 4); a feedforward pass through it is exactly an
  MLP forward pass, and classification reads out ``argmax(x[L])``.

Either way each connection ``i`` has a *predictor* layer ``p`` and a *target*
layer ``t`` with ``mu[i] = f(x[p]) @ W[i].T + b[i]`` and ``eps[i] = x[t] - mu[i]``
(``W[i]`` has shape ``[dim_t, dim_p]``). The energy is ``F = sum_i 0.5||eps[i]||^2``.

Both inference (relaxing free value neurons) and learning (updating weights) are
gradient *descent* on F, done with **manually derived** local updates -- never
``loss.backward()``.

Manual gradients (verified against autograd in tests/test_gradients.py, for both
directions). Each connection ``i`` contributes to two value layers::

    dF/dx[t] += eps[i]                          (own error, at the target layer)
    dF/dx[p] += - f'(x[p]) * (eps[i] @ W[i])    (drive, at the predictor layer)
    dF/dW[i]  = - eps[i].T @ f(x[p])
    dF/db[i]  = - sum_batch eps[i]
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
        direction: str = "generative",
        seed: int | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        if len(dims) < 2:
            raise ValueError("need at least 2 layers (input + output)")
        if direction not in ("generative", "discriminative"):
            raise ValueError(f"unknown direction {direction!r}")
        self.dims = list(dims)
        self.L = len(dims) - 1  # top layer index; there are L weight matrices
        self.gamma = gamma
        self.T = T
        self.direction = direction
        self.activation_name = activation
        self.f, self.f_grad = get_activation(activation)
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )

        # Per-connection predictor/target layer indices. Generative predicts the
        # lower layer from the upper one; discriminative predicts the upper from
        # the lower. Everything else (predict/grad/init) is written against these.
        if direction == "generative":
            self._pred = [i + 1 for i in range(self.L)]  # upper predicts ...
            self._targ = [i for i in range(self.L)]  # ... lower
        else:  # discriminative
            self._pred = [i for i in range(self.L)]  # lower predicts ...
            self._targ = [i + 1 for i in range(self.L)]  # ... upper

        if seed is not None:
            gen = torch.Generator().manual_seed(seed)
        else:
            gen = None

        # W[i]: [dim_targ, dim_pred], b[i]: [dim_targ], for i = 0..L-1.
        Ws, bs = [], []
        for l in range(self.L):
            out_dim, in_dim = dims[self._targ[l]], dims[self._pred[l]]
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
        """Compute ``mu[i]`` and ``eps[i]`` for each connection ``i = 0 .. L-1``.

        ``mu[i] = f(x[pred]) @ W[i].T + b[i]`` and ``eps[i] = x[targ] - mu[i]``,
        where ``pred``/``targ`` depend on ``self.direction``. ``Ws``/``bs`` allow
        passing external (e.g. autograd-tracked) parameters; otherwise the
        model's own parameters are used.
        """
        Ws = self.W if Ws is None else Ws
        bs = self.b if bs is None else bs
        mus, errors = [], []
        for l in range(self.L):
            mu = self.f(xs[self._pred[l]]) @ Ws[l].T + bs[l]
            mus.append(mu)
            errors.append(xs[self._targ[l]] - mu)
        return mus, errors

    # ------------------------------------------------------------------ #
    # Manual gradients (the heart of the model)
    # ------------------------------------------------------------------ #
    def grad_x(
        self, xs: list[torch.Tensor], errors: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """dF/dx[l] for every layer ``l = 0 .. L`` (matches autograd of F).

        Each connection ``i`` adds its own error at the target layer and a drive
        (through ``f'``) at the predictor layer; we accumulate both per layer.
        """
        grads = [torch.zeros_like(x) for x in xs]
        for i in range(self.L):
            t, p = self._targ[i], self._pred[i]
            grads[t] = grads[t] + errors[i]  # own error at the target layer
            grads[p] = grads[p] - self.f_grad(xs[p]) * (errors[i] @ self.W[i])
        return grads

    def grad_w(
        self, xs: list[torch.Tensor], errors: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """dF/dW[i], dF/db[i] for ``i = 0 .. L-1`` (summed over batch; matches autograd)."""
        gW, gb = [], []
        for i in range(self.L):
            gW.append(-errors[i].T @ self.f(xs[self._pred[i]]))
            gb.append(-errors[i].sum(dim=0))
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
        seeded by a bottom-up pass so inference starts near a fixed point and
        converges in far fewer steps. The warm start depends on the direction:

        * discriminative -- the *exact* forward prediction
          ``x[l+1] = f(x[l]) @ W[l].T + b[l]`` (an MLP forward pass), so with
          only the input clamped the settled state already equals the MLP output.
        * generative -- a cheap encoder heuristic ``x[l+1] = f(x[l]) @ W[l]``
          (the transpose direction), which is good enough to warm-start the
          top-down relaxation.

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
                if self.direction == "discriminative":
                    xs[l + 1] = self.f(xs[l]) @ self.W[l].T + self.b[l]
                else:
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
