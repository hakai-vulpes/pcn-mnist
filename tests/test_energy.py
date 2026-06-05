"""Inference must actually minimize the energy.

For a small enough step size ``gamma``, gradient descent on F should make the
energy non-increasing from one inference step to the next, settling toward a
fixed point. If this fails the inference dynamics are wrong (or gamma too big).
"""

import torch

from pcn.model import PCN


def test_energy_non_increasing_during_inference():
    torch.manual_seed(0)
    dims = [20, 16, 12, 6]
    model = PCN(dims=dims, activation="tanh", init_std=0.3, gamma=0.05, T=80, seed=0)
    model.double()

    batch = 8
    x_bottom = torch.randn(batch, dims[0], dtype=torch.float64)
    x_top = torch.randn(batch, dims[-1], dtype=torch.float64)
    xs = model.init_states(batch, x_bottom=x_bottom, x_top=x_top, feedforward=True)

    # Clamp both ends (supervised-style); relax the hidden layers.
    clamped = {0, model.L}
    _, energies = model.infer(xs, clamped=clamped, record_energy=True)

    diffs = torch.tensor(energies[1:]) - torch.tensor(energies[:-1])
    # Allow a tiny positive tolerance for floating-point noise near the fixed point.
    assert (
        diffs <= 1e-9
    ).all(), (
        f"energy increased during inference; max step increase {diffs.max().item()}"
    )
    # And it should actually decrease meaningfully overall.
    assert energies[-1] < energies[0]


def test_energy_settles_to_plateau():
    torch.manual_seed(1)
    dims = [20, 16, 6]
    model = PCN(dims=dims, activation="tanh", init_std=0.3, gamma=0.05, T=300, seed=1)
    model.double()

    batch = 4
    x_bottom = torch.randn(batch, dims[0], dtype=torch.float64)
    xs = model.init_states(batch, x_bottom=x_bottom, feedforward=True)
    _, energies = model.infer(xs, clamped={0}, record_energy=True)

    e = torch.tensor(energies)
    total_descent = e[0] - e[-1]
    # "Settling": the energy change over the final 10% of inference is a tiny
    # fraction of the whole descent -- i.e. we are approaching a fixed point,
    # not still moving fast. (Robust to the exact convergence rate / gamma.)
    n = len(e)
    last_segment = e[int(0.9 * n)] - e[-1]
    assert last_segment < 0.01 * total_descent
