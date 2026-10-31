"""The central correctness oracle.

We build the PCN energy F on a random batch and check that the *manual*
gradients used by the model -- dF/dx (inference) and dF/dW, dF/db (learning) --
match ``torch.autograd.grad`` of the very same energy to high precision. This is
the test that catches index/transpose bugs, which are the #1 source of silent
PCN failure.

Autograd is used here ONLY as an oracle. The model itself never calls
``.backward()`` for credit assignment.
"""

import torch

from pcn.functional import energy
from pcn.model import PCN


def _make_model_and_states(seed=0, activation="tanh", direction="generative"):
    torch.manual_seed(seed)
    dims = [12, 8, 6, 4]
    model = PCN(
        dims=dims, activation=activation, init_std=0.3, seed=seed, direction=direction
    )
    batch = 5
    # Random value states for every layer 0..L.
    xs = [torch.randn(batch, d, dtype=torch.float64) for d in dims]
    model.double()
    return model, xs


def _autograd_grads(model, xs):
    """Reference dF/dx, dF/dW, dF/db from autograd on the energy."""
    xs_leaf = [x.clone().detach().requires_grad_(True) for x in xs]
    Ws = [W.clone().detach().requires_grad_(True) for W in model.W]
    bs = [b.clone().detach().requires_grad_(True) for b in model.b]
    _, errors = model.predict(xs_leaf, Ws=Ws, bs=bs)
    F = energy(errors)
    grads = torch.autograd.grad(F, xs_leaf + Ws + bs)
    n_x = len(xs_leaf)
    n_w = len(Ws)
    gx = list(grads[:n_x])
    gW = list(grads[n_x : n_x + n_w])
    gb = list(grads[n_x + n_w :])
    return gx, gW, gb


def test_manual_dF_dx_matches_autograd():
    model, xs = _make_model_and_states()
    _, errors = model.predict(xs)
    manual_gx = model.grad_x(xs, errors)
    gx, _, _ = _autograd_grads(model, xs)
    for l in range(len(xs)):
        assert torch.allclose(manual_gx[l], gx[l], atol=1e-8, rtol=1e-6), (
            f"dF/dx mismatch at layer {l}: max abs diff "
            f"{(manual_gx[l] - gx[l]).abs().max().item()}"
        )


def test_manual_dF_dW_db_matches_autograd():
    model, xs = _make_model_and_states()
    _, errors = model.predict(xs)
    manual_gW, manual_gb = model.grad_w(xs, errors)
    _, gW, gb = _autograd_grads(model, xs)
    for l in range(len(model.W)):
        assert torch.allclose(manual_gW[l], gW[l], atol=1e-8, rtol=1e-6), (
            f"dF/dW mismatch at layer {l}: max abs diff "
            f"{(manual_gW[l] - gW[l]).abs().max().item()}"
        )
        assert torch.allclose(manual_gb[l], gb[l], atol=1e-8, rtol=1e-6), (
            f"dF/db mismatch at layer {l}: max abs diff "
            f"{(manual_gb[l] - gb[l]).abs().max().item()}"
        )


def test_relu_dF_dx_matches_autograd():
    # The f'(x) term differs by activation -- check a second one.
    model, xs = _make_model_and_states(activation="relu")
    _, errors = model.predict(xs)
    manual_gx = model.grad_x(xs, errors)
    gx, _, _ = _autograd_grads(model, xs)
    for l in range(len(xs)):
        assert torch.allclose(
            manual_gx[l], gx[l], atol=1e-8, rtol=1e-6
        ), f"relu dF/dx mismatch at layer {l}"


# --- Discriminative direction (predictions flow input -> output) ------------ #
# The same energy/gradient identities must hold when the network predicts
# bottom-up (the "PC approximates backprop" orientation used for classification).


def test_discriminative_dF_dx_matches_autograd():
    model, xs = _make_model_and_states(direction="discriminative")
    _, errors = model.predict(xs)
    manual_gx = model.grad_x(xs, errors)
    gx, _, _ = _autograd_grads(model, xs)
    for l in range(len(xs)):
        assert torch.allclose(manual_gx[l], gx[l], atol=1e-8, rtol=1e-6), (
            f"discriminative dF/dx mismatch at layer {l}: max abs diff "
            f"{(manual_gx[l] - gx[l]).abs().max().item()}"
        )


def test_discriminative_dF_dW_db_matches_autograd():
    model, xs = _make_model_and_states(direction="discriminative")
    _, errors = model.predict(xs)
    manual_gW, manual_gb = model.grad_w(xs, errors)
    _, gW, gb = _autograd_grads(model, xs)
    for l in range(len(model.W)):
        assert torch.allclose(manual_gW[l], gW[l], atol=1e-8, rtol=1e-6), (
            f"discriminative dF/dW mismatch at layer {l}: max abs diff "
            f"{(manual_gW[l] - gW[l]).abs().max().item()}"
        )
        assert torch.allclose(manual_gb[l], gb[l], atol=1e-8, rtol=1e-6), (
            f"discriminative dF/db mismatch at layer {l}: max abs diff "
            f"{(manual_gb[l] - gb[l]).abs().max().item()}"
        )
