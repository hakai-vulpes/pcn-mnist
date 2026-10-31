"""Evaluation: classification accuracy, reconstruction, linear probe.

Classification clamps the image at the bottom, leaves the label end FREE, runs
inference, and reads out ``argmax(x[L])``. Reconstruction clamps the image, runs
inference with the top free, and compares ``mu[0]`` (the model's prediction of
the image) to the input. The linear probe measures representation quality by
fitting a logistic regression on settled top-layer states.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from pcn.functional import energy
from pcn.model import PCN


@torch.no_grad()
def settle_with_diagnostics(
    model: PCN,
    xs: list[torch.Tensor],
    clamped: set[int],
    T: int | None = None,
    gamma: float | None = None,
) -> tuple[list[torch.Tensor], list[float], dict[int, list[float]]]:
    """Run inference recording energy and per-layer error norms each step.

    Returns ``(settled_states, energies, error_norms)`` where ``error_norms`` maps
    each error layer ``l`` to its ``||eps[l]||`` over inference steps. Used for the
    'settling' visualizations.
    """
    T = model.T if T is None else T
    gamma = model.gamma if gamma is None else gamma
    xs = [x.clone() for x in xs]
    energies: list[float] = []
    norms: dict[int, list[float]] = {l: [] for l in range(model.L)}

    def record() -> None:
        _, errors = model.predict(xs)
        energies.append(float(energy(errors)))
        for l in range(model.L):
            norms[l].append(float(errors[l].norm()))

    record()
    for _ in range(T):
        _, errors = model.predict(xs)
        grads = model.grad_x(xs, errors)
        for l in range(model.L + 1):
            if l not in clamped:
                xs[l] = xs[l] - gamma * grads[l]
        record()
    return xs, energies, norms


@torch.no_grad()
def classify_batch(
    model: PCN, images: torch.Tensor, T: int | None = None, gamma: float | None = None
) -> torch.Tensor:
    """Return predicted class indices for a batch of images."""
    xs = model.init_states(images.shape[0], x_bottom=images, feedforward=True)
    settled, _ = model.infer(xs, clamped={0}, T=T, gamma=gamma)
    return settled[model.L].argmax(dim=1)


@torch.no_grad()
def classify_accuracy(
    model: PCN,
    loader: Iterable,
    device: str | torch.device = "cpu",
    T: int | None = None,
    gamma: float | None = None,
) -> float:
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        preds = classify_batch(model, images, T=T, gamma=gamma)
        correct += int((preds == labels).sum())
        total += labels.shape[0]
    return correct / max(total, 1)


@torch.no_grad()
def reconstruct_batch(
    model: PCN, images: torch.Tensor, T: int | None = None, gamma: float | None = None
) -> torch.Tensor:
    """Reconstruct images: clamp image, settle, return mu[0] (prediction of x[0]).

    The reconstruction is the top-down prediction of the bottom layer from the
    settled hidden state -- it shows what the model "believes" the image is.
    """
    xs = model.init_states(images.shape[0], x_bottom=images, feedforward=True)
    settled, _ = model.infer(xs, clamped={0}, T=T, gamma=gamma)
    mus, _ = model.predict(settled)
    return mus[0]


@torch.no_grad()
def generate_from_labels(
    model: PCN, targets: torch.Tensor, T: int | None = None, gamma: float | None = None
) -> torch.Tensor:
    """Class-conditioned generation: clamp the top one-hot, free the bottom.

    Relaxing with the label clamped and the image end free lets the network
    "dream" an image for the class. The settled bottom value ``x[0]`` is the
    generated digit. ``targets`` is a ``[B, num_classes]`` one-hot tensor.
    """
    batch = targets.shape[0]
    # Start hidden/bottom from zeros, then overwrite the top with the clamp.
    xs = model.init_states(batch, x_bottom=None, x_top=targets, feedforward=False)
    settled, _ = model.infer(xs, clamped={model.L}, T=T, gamma=gamma)
    return settled[0]


@torch.no_grad()
def reconstruction_mse(
    model: PCN,
    loader: Iterable,
    device: str | torch.device = "cpu",
    T: int | None = None,
    gamma: float | None = None,
) -> float:
    model.eval()
    total_se, total_n = 0.0, 0
    for images, _ in loader:
        images = images.to(device)
        recon = reconstruct_batch(model, images, T=T, gamma=gamma)
        total_se += float(((recon - images) ** 2).sum())
        total_n += images.numel()
    return total_se / max(total_n, 1)


@torch.no_grad()
def extract_features(
    model: PCN,
    loader: Iterable,
    layer: int | None = None,
    device: str | torch.device = "cpu",
    T: int | None = None,
    gamma: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Settle each batch (image clamped, top free) and collect a hidden layer.

    Returns ``(features, labels)`` as CPU tensors. ``layer`` defaults to the top.
    """
    model.eval()
    layer = model.L if layer is None else layer
    feats, ys = [], []
    for images, labels in loader:
        images = images.to(device)
        xs = model.init_states(images.shape[0], x_bottom=images, feedforward=True)
        settled, _ = model.infer(xs, clamped={0}, T=T, gamma=gamma)
        feats.append(settled[layer].cpu())
        ys.append(labels.cpu())
    return torch.cat(feats), torch.cat(ys)


def linear_probe(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
) -> float:
    """Fit logistic regression on features; return test accuracy."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(train_feats.numpy(), train_labels.numpy())
    return float(clf.score(test_feats.numpy(), test_labels.numpy()))
