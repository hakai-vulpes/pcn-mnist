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

from pcn.model import PCN


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
