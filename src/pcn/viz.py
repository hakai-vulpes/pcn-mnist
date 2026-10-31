"""Visualization helpers for PCN experiments.

All functions save a figure to ``path`` and return it. They use a non-interactive
matplotlib backend so they work headless.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from pcn.data import MNIST_MEAN, MNIST_STD  # noqa: E402


def _unstandardize(x: torch.Tensor) -> np.ndarray:
    """Map standardized pixels back to [0,1] for display."""
    img = x.detach().cpu().float() * MNIST_STD + MNIST_MEAN
    return img.clamp(0, 1).numpy()


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_energy_descent(
    energies: list[float], path: str | Path, title: str = "Energy during inference"
) -> Path:
    """The 'settling' curve: F vs inference step."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(energies, lw=2)
    ax.set_xlabel("inference step")
    ax.set_ylabel("energy F")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_layer_error_norms(
    error_norms: dict[int, list[float]], path: str | Path
) -> Path:
    """Per-layer ||eps[l]|| over inference steps."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for l, norms in sorted(error_norms.items()):
        ax.plot(norms, lw=2, label=f"layer {l}")
    ax.set_xlabel("inference step")
    ax.set_ylabel("||eps[l]||")
    ax.set_title("Per-layer error norms during inference")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_training_curves(
    curves: dict[str, tuple[list[float], list[float]]],
    path: str | Path,
    ylabel: str = "validation accuracy",
) -> Path:
    """Overlay multiple (epochs, values) curves on one axis (PCN vs backprop)."""
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for label, (xs, ys) in curves.items():
        ax.plot(xs, ys, lw=2, marker="o", ms=3, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title("Training curves")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_image_grid(
    images: torch.Tensor, path: str | Path, nrow: int = 10, title: str | None = None
) -> Path:
    """Grid of MNIST images given as standardized [N, 784] rows."""
    n = images.shape[0]
    ncol = nrow
    nrows = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrows, ncol, figsize=(ncol * 0.9, nrows * 0.9))
    axes = np.atleast_2d(axes)
    for i in range(nrows * ncol):
        ax = axes.flat[i]
        ax.axis("off")
        if i < n:
            ax.imshow(
                _unstandardize(images[i]).reshape(28, 28), cmap="gray", vmin=0, vmax=1
            )
    if title:
        fig.suptitle(title)
    return _save(fig, path)


def plot_reconstruction_pairs(
    originals: torch.Tensor, recons: torch.Tensor, path: str | Path, n: int = 10
) -> Path:
    """Two rows: originals (top) and reconstructions (bottom)."""
    n = min(n, originals.shape[0])
    fig, axes = plt.subplots(2, n, figsize=(n * 0.9, 2.2))
    for i in range(n):
        axes[0, i].imshow(
            _unstandardize(originals[i]).reshape(28, 28), cmap="gray", vmin=0, vmax=1
        )
        axes[1, i].imshow(
            _unstandardize(recons[i]).reshape(28, 28), cmap="gray", vmin=0, vmax=1
        )
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("orig")
    axes[1, 0].set_ylabel("recon")
    fig.suptitle("Original (top) vs reconstruction (bottom)")
    return _save(fig, path)


def plot_receptive_fields(W0: torch.Tensor, path: str | Path, n: int = 64) -> Path:
    """First-layer receptive fields: columns of W[0] (each maps to a 28x28 image).

    ``W[0]`` has shape ``[784, dim_1]``; column ``j`` is the image-space pattern
    that hidden unit ``j`` contributes top-down.
    """
    W0 = W0.detach().cpu()
    n = min(n, W0.shape[1])
    ncol = int(np.ceil(np.sqrt(n)))
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 0.8, nrow * 0.8))
    axes = np.atleast_2d(axes)
    for i in range(nrow * ncol):
        ax = axes.flat[i]
        ax.axis("off")
        if i < n:
            rf = W0[:, i].reshape(28, 28).numpy()
            lim = np.abs(rf).max() + 1e-8
            ax.imshow(rf, cmap="RdBu_r", vmin=-lim, vmax=lim)
    fig.suptitle("First-layer receptive fields (columns of W[0])")
    return _save(fig, path)


def plot_confusion_matrix(
    y_true: torch.Tensor, y_pred: torch.Tensor, path: str | Path, num_classes: int = 10
) -> Path:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true.tolist(), y_pred.tolist(), strict=True):
        cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_title("Confusion matrix")
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=7,
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, path)


def plot_tsne(
    features: torch.Tensor, labels: torch.Tensor, path: str | Path, seed: int = 0
) -> Path:
    """2-D t-SNE of latent features colored by class."""
    from sklearn.manifold import TSNE

    emb = TSNE(
        n_components=2, init="pca", random_state=seed, perplexity=30
    ).fit_transform(features.detach().cpu().numpy())
    fig, ax = plt.subplots(figsize=(5.5, 5))
    sc = ax.scatter(
        emb[:, 0], emb[:, 1], c=labels.cpu().numpy(), cmap="tab10", s=6, alpha=0.7
    )
    ax.set_title("t-SNE of settled latent representations")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(sc, ax=ax, ticks=range(10), fraction=0.046)
    return _save(fig, path)
