"""Generate all Phase 6 figures from the trained models and logs.

Produces, under ``outputs/figures/``:
  - training_curves.png   PCN vs backprop MLP validation accuracy
  - energy_descent.png     energy F during inference (the "settling")
  - layer_error_norms.png  per-layer ||eps|| during inference
  - confusion_matrix.png   supervised PCN test confusion matrix
  - receptive_fields.png   generative model's first-layer receptive fields
  - reconstructions.png     original vs reconstructed test digits
  - generations.png         class-conditioned samples

Run after the supervised, baseline, and generative experiments. Missing inputs
are skipped with a note rather than crashing.

Usage:
    uv run python experiments/make_figures.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from pcn.data import load_mnist, onehot_targets
from pcn.eval import (
    classify_batch,
    generate_from_labels,
    reconstruct_batch,
    settle_with_diagnostics,
)
from pcn.model import PCN
from pcn.utils import get_device, load_config
from pcn.viz import (
    plot_confusion_matrix,
    plot_energy_descent,
    plot_image_grid,
    plot_layer_error_norms,
    plot_receptive_fields,
    plot_reconstruction_pairs,
    plot_training_curves,
)

FIG = Path("outputs/figures")


def _read_curve(path: Path, value_col: str) -> tuple[list[float], list[float]]:
    if not path.exists():
        return [], []
    xs, ys = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            xs.append(float(row["epoch"]))
            ys.append(float(row[value_col]))
    return xs, ys


def _load(cfg_path: str, weights: Path, direction: str, device) -> PCN | None:
    if not weights.exists():
        print(f"  (skip) no weights at {weights}")
        return None
    cfg = load_config(cfg_path)
    model = PCN(
        dims=cfg["dims"],
        activation=cfg["activation"],
        init=cfg["init"],
        init_std=cfg["init_std"],
        direction=direction,
        gamma=cfg["gamma"],
        T=cfg["T"],
        seed=cfg["seed"],
        device=device,
    )
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    return model, cfg


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    device = get_device("auto")
    data = load_mnist()
    _, _, test_loader = data.loaders(batch_size=256)

    # --- Training curves: PCN vs backprop -------------------------------- #
    ipc_x, ipc_y = _read_curve(Path("outputs/supervised_ipc/curve.csv"), "val_acc")
    pcn_x, pcn_y = _read_curve(Path("outputs/supervised/curve.csv"), "val_acc")
    mlp_x, mlp_y = _read_curve(Path("outputs/baseline/curve.csv"), "val_acc")
    curves = {}
    if ipc_x:
        curves["PCN — incremental PC (local)"] = (ipc_x, ipc_y)
    if pcn_x:
        curves["PCN — standard PC (local)"] = (pcn_x, pcn_y)
    if mlp_x:
        curves["MLP (backprop)"] = (mlp_x, mlp_y)
    if curves:
        plot_training_curves(curves, FIG / "training_curves.png")
        print("wrote training_curves.png")

    # --- Supervised model: settling, confusion matrix -------------------- #
    # The headline classifier is the incremental-PC model (≥97% test); fall
    # back to the standard-PC run, then to its last-epoch checkpoint.
    sup = _load(
        "configs/supervised_ipc.yaml",
        Path("outputs/supervised_ipc/best.pt"),
        "discriminative",
        device,
    )
    if sup is None:
        sup = _load(
            "configs/supervised.yaml",
            Path("outputs/supervised/best.pt"),
            "discriminative",
            device,
        )
    if sup is None:
        sup = _load(
            "configs/supervised.yaml",
            Path("outputs/supervised/model.pt"),
            "discriminative",
            device,
        )
    if sup is not None:
        model, cfg = sup
        images, labels = next(iter(test_loader))
        images = images.to(device)
        tgt = onehot_targets(labels, 10, -1, 1).to(device)
        # Settling with image + true label clamped (the inference phase of
        # learning) shows the energy descending to a fixed point.
        xs = model.init_states(
            images.shape[0], x_bottom=images, x_top=tgt, feedforward=True
        )
        _, energies, norms = settle_with_diagnostics(
            model, xs, clamped={0, model.L}, T=cfg["T_test"]
        )
        plot_energy_descent(energies, FIG / "energy_descent.png")
        plot_layer_error_norms(norms, FIG / "layer_error_norms.png")
        print("wrote energy_descent.png, layer_error_norms.png")

        # Confusion matrix over the full test set.
        preds, ys = [], []
        for imgs, lbls in test_loader:
            preds.append(classify_batch(model, imgs.to(device), T=cfg["T_test"]).cpu())
            ys.append(lbls)
        plot_confusion_matrix(
            torch.cat(ys), torch.cat(preds), FIG / "confusion_matrix.png"
        )
        print("wrote confusion_matrix.png")

    # --- Generative model: receptive fields, reconstructions, samples ---- #
    gen = _load(
        "configs/generative.yaml",
        Path("outputs/generative/model.pt"),
        "generative",
        device,
    )
    if gen is not None:
        model, cfg = gen
        # W[0] maps layer 1 -> image (shape [784, dim1]); columns are RFs.
        plot_receptive_fields(model.W[0].detach(), FIG / "receptive_fields.png")

        images, _ = next(iter(test_loader))
        images = images.to(device)
        recon = reconstruct_batch(model, images, T=cfg["T_test"])
        plot_reconstruction_pairs(
            images.cpu(), recon.cpu(), FIG / "reconstructions.png", n=10
        )

        # Generation needs more settling than reconstruction (the whole image
        # is free), so use a longer relaxation here than cfg["T_test"].
        gen_T = max(cfg["T_test"], 500)
        labels = torch.arange(10).repeat_interleave(8)
        onehots = onehot_targets(labels, 10, -1, 1).to(device)
        gens = generate_from_labels(model, onehots, T=gen_T)
        plot_image_grid(
            gens.cpu(),
            FIG / "generations.png",
            nrow=8,
            title="Class-conditioned samples (rows: classes 0-9)",
        )
        print("wrote receptive_fields.png, reconstructions.png, generations.png")

    print(f"\nfigures in {FIG}/")


if __name__ == "__main__":
    main()
