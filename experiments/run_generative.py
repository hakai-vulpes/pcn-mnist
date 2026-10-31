"""Train a generative PCN (top-down) with image + label clamped.

This is the generative counterpart to the supervised classifier: same dims, but
predictions flow label -> image, so the model learns to *reconstruct* and to
*generate* class-conditioned digits. We then measure:

  (a) reconstruction MSE on the test set,
  (b) class-conditioned generation (saved as a digit grid),
  (c) a linear probe on settled hidden representations vs a random-net baseline,
  (d) t-SNE of those hidden representations.

Usage:
    uv run python experiments/run_generative.py --config configs/generative.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from pcn.data import load_mnist, onehot_targets
from pcn.eval import (
    extract_features,
    generate_from_labels,
    linear_probe,
    reconstruction_mse,
)
from pcn.model import PCN
from pcn.train import train_supervised_epoch
from pcn.utils import CSVLogger, get_device, load_config, merge_overrides, set_seed
from pcn.viz import plot_image_grid, plot_tsne


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/generative.yaml")
    p.add_argument("--seed", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--T", type=int)
    p.add_argument("--T_test", type=int)
    p.add_argument("--gamma", type=float)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--probe_layer", type=int)
    p.add_argument("--device")
    p.add_argument("--out_dir")
    return p.parse_args()


def _build(cfg: dict, device, seed: int) -> PCN:
    return PCN(
        dims=cfg["dims"],
        activation=cfg["activation"],
        init=cfg["init"],
        init_std=cfg["init_std"],
        direction="generative",
        gamma=cfg["gamma"],
        T=cfg["T"],
        seed=seed,
        device=device,
    )


def main() -> None:
    args = parse_args()
    cfg = merge_overrides(load_config(args.config), vars(args))
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_layer = cfg.get("probe_layer", cfg["dims"].__len__() - 2)
    print(f"device={device}  config={cfg}")

    data = load_mnist()
    train_loader, val_loader, test_loader = data.loaders(batch_size=cfg["batch_size"])

    model = _build(cfg, device, cfg["seed"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    logger = CSVLogger(
        out_dir / "curve.csv", ["epoch", "train_energy", "val_mse", "seconds"]
    )
    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        energy = train_supervised_epoch(
            model,
            train_loader,
            optimizer=opt,
            device=device,
            T=cfg["T"],
            gamma=cfg["gamma"],
            target_low=cfg["target_low"],
            target_high=cfg["target_high"],
        )
        val_mse = reconstruction_mse(model, val_loader, device=device, T=cfg["T_test"])
        dt = time.time() - t0
        logger.log(
            epoch=epoch, train_energy=energy, val_mse=val_mse, seconds=round(dt, 1)
        )
        print(
            f"epoch {epoch:3d}  energy/ex {energy:8.3f}  val_mse {val_mse:.4f}  ({dt:.1f}s)"
        )
        torch.save(model.state_dict(), out_dir / "model.pt")

    # (a) reconstruction MSE on the test set.
    test_mse = reconstruction_mse(model, test_loader, device=device, T=cfg["T_test"])

    # (b) class-conditioned generation: clamp each one-hot, free the image end.
    labels = torch.arange(10).repeat_interleave(8)  # 8 samples per class
    onehots = onehot_targets(labels, 10, cfg["target_low"], cfg["target_high"]).to(
        device
    )
    gens = generate_from_labels(model, onehots, T=cfg["T_test"])
    plot_image_grid(
        gens.cpu(),
        out_dir / "generations.png",
        nrow=8,
        title="Class-conditioned samples",
    )

    # (c) linear probe on settled hidden representations vs an untrained net.
    tr_feats, tr_y = extract_features(
        model, train_loader, layer=probe_layer, device=device, T=cfg["T_test"]
    )
    te_feats, te_y = extract_features(
        model, test_loader, layer=probe_layer, device=device, T=cfg["T_test"]
    )
    probe_acc = linear_probe(tr_feats, tr_y, te_feats, te_y)

    rand_model = _build(cfg, device, cfg["seed"] + 1)
    rtr, rtr_y = extract_features(
        rand_model, train_loader, layer=probe_layer, device=device, T=cfg["T_test"]
    )
    rte, rte_y = extract_features(
        rand_model, test_loader, layer=probe_layer, device=device, T=cfg["T_test"]
    )
    rand_acc = linear_probe(rtr, rtr_y, rte, rte_y)

    # (d) t-SNE of a subset of the test representations.
    plot_tsne(te_feats[:2000], te_y[:2000], out_dir / "tsne.png", seed=cfg["seed"])

    print(
        f"\nFINAL  test_recon_mse {test_mse:.4f}  probe_layer {probe_layer}  "
        f"linear_probe_acc {probe_acc:.4f}  random_features_acc {rand_acc:.4f}"
    )
    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "test_recon_mse": test_mse,
                "probe_layer": probe_layer,
                "linear_probe_acc": probe_acc,
                "random_features_acc": rand_acc,
                "config": cfg,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
