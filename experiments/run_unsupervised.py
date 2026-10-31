"""Train a generative/unsupervised PCN (clamp image only, free top latent).

Then measure representation quality with a linear probe and reconstruction MSE.

Usage:
    uv run python experiments/run_unsupervised.py --config configs/unsupervised.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from pcn.data import load_mnist
from pcn.eval import extract_features, linear_probe, reconstruction_mse
from pcn.model import PCN
from pcn.train import train_unsupervised_epoch
from pcn.utils import CSVLogger, get_device, load_config, merge_overrides, set_seed
from pcn.viz import plot_tsne


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/unsupervised.yaml")
    p.add_argument("--seed", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--T", type=int)
    p.add_argument("--T_test", type=int)
    p.add_argument("--gamma", type=float)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--device")
    p.add_argument("--out_dir")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = merge_overrides(load_config(args.config), vars(args))
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  config={cfg}")

    data = load_mnist()
    train_loader, val_loader, test_loader = data.loaders(batch_size=cfg["batch_size"])

    model = PCN(
        dims=cfg["dims"],
        activation=cfg["activation"],
        init=cfg["init"],
        init_std=cfg["init_std"],
        gamma=cfg["gamma"],
        T=cfg["T"],
        seed=cfg["seed"],
        device=device,
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    logger = CSVLogger(
        out_dir / "curve.csv", ["epoch", "train_energy", "val_mse", "seconds"]
    )
    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        energy = train_unsupervised_epoch(
            model,
            train_loader,
            optimizer=opt,
            device=device,
            T=cfg["T"],
            gamma=cfg["gamma"],
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

    # Representation quality: linear probe on settled top-layer features.
    print("extracting features for linear probe ...")
    train_feats, train_y = extract_features(
        model, train_loader, device=device, T=cfg["T_test"]
    )
    test_feats, test_y = extract_features(
        model, test_loader, device=device, T=cfg["T_test"]
    )
    probe_acc = linear_probe(train_feats, train_y, test_feats, test_y)

    # Random-features baseline: probe on an untrained net's representations.
    rand_model = PCN(
        dims=cfg["dims"],
        activation=cfg["activation"],
        init=cfg["init"],
        init_std=cfg["init_std"],
        gamma=cfg["gamma"],
        T=cfg["T"],
        seed=cfg["seed"] + 1,
        device=device,
    )
    rtr, rtr_y = extract_features(
        rand_model, train_loader, device=device, T=cfg["T_test"]
    )
    rte, rte_y = extract_features(
        rand_model, test_loader, device=device, T=cfg["T_test"]
    )
    rand_acc = linear_probe(rtr, rtr_y, rte, rte_y)

    # t-SNE of the learned top-latent code, colored by class (Phase 5d).
    plot_tsne(test_feats[:2000], test_y[:2000], out_dir / "tsne.png", seed=cfg["seed"])

    test_mse = reconstruction_mse(model, test_loader, device=device, T=cfg["T_test"])
    print(
        f"\nFINAL  test_recon_mse {test_mse:.4f}  "
        f"linear_probe_acc {probe_acc:.4f}  random_features_acc {rand_acc:.4f}"
    )
    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "test_recon_mse": test_mse,
                "linear_probe_acc": probe_acc,
                "random_features_acc": rand_acc,
                "config": cfg,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
