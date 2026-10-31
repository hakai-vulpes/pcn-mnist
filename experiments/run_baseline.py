"""Train the same-architecture backprop MLP baseline on MNIST.

Usage:
    uv run python experiments/run_baseline.py --config configs/supervised.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from pcn.baseline import MLP, mlp_accuracy, train_mlp_epoch
from pcn.data import load_mnist
from pcn.utils import CSVLogger, get_device, load_config, merge_overrides, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/supervised.yaml")
    p.add_argument("--seed", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--loss", choices=["ce", "mse"], default="ce")
    p.add_argument("--device")
    p.add_argument("--out_dir", default="outputs/baseline")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = merge_overrides(load_config(args.config), vars(args))
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  dims={cfg['dims']}  activation={cfg['activation']}")

    data = load_mnist()
    train_loader, val_loader, test_loader = data.loaders(batch_size=cfg["batch_size"])

    model = MLP(dims=cfg["dims"], activation=cfg["activation"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    logger = CSVLogger(
        out_dir / "curve.csv", ["epoch", "train_loss", "val_acc", "seconds"]
    )
    best_val = 0.0
    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        loss = train_mlp_epoch(
            model, train_loader, optimizer=opt, device=device, loss=cfg["loss"]
        )
        val_acc = mlp_accuracy(model, val_loader, device=device)
        dt = time.time() - t0
        best_val = max(best_val, val_acc)
        logger.log(epoch=epoch, train_loss=loss, val_acc=val_acc, seconds=round(dt, 1))
        print(f"epoch {epoch:3d}  loss {loss:.4f}  val_acc {val_acc:.4f}  ({dt:.1f}s)")
        torch.save(model.state_dict(), out_dir / "model.pt")

    test_acc = mlp_accuracy(model, test_loader, device=device)
    print(f"\nFINAL  test_acc {test_acc:.4f}  best_val {best_val:.4f}")
    (out_dir / "results.json").write_text(
        json.dumps(
            {"test_acc": test_acc, "best_val_acc": best_val, "config": cfg}, indent=2
        )
    )


if __name__ == "__main__":
    main()
