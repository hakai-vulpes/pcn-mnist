"""Train a supervised PCN on full MNIST and report test accuracy.

Usage:
    uv run python experiments/run_supervised.py --config configs/supervised.yaml
    uv run python experiments/run_supervised.py --epochs 10 --T 40   # overrides
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from pcn.data import load_mnist
from pcn.eval import classify_accuracy
from pcn.model import PCN
from pcn.train import train_supervised_epoch
from pcn.utils import CSVLogger, get_device, load_config, merge_overrides, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/supervised.yaml")
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
        direction=cfg.get("direction", "discriminative"),
        gamma=cfg["gamma"],
        T=cfg["T"],
        seed=cfg["seed"],
        device=device,
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    # Optional cosine LR decay: PC inference-learning fits slower than backprop,
    # so we train for more epochs and anneal the weight LR for a stable landing.
    sched = None
    if cfg.get("lr_schedule") == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg["epochs"], eta_min=cfg.get("lr_min", 1e-5)
        )

    logger = CSVLogger(
        out_dir / "curve.csv", ["epoch", "train_energy", "val_acc", "seconds"]
    )
    best_val = 0.0
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
            incremental=cfg.get("incremental", False),
        )
        val_acc = classify_accuracy(model, val_loader, device=device, T=cfg["T_test"])
        dt = time.time() - t0
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), out_dir / "best.pt")
        logger.log(
            epoch=epoch, train_energy=energy, val_acc=val_acc, seconds=round(dt, 1)
        )
        print(
            f"epoch {epoch:3d}  energy/ex {energy:8.3f}  val_acc {val_acc:.4f}  ({dt:.1f}s)"
        )
        if sched is not None:
            sched.step()
        torch.save(model.state_dict(), out_dir / "model.pt")

    # Report the best-val checkpoint (val accuracy oscillates between epochs).
    best_path = out_dir / "best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    test_acc = classify_accuracy(model, test_loader, device=device, T=cfg["T_test"])
    print(f"\nFINAL  test_acc {test_acc:.4f}  best_val {best_val:.4f}")
    (out_dir / "results.json").write_text(
        json.dumps(
            {"test_acc": test_acc, "best_val_acc": best_val, "config": cfg}, indent=2
        )
    )


if __name__ == "__main__":
    main()
