import sys
sys.path.insert(0, "src")

import os
from pathlib import Path
import pandas as pd
import torch

from config import config
from seed import set_seed
from transforms import build_transforms
from data import (
    build_datasets,
    split_val_test,
    build_train_val_dataloaders,
    build_test_dataloader,
)
from models import MobileNet
from utils import get_device, build_criterion, build_optimizer
from engine import train_one_epoch, evaluate_one_epoch


def save_result(csv_path, row):
    df = pd.DataFrame([row])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


def main():
    model_dir = Path("model")
    results_dir = Path("results")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    set_seed(config["seed"])
    device = get_device()

    data_root = Path("data")
    epochs = config["epochs"]

    csv_path = str(results_dir / "mobilenet_results.csv")

    for a in config["alpha_list"]:
        for p in config["p_list"]:
            train_tf, eval_tf = build_transforms(p)

            train_set, val_full = build_datasets(str(data_root), train_tf, eval_tf)
            val_set, test_set = split_val_test(val_full)

            train_loader, val_loader = build_train_val_dataloaders(
                train_set,
                val_set,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                pin_memory=config["pin_memory"],
            )
            test_loader = build_test_dataloader(
                test_set,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                pin_memory=config["pin_memory"],
            )

            model = MobileNet(alpha=a, num_classes=200).to(device)
            criterion = build_criterion()
            optimizer = build_optimizer(
                model,
                lr=config["lr"],
                weight_decay=config["weight_decay"],
            )

            best_val_acc = -1.0
            best_val_loss = None
            best_path = model_dir / f"mobilenet_a{fmt(a)}_p{fmt(p)}_best.pth"

            for epoch in range(1, epochs + 1):
                tr_loss, tr_acc, tr_t = train_one_epoch(
                    model, train_loader, optimizer, criterion, device, epoch, epochs
                )
                va_loss, va_acc, va_t = evaluate_one_epoch(
                    model, val_loader, criterion, device, epoch, epochs, mode="Val"
                )
                ep_t = tr_t + va_t

                print(
                    f"[a={a}, p={p}] "
                    f"[{epoch:03d}/{epochs}] "
                    f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
                    f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} "
                    f"time={ep_t:.1f}s"
                )

                if va_acc > best_val_acc:
                    best_val_acc = va_acc
                    best_val_loss = va_loss
                    torch.save(model.state_dict(), str(best_path))

            model.load_state_dict(torch.load(str(best_path), map_location=device))
            te_loss, te_acc, te_t = evaluate_one_epoch(
                model, test_loader, criterion, device, 1, 1, mode="Test"
            )

            print(f"[a={a}, p={p}] [BEST] val_acc={best_val_acc:.4f}")
            print(f"[a={a}, p={p}] [TEST] test_loss={te_loss:.4f} test_acc={te_acc:.4f} time={te_t:.1f}s")

            row = {
                "model": "MobileNet",
                "alpha": a,
                "p": p,
                "seed": config["seed"],
                "epochs": epochs,
                "batch_size": config["batch_size"],
                "lr": config["lr"],
                "weight_decay": config["weight_decay"],
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "test_loss": te_loss,
                "test_acc": te_acc,
            }
            save_result(csv_path, row)

    print(f"[CSV] {csv_path}")


if __name__ == "__main__":
    main()