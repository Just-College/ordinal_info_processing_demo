import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"
CSV_DIR = OUTPUT_DIR / "csv"


def ensure_output_dirs():
    for path in [MODEL_DIR, FIGURE_DIR, CSV_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}
    return checkpoint_state_dict(checkpoint), metadata


def save_checkpoint(model, path, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": metadata or {},
        },
        path,
    )


def evaluate_model(model, loader, loss_fn, device, t_max, phoneme_len):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    second_start = t_max + phoneme_len + t_max
    second_end = second_start + phoneme_len

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += loss_fn(out, y).item()

            pred1 = out[:, t_max : t_max + phoneme_len, :].mean(dim=1).argmax(dim=1)
            target1 = y[:, t_max : t_max + phoneme_len, :].mean(dim=1).argmax(dim=1)
            pred2 = out[:, second_start:second_end, :].mean(dim=1).argmax(dim=1)
            target2 = y[:, second_start:second_end, :].mean(dim=1).argmax(dim=1)
            correct += (pred1 == target1).sum().item()
            correct += (pred2 == target2).sum().item()
            total += 2 * x.size(0)

    return total_loss / len(loader), 100.0 * correct / total


def train_model(
    model,
    loader_train,
    loader_test,
    optimizer,
    scheduler,
    loss_fn,
    device,
    epochs,
    t_max,
    phoneme_len,
    run_name,
    checkpoint_metadata=None,
):
    ensure_output_dirs()
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(
            loader_train,
            desc=f"{run_name} epoch {epoch + 1}/{epochs}",
            dynamic_ncols=False,
            leave=False,
        ):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if scheduler is not None:
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
        else:
            lr = optimizer.param_groups[0]["lr"]

        train_loss = total_loss / len(loader_train)
        test_loss, test_acc = evaluate_model(
            model, loader_test, loss_fn, device, t_max, phoneme_len
        )
        history.append([epoch + 1, train_loss, test_loss, test_acc, lr])
        print(
            f"{run_name} epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, test_loss={test_loss:.4f}, "
            f"test_acc={test_acc:.2f}%, lr={lr:.6g}"
        )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = CSV_DIR / f"{run_name}_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_loss", "test_acc", "lr"])
        writer.writerows(history)

    checkpoint_path = MODEL_DIR / f"{run_name}_{timestamp}.pt"
    latest_path = MODEL_DIR / f"{run_name}_latest.pt"
    metadata = checkpoint_metadata or {}
    metadata = {**metadata, "history_csv": str(csv_path)}
    save_checkpoint(model, checkpoint_path, metadata=metadata)
    save_checkpoint(model, latest_path, metadata=metadata)
    return history, checkpoint_path, latest_path, csv_path
