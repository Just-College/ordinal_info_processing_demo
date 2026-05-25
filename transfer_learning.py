import csv
import time

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import make_transfer_sequences, SyntheticSequenceDataset, collate_fn
from models import CustomContinuousRNN
from train_utils import (
    CSV_DIR,
    FIGURE_DIR,
    MODEL_DIR,
    ensure_output_dirs,
    evaluate_model,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


PRETRAINED_CHECKPOINT = MODEL_DIR / "synthetic_latest.pt"
TRANSFER_MODES = ["scratch_all", "pretrained_template_frozen"]
INPUT_DIM = 16
SAMPLES_PER_CLASS = 128
TEST_SAMPLES_PER_CLASS = 216
EPOCHS = 60
BATCH_SIZE = 16
LR = 1e-3
HIDDEN_DIM = None
TAU = 2.0
T_MAX = 30
PHONEME_LEN = 3
NOISE_STD = 0.01
SEED = 11

MODES = {
    "scratch_all": {
        "use_pretrained": False,
        "freeze_template": False,
        "label": "scratch, train all",
    },
    "pretrained_template_frozen": {
        "use_pretrained": True,
        "freeze_template": True,
        "label": "pretrained template, freeze Wrec/Wout",
    },
    "pretrained_template_finetune": {
        "use_pretrained": True,
        "freeze_template": False,
        "label": "pretrained template, finetune all",
    },
}


def copy_template_weights(target_model, source_state):
    copied = []
    target_state = target_model.state_dict()
    with torch.no_grad():
        for name in ["W_rec", "I_b", "W_out.weight", "W_out.bias"]:
            if name in source_state and source_state[name].shape == target_state[name].shape:
                target_state[name].copy_(source_state[name])
                copied.append(name)
    target_model.load_state_dict(target_state)
    return copied


def freeze_template(model):
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("W_in")


def train_mode(mode_name, source_state, loaders, device, hidden_dim):
    mode = MODES[mode_name]
    train_loader, test_loader = loaders
    set_seed(SEED)

    model = CustomContinuousRNN(
        input_dim=INPUT_DIM,
        hidden_dim=hidden_dim,
        output_dim=4,
        tau=TAU,
    ).to(device)
    copied = []
    if mode["use_pretrained"]:
        copied = copy_template_weights(model, source_state)
        if mode["freeze_template"]:
            freeze_template(model)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    loss_fn = nn.MSELoss()
    history = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        train_loss /= len(train_loader)
        test_loss, test_acc = evaluate_model(
            model,
            test_loader,
            loss_fn,
            device,
            T_MAX,
            PHONEME_LEN,
        )
        lr = scheduler.get_last_lr()[0]
        history.append([epoch + 1, train_loss, test_loss, test_acc, lr])
        print(
            f"{mode_name} epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, test_loss={test_loss:.4f}, "
            f"test_acc={test_acc:.2f}%, lr={lr:.6g}"
        )

    metadata = {
        "task": "transfer_learning",
        "mode": mode_name,
        "copied_template_weights": copied,
        "input_dim": INPUT_DIM,
        "hidden_dim": hidden_dim,
        "output_dim": 4,
        "tau": TAU,
        "t_max": T_MAX,
        "phoneme_len": PHONEME_LEN,
        "noise_std": NOISE_STD,
        "seed": SEED,
    }
    save_checkpoint(model, MODEL_DIR / f"transfer_{mode_name}_latest.pt", metadata=metadata)
    return history


def save_history(mode_name, history, timestamp):
    path = CSV_DIR / f"transfer_{mode_name}_{timestamp}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_loss", "test_acc", "lr"])
        writer.writerows(history)
    return path


def main():
    unknown_modes = [mode for mode in TRANSFER_MODES if mode not in MODES]
    if unknown_modes:
        raise ValueError(f"Unknown modes: {unknown_modes}. Valid modes: {list(MODES)}")

    ensure_output_dirs()
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source_state, source_metadata = load_checkpoint(PRETRAINED_CHECKPOINT, device)
    source_hidden_dim = int(source_metadata.get("hidden_dim", HIDDEN_DIM or 50))
    hidden_dim = source_hidden_dim if HIDDEN_DIM is None else HIDDEN_DIM
    if source_hidden_dim != hidden_dim:
        raise ValueError(
            f"Pretrained hidden_dim={source_hidden_dim}, but transfer hidden_dim={hidden_dim}."
        )

    sequences = make_transfer_sequences(input_dim=INPUT_DIM, seed=SEED)
    train_dataset = SyntheticSequenceDataset(
        sequences,
        SAMPLES_PER_CLASS,
        T_MAX,
        NOISE_STD,
        phoneme_len=PHONEME_LEN,
        fixed=False,
    )
    test_dataset = SyntheticSequenceDataset(
        sequences,
        TEST_SAMPLES_PER_CLASS,
        T_MAX,
        NOISE_STD,
        phoneme_len=PHONEME_LEN,
        fixed=True,
    )
    loaders = (
        DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn),
        DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn),
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    histories = {}
    csv_paths = {}
    for mode_name in TRANSFER_MODES:
        histories[mode_name] = train_mode(mode_name, source_state, loaders, device, hidden_dim)
        csv_paths[mode_name] = save_history(mode_name, histories[mode_name], timestamp)

    plt.figure(figsize=(8, 5))
    for mode_name, history in histories.items():
        epochs = [row[0] for row in history]
        acc = [row[3] for row in history]
        plt.plot(epochs, acc, linewidth=2, label=MODES[mode_name]["label"])
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy (%)")
    plt.title("Transfer Learning Effect")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_fig = FIGURE_DIR / f"transfer_learning_accuracy_{timestamp}.png"
    plt.savefig(out_fig, dpi=220)

    for mode_name, path in csv_paths.items():
        print(f"Saved {mode_name} history: {path}")
    print(f"Saved transfer comparison figure: {out_fig}")


if __name__ == "__main__":
    main()
