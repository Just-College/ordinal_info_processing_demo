import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import TimitSequenceDataset, collate_fn
from load_sec3_data import build_sec3_timit_sequences
from models import SimpleRNN

# ----------------------------
# Global Configurations
# ----------------------------
REPO_ROOT = Path(__file__).parent.parent
MODEL_DIR = f"{REPO_ROOT}/model/"
CSV_DIR = f"{REPO_ROOT}/csv/"
PRETRAINED_MODEL = f"{MODEL_DIR}/rnn_mfcc3_hid64_spc128_sec2_seed1107.pt"

# fmt: off
SAMPLE_PER_CLASS = 128          # 训练集每类样本数
TEST_SAMPLE_PER_CLASS = 1024    # 测试集每类样本数
T_MAX = 30                      # 随机噪声最大长度
PHONEME_LEN = 3                 # 每个音素持续时间
NOISE_STD = 0.01                # 噪声标准差
N_MFCC = 16                     # TIMIT MFCC 输入维度
HIDDEN_DIM = 64                 # 必须匹配 sec2 pretrained W_rec
TAU = 2                         # RNN时间常数
BATCH_SIZE = 16                 # 批大小
EPOCHS = 120                    # 训练轮数
LR = 1e-3
SEED = 1107
GRAD_CLIP_NORM = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# fmt: on

MODES = ["scratch_all", "sec2_pretrained_freeze_wrec"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_mse_loss(out, target, mask):
    mask = mask.to(out.device).unsqueeze(-1)
    return ((out - target) ** 2 * mask).sum() / (mask.sum() * target.shape[-1])


def evaluate_final_readout(out, class_ids, lengths, window=5):
    correct = 0
    total = 0
    for i in range(out.shape[0]):
        class_id = int(class_ids[i].item())
        true_len = int(lengths[i].item())

        readout_start = max(0, true_len - window)
        readout_end = true_len
        readout_mean = out[i, readout_start:readout_end, :].mean(dim=0)
        pred_class = readout_mean.argmax().item()
        correct += int(pred_class == class_id)
        total += 1
    return correct, total


def build_loaders(sequences):
    train_dataset = TimitSequenceDataset(
        sequences,
        SAMPLE_PER_CLASS,
        T_MAX,
        NOISE_STD,
        phoneme_len=PHONEME_LEN,
        fixed=False,
    )
    test_dataset = TimitSequenceDataset(
        sequences,
        TEST_SAMPLE_PER_CLASS,
        T_MAX,
        NOISE_STD,
        phoneme_len=PHONEME_LEN,
        fixed=True,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )
    return train_loader, test_loader


def load_pretrained_wrec(model):
    pretrained = torch.load(PRETRAINED_MODEL, map_location=DEVICE)
    state_dict = (
        pretrained.get("state_dict", pretrained)
        if isinstance(pretrained, dict)
        else pretrained
    )
    pretrained_wrec = state_dict["W_rec"]
    if model.W_rec.shape != pretrained_wrec.shape:
        raise ValueError(
            f"W_rec shape mismatch: pretrained {tuple(pretrained_wrec.shape)} "
            f"vs model {tuple(model.W_rec.shape)}"
        )
    with torch.no_grad():
        model.W_rec.copy_(pretrained_wrec)
    print(f"Loaded pretrained W_rec from: {PRETRAINED_MODEL}")
    return pretrained_wrec.detach().clone()


def configure_model(mode, output_dim):
    model = SimpleRNN(N_MFCC, HIDDEN_DIM, output_dim, tau=TAU).to(DEVICE)
    if mode == "scratch_all":
        for param in model.parameters():
            param.requires_grad = True
    elif mode == "sec2_pretrained_freeze_wrec":
        load_pretrained_wrec(model)
        for name, param in model.named_parameters():
            param.requires_grad = name != "W_rec"
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return model


def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, mask, class_ids, _, lengths in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            mask = mask.to(DEVICE)
            out = model(x)
            total_loss += masked_mse_loss(out, y, mask).item()
            batch_correct, batch_total = evaluate_final_readout(
                out.cpu(), class_ids, lengths
            )
            correct += batch_correct
            total += batch_total
    return total_loss / len(loader), correct / total * 100.0


def train_one_mode(mode, train_loader, test_loader, output_dim):
    set_seed(SEED)
    model = configure_model(mode, output_dim)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=LR, betas=(0.9, 0.999))
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{mode}: trainable_params={trainable_params}/{total_params}")

    rows = []
    mode_start = time.perf_counter()
    for epoch in range(EPOCHS):
        epoch_start = time.perf_counter()
        model.train()
        train_loss = 0.0
        for x, y, mask, _, _, _ in tqdm(
            train_loader,
            desc=f"{mode} epoch {epoch + 1}/{EPOCHS}",
            dynamic_ncols=False,
            leave=False,
        ):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            mask = mask.to(DEVICE)
            out = model(x)
            loss = masked_mse_loss(out, y, mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, GRAD_CLIP_NORM)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        test_loss, test_acc = evaluate(model, test_loader)
        epoch_sec = time.perf_counter() - epoch_start
        elapsed_sec = time.perf_counter() - mode_start
        rows.append(
            [
                mode,
                epoch + 1,
                train_loss,
                test_loss,
                test_acc,
                LR,
                epoch_sec,
                elapsed_sec,
                trainable_params,
                total_params,
            ]
        )
        print(
            f"{mode} epoch {epoch + 1}: train_loss={train_loss:.4f}, "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.2f}%, "
            f"lr={LR:.6g}, epoch_sec={epoch_sec:.2f}, elapsed_sec={elapsed_sec:.2f}"
        )
    return rows, model


def save_comparison(rows_by_mode, models):
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    csv_path = f"{CSV_DIR}/sec3_transfer_compare_seed{SEED}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "mode",
                "epoch",
                "train_loss",
                "test_loss",
                "test_acc",
                "lr",
                "epoch_sec",
                "elapsed_sec",
                "trainable_params",
                "total_params",
            ]
        )
        for rows in rows_by_mode.values():
            writer.writerows(rows)
    print(f"Saved comparison history: {csv_path}")

    for mode, model in models.items():
        path = f"{MODEL_DIR}/sec3_transfer_compare_{mode}_seed{SEED}.pt"
        torch.save(model.state_dict(), path)
        print(f"Saved {mode} model: {path}")
    return csv_path


def print_speed_summary(rows_by_mode):
    thresholds = [80.0, 90.0, 95.0, 99.0]
    print("Speed summary:")
    for mode, rows in rows_by_mode.items():
        best_acc = max(row[4] for row in rows)
        final_acc = rows[-1][4]
        print(f"{mode}: final_acc={final_acc:.2f}%, best_acc={best_acc:.2f}%")
        for threshold in thresholds:
            reached = [row for row in rows if row[4] >= threshold]
            if reached:
                row = reached[0]
                print(
                    f"  >= {threshold:.0f}%: epoch={row[1]}, elapsed_sec={row[7]:.2f}"
                )
            else:
                print(f"  >= {threshold:.0f}%: not reached")


def main():
    if not os.path.exists(PRETRAINED_MODEL):
        raise FileNotFoundError(f"Missing pretrained sec2 model: {PRETRAINED_MODEL}")
    os.makedirs(CSV_DIR, exist_ok=True)
    set_seed(SEED)
    sequences = build_sec3_timit_sequences()
    train_loader, test_loader = build_loaders(sequences)

    rows_by_mode = {}
    models = {}
    for mode in MODES:
        rows, model = train_one_mode(
            mode, train_loader, test_loader, output_dim=len(sequences)
        )
        rows_by_mode[mode] = rows
        models[mode] = model

    save_comparison(rows_by_mode, models)
    print_speed_summary(rows_by_mode)


if __name__ == "__main__":
    main()
