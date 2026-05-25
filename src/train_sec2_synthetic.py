import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import SyntheticSequenceDataset, collate_fn
from models import SimpleRNN

# ----------------------------
# Global Configurations
# ----------------------------
MODEL_DIR = f"{Path(__file__).parent.parent}/model/"

# fmt: off
# synthetic primitives & sequences
SAMPLE_PER_CLASS = 128          # 训练集每类样本数
TEST_SAMPLE_PER_CLASS = 256     # 测试集每类样本数
T_MAX = 30                      # 随机噪声最大长度
PHONEME_LEN = 3                 # 每个音素持续时间
NOISE_STD = 0.01                # 噪声标准差
N_MFCC = 3                      # 输入维度
HIDDEN_DIM = 64                 # RNN隐藏层神经元数
TAU = 2                         # RNN时间常数
BATCH_SIZE = 16                 # 批大小
EPOCHS = 120                    # 训练轮数
LR = 3e-4                       # 学习率
SEED = 1107                     # 随机种子
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# fmt: on


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


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------
# Training
# ----------------------------
def train():
    set_seed(SEED)
    # synthetic primitives & sequences
    a = np.array([1, 1, 0], dtype=np.float32)
    b = np.array([1, 0, 1], dtype=np.float32)
    c = np.array([0, 1, 1], dtype=np.float32)
    SEQUENCES_SYN = [(a, b), (a, c), (b, a), (b, c)]
    dataset_train = SyntheticSequenceDataset(
        SEQUENCES_SYN, SAMPLE_PER_CLASS, T_MAX, NOISE_STD, phoneme_len=PHONEME_LEN
    )
    loader_train = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    dataset_test = SyntheticSequenceDataset(
        SEQUENCES_SYN,
        TEST_SAMPLE_PER_CLASS,
        T_MAX,
        NOISE_STD,
        phoneme_len=PHONEME_LEN,
        fixed=True,
    )
    loader_test = DataLoader(
        dataset_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )
    model = SimpleRNN(N_MFCC, HIDDEN_DIM, len(SEQUENCES_SYN), TAU).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y, mask, _, _, _ in tqdm(
            loader_train,
            desc=f"Epoch {epoch+1}/{EPOCHS}",
            dynamic_ncols=False,
            leave=True,
        ):
            x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
            out = model(x)
            loss = masked_mse_loss(out, y, mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print(
            f"Epoch {epoch+1} loss: {total_loss / len(loader_train):.4f}, lr: {LR:.6f}"
        )
    # ----------- Evaluation -----------
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, mask, class_ids, t_noises, lengths in loader_test:
            x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
            out = model(x)
            loss = masked_mse_loss(out, y, mask)
            test_loss += loss.item()
            batch_correct, batch_total = evaluate_final_readout(
                out.cpu(), class_ids, lengths
            )
            correct += batch_correct
            total += batch_total
    print(f"Test set average loss: {test_loss / len(loader_test):.4f}")
    print(f"Test set accuracy: {correct / total * 100:.2f}%")
    # ----------- Saving Model -----------
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(
        MODEL_DIR,
        f"rnn_mfcc{N_MFCC}_hid{HIDDEN_DIM}_spc{SAMPLE_PER_CLASS}_sec2_seed{SEED}.pt",
    )
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    train()
