import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ----------------------------
# Global Configurations
# ----------------------------
MODEL_DIR = f"{Path(__file__).parent.parent}/model/"

# fmt: off
# synthetic primitives & sequences
SAMPLE_PER_CLASS = 64           # 训练集每类样本数
TEST_SAMPLE_PER_CLASS = 216     # 测试集每类样本数
T_MAX = 30                      # 随机噪声最大长度
PHONEME_LEN = 3                 # 每个音素持续时间
NOISE_STD = 0.01                # 噪声标准差
N_MFCC = 3                      # 输入维度
HIDDEN_DIM = 64                 # RNN隐藏层神经元数
TAU = 2                         # RNN时间常数
BATCH_SIZE = 16                 # 批大小
EPOCHS = 100                    # 训练轮数
LR = 1e-3                       # 学习率
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# fmt: on

# ----------------------------
# Synthetic Helper Functions
# ----------------------------
def generate_synthetic_sequence(seq, T_max, noise_std, phoneme_len=3, fixed=False):
    # 格式：T_MAX空窗 + seg1 + T噪声 + seg2 + T_MAX空窗
    seg1 = torch.tensor(seq[0], dtype=torch.float32).unsqueeze(0).repeat(phoneme_len, 1)
    seg2 = torch.tensor(seq[1], dtype=torch.float32).unsqueeze(0).repeat(phoneme_len, 1)

    T_noise = T_max
    if not fixed:
        T_noise = random.randint(0, T_max)

    total_len = T_max + phoneme_len + T_noise + phoneme_len + T_max
    x = torch.randn(total_len, 3) * noise_std
    x[T_max : T_max + phoneme_len, :] = seg1
    x[
        T_max + phoneme_len + T_noise : T_max + phoneme_len + T_noise + phoneme_len, :
    ] = seg2
    lengths = [phoneme_len, phoneme_len]  # 每个音素长度
    return x, lengths, T_noise


class SyntheticSequenceDataset(Dataset):
    def __init__(
        self,
        sequences,
        n_samples_per_class,
        T_max,
        noise_std,
        phoneme_len=3,
        fixed=False,
    ):
        self.samples = []
        self.labels = []
        self.sequences = sequences
        self.phoneme_len = phoneme_len
        for idx, seq in enumerate(sequences):
            for _ in range(n_samples_per_class):
                x, lengths, T_noise = generate_synthetic_sequence(
                    seq, T_max, noise_std, phoneme_len, fixed
                )
                self.samples.append(x)
                self.labels.append((idx, lengths, T_noise))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        class_id, lengths, T_noise = self.labels[idx]
        y = generate_ramping_target(lengths, class_id, len(self.sequences), T_noise)
        return x, y, class_id, T_noise


# ----------------------------
# Collate function for dynamic padding
# ----------------------------
def collate_fn(batch):
    xs, ys, class_ids, t_noises = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    xs_padded = torch.stack(
        [torch.nn.functional.pad(x, (0, 0, 0, max_len - x.shape[0])) for x in xs]
    )
    ys_padded = torch.stack(
        [torch.nn.functional.pad(y, (0, 0, 0, max_len - y.shape[0])) for y in ys]
    )
    mask = torch.zeros(len(xs), max_len, dtype=torch.bool)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    return (
        xs_padded,
        ys_padded,
        mask,
        torch.tensor(class_ids, dtype=torch.long),
        torch.tensor(t_noises, dtype=torch.long),
        lengths,
    )


def masked_mse_loss(out, target, mask):
    mask = mask.to(out.device).unsqueeze(-1)
    return ((out - target) ** 2 * mask).sum() / (mask.sum() * target.shape[-1])


def evaluate_coarse_to_fine(out, class_ids, t_noises, lengths, window=5):
    correct = 0
    total = 0
    for i in range(out.shape[0]):
        class_id = int(class_ids[i].item())
        t_noise = int(t_noises[i].item())
        true_len = int(lengths[i].item())

        first_end = T_MAX + PHONEME_LEN + t_noise
        first_start = max(T_MAX, first_end - window)
        second_start = max(first_end, true_len - window)
        second_end = true_len

        first_mean = out[i, first_start:first_end, :].mean(dim=0)
        branch_score_01 = first_mean[:2].max()
        branch_score_23 = first_mean[2:].max()
        pred_branch = 0 if branch_score_01 >= branch_score_23 else 1
        target_branch = 0 if class_id in [0, 1] else 1

        second_mean = out[i, second_start:second_end, :].mean(dim=0)
        pred_class = second_mean.argmax().item()

        correct += int(pred_branch == target_branch)
        correct += int(pred_class == class_id)
        total += 2
    return correct, total


# ----------------------------
# Model Definition
# ----------------------------
class CustomContinuousRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, tau=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.tau = tau
        w_rec_std = 0.2 / np.sqrt(hidden_dim)
        self.W_in = nn.Linear(input_dim, hidden_dim)
        self.W_rec = nn.Parameter(
            torch.normal(
                torch.zeros(hidden_dim, hidden_dim),
                w_rec_std * torch.ones(hidden_dim, hidden_dim),
            )
        )
        self.I_b = nn.Parameter(torch.zeros(hidden_dim))
        self.W_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_seq):
        B, T, _ = input_seq.shape
        device = input_seq.device
        x = torch.zeros(B, self.hidden_dim, device=device)
        r = torch.tanh(x)
        outputs = []
        for t in range(T):
            I_t = input_seq[:, t, :]
            dx = (
                -x + torch.matmul(r, self.W_rec.T) + self.W_in(I_t) + self.I_b
            ) / self.tau
            x = x + dx
            r = torch.tanh(x)
            y_t = self.W_out(r)
            outputs.append(y_t.unsqueeze(1))
        return torch.cat(outputs, dim=1)


# ----------------------------
# Ramping Target
# ----------------------------
def generate_ramping_target(lengths, class_id, num_classes, T_noise):
    O_init = 0.25
    O_1 = 0.5
    O_2 = 1.0
    O_sile = 0.0
    phoneme_len = lengths[0]
    total_len = T_MAX + phoneme_len + T_noise + phoneme_len + T_MAX

    target = torch.full((total_len, num_classes), O_init)

    # 第一项的时间区间
    idx1_start = T_MAX
    idx1_end = idx1_start + phoneme_len + T_noise
    # 第二项的时间区间
    idx2_start = idx1_end
    idx2_end = idx2_start + phoneme_len + T_MAX

    # 第一阶段：按 class_id 的上层组 激活两个 neuron
    if class_id in [0, 1]:
        target[idx1_start:idx1_end, 0] = O_1
        target[idx1_start:idx1_end, 1] = O_1
        target[idx1_start:idx1_end, 2:] = O_sile
    else:
        target[idx1_start:idx1_end, 2] = O_1
        target[idx1_start:idx1_end, 3] = O_1
        target[idx1_start:idx1_end, :2] = O_sile

    # 第二阶段：只留下目标 neuron ramp 到 O2，其余归零
    target[idx2_start:idx2_end, :] = O_sile
    target[idx2_start:idx2_end, class_id] = O_2

    return target


# ----------------------------
# Training
# ----------------------------
def train():
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
    model = CustomContinuousRNN(N_MFCC, HIDDEN_DIM, len(SEQUENCES_SYN), TAU).to(DEVICE)
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
    # ----------- 测试集评估 -----------
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
            batch_correct, batch_total = evaluate_coarse_to_fine(
                out.cpu(), class_ids, t_noises, lengths
            )
            correct += batch_correct
            total += batch_total
    print(f"Test set average loss: {test_loss / len(loader_test):.4f}")
    print(f"Test set accuracy: {correct / total * 100:.2f}%")
    # ----------- 保存模型 -----------
    os.makedirs(MODEL_DIR, exist_ok=True)
    current_time = time.strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(
        MODEL_DIR,
        f"rnn_mfcc{N_MFCC}_hid{HIDDEN_DIM}_spc{SAMPLE_PER_CLASS}_sec2_{current_time}.pt",
    )
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    train()
