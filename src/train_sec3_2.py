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
# 全局配置参数（统一管理）
# ----------------------------
MODEL_DIR = f"{Path(__file__).parent.parent}/model/"

# synthetic primitives & sequences
SAMPLE_PER_CLASS = 256  # 训练集每类样本数
TEST_SAMPLE_PER_CLASS = 64  # 测试集每类样本数
T_MAX = 30  # 随机噪声最大长度
PHONEME_LEN = 3  # 每个音素持续时间
NOISE_STD = 0.01  # 噪声标准差
N_MFCC = 3  # 输入维度
HIDDEN_DIM = 50  # RNN隐藏层神经元数
TAU = 2  # RNN时间常数
BATCH_SIZE = 16  # 批大小
EPOCHS = 100  # 训练轮数
LR = 10 ** random.uniform(-4, -2)  # 学习率（随机采样）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 设备


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


def generate_synthetic_sequence_3item(seq, T_max, noise_std, phoneme_len=3, fixed=False):
    # 格式：T_MAX空窗 + seg1 + T噪声1 + seg2 + T噪声2 + seg3 + T_MAX空窗
    seg1 = torch.tensor(seq[0], dtype=torch.float32).unsqueeze(0).repeat(phoneme_len, 1)
    seg2 = torch.tensor(seq[1], dtype=torch.float32).unsqueeze(0).repeat(phoneme_len, 1)
    seg3 = torch.tensor(seq[2], dtype=torch.float32).unsqueeze(0).repeat(phoneme_len, 1)
    T_noise1 = random.randint(0, T_max) if not fixed else T_max
    T_noise2 = random.randint(0, T_max) if not fixed else T_max
    total_len = T_MAX + phoneme_len + T_noise1 + phoneme_len + T_noise2 + phoneme_len + T_MAX
    x = torch.randn(total_len, N_MFCC) * noise_std
    x[T_MAX : T_MAX + phoneme_len, :] = seg1
    x[T_MAX + phoneme_len + T_noise1 : T_MAX + phoneme_len + T_noise1 + phoneme_len, :] = seg2
    x[T_MAX + phoneme_len + T_noise1 + phoneme_len + T_noise2 : T_MAX + phoneme_len + T_noise1 + phoneme_len + T_noise2 + phoneme_len, :] = seg3
    lengths = [phoneme_len, phoneme_len, phoneme_len]
    return x, lengths, (T_noise1, T_noise2)


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
        return x, y


class SyntheticSequence3ItemDataset(Dataset):
    def __init__(self, sequences, n_samples_per_class, T_max, noise_std, phoneme_len=3, fixed=False):
        self.samples = []
        self.labels = []
        self.sequences = sequences
        self.phoneme_len = phoneme_len
        for idx, seq in enumerate(sequences):
            for _ in range(n_samples_per_class):
                x, lengths, T_noises = generate_synthetic_sequence_3item(seq, T_max, noise_std, phoneme_len, fixed)
                self.samples.append(x)
                self.labels.append((idx, lengths, T_noises))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        x = self.samples[idx]
        class_id, lengths, T_noises = self.labels[idx]
        y = generate_ramping_target_3item(lengths, class_id, len(self.sequences), T_noises)
        return x, y


# ----------------------------
# Collate function for dynamic padding
# ----------------------------
def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    xs_padded = torch.stack(
        [torch.nn.functional.pad(x, (0, 0, 0, max_len - x.shape[0])) for x in xs]
    )
    ys_padded = torch.stack(
        [torch.nn.functional.pad(y, (0, 0, 0, max_len - y.shape[0])) for y in ys]
    )
    return xs_padded, ys_padded


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


def generate_ramping_target_3item(lengths, class_id, num_classes, T_noises):
    O_init = 0.25
    O_1 = 0.5
    O_2 = 0.75
    O_3 = 1.0
    O_sile = 0.0
    phoneme_len = lengths[0]
    T_noise1, T_noise2 = T_noises
    total_len = T_MAX + phoneme_len + T_noise1 + phoneme_len + T_noise2 + phoneme_len + T_MAX
    target = torch.full((total_len, num_classes), O_init)
    # 三阶段分段
    idx1_start = T_MAX
    idx1_end = idx1_start + phoneme_len + T_noise1
    idx2_start = idx1_end
    idx2_end = idx2_start + phoneme_len + T_noise2
    idx3_start = idx2_end
    idx3_end = idx3_start + phoneme_len + T_MAX
    # 第一阶段
    target[idx1_start:idx1_end, :] = O_init
    target[idx1_start:idx1_end, class_id] = O_1
    # 第二阶段
    target[idx2_start:idx2_end, :] = O_sile
    target[idx2_start:idx2_end, class_id] = O_2
    # 第三阶段
    target[idx3_start:idx3_end, :] = O_sile
    target[idx3_start:idx3_end, class_id] = O_3
    return target


# ----------------------------
# Training
# ----------------------------
def train():
    # synthetic primitives & sequences
    a = np.array([1, 1, 0], dtype=np.float32)
    b = np.array([1, 0, 1], dtype=np.float32)
    c = np.array([0, 1, 1], dtype=np.float32)
    SEQUENCES_SYN_3 = [
        (a, b, a),
        (a, b, c),
        (a, c, a),
        (a, c, b),
        (b, a, b),
        (b, a, c),
        (b, c, a),
        (b, c, b),
    ]
    dataset_train = SyntheticSequence3ItemDataset(
        SEQUENCES_SYN_3, SAMPLE_PER_CLASS, T_MAX, NOISE_STD, phoneme_len=PHONEME_LEN
    )
    loader_train = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    dataset_test = SyntheticSequence3ItemDataset(
        SEQUENCES_SYN_3,
        TEST_SAMPLE_PER_CLASS,
        T_MAX,
        NOISE_STD,
        phoneme_len=PHONEME_LEN,
        fixed=True,
    )
    loader_test = DataLoader(
        dataset_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )
    model = CustomContinuousRNN(N_MFCC, HIDDEN_DIM, len(SEQUENCES_SYN_3), TAU).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    loss_fn = nn.MSELoss()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y in tqdm(
            loader_train,
            desc=f"Epoch {epoch+1}/{EPOCHS}",
            dynamic_ncols=False,
            leave=True,
        ):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(
            f"Epoch {epoch+1} loss: {total_loss / len(loader_train):.4f}, lr: {scheduler.get_last_lr()[0]:.6f}"
        )
    # ----------- 测试集评估 -----------
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader_test:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = loss_fn(out, y)
            test_loss += loss.item()
            # 评估准确率：整体序列平均
            pred = out.mean(dim=1).argmax(dim=1)
            target = y.mean(dim=1).argmax(dim=1)
            correct += (pred == target).sum().item()
            total += x.size(0)
    print(f"Test set average loss: {test_loss / len(loader_test):.4f}")
    print(f"Test set accuracy: {correct / total * 100:.2f}%")
    # ----------- 保存模型 -----------
    os.makedirs(MODEL_DIR, exist_ok=True)
    current_time = time.strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(
        MODEL_DIR,
        f"rnn_mfcc{N_MFCC}_hid{HIDDEN_DIM}_spc{SAMPLE_PER_CLASS}_sec3_2_{current_time}.pt",
    )
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    train()
