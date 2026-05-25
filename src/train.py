import os
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ----------------------------
# 全局配置参数（统一管理）
# ----------------------------
# 数据路径与类别
TRAIN_DATA_DIR = f"{Path(__file__).parent.parent}/data/TIMIT/TRAIN/"
TEST_DATA_DIR = f"{Path(__file__).parent.parent}/data/TIMIT/TEST/"
MODEL_DIR = f"{Path(__file__).parent.parent}/model/"

PRIMITIVE_PHONEMES = ["pcl", "tcl", "pau"]
SEQUENCES = [("pcl", "tcl"), ("pcl", "pau"), ("tcl", "pcl"), ("tcl", "pau")]

# 数据增强与采样
SAMPLE_PER_CLASS = 4000  # 训练集每类样本数
TEST_SAMPLE_PER_CLASS = 1000  # 测试集每类样本数
AUGMENT_TIMES = 2  # 每个原始样本生成多少份增强样本
PHONEME_LEN = 3  # 每个音素片段MFCC帧数
T_MAX = 30  # 随机噪声最大长度
NOISE_STD = 0.01  # 噪声标准差

# RNN模型参数
N_MFCC = 3  # MFCC特征维度
HIDDEN_DIM = 50  # RNN隐藏层神经元数
TAU = 2  # RNN时间常数

# 训练参数
BATCH_SIZE = 16  # 批大小
EPOCHS = 100  # 训练轮数
LR = 10 ** random.uniform(-4, -2)  # 学习率（随机采样）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 设备


# ----------------------------
# Helper Functions
# ----------------------------
def find_phoneme_segments(data_root, target_phonemes):
    segments = {phn: [] for phn in target_phonemes}
    found_phn_files = 0
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.lower().endswith(".phn"):
                phn_path = os.path.join(root, file)
                wav_path = phn_path[:-4] + ".wav"
                if not os.path.exists(wav_path):
                    wav_path = phn_path[:-4] + ".WAV"  # 兼容大写
                    if not os.path.exists(wav_path):
                        print(f"WAV file not found for PHN: {phn_path}")
                        continue
                found_phn_files += 1
                with open(phn_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 3:
                            continue
                        start, end, label = parts
                        label = label.lower()
                        if label in target_phonemes:
                            segments[label].append((wav_path, int(start), int(end)))
    print(f"Scanned PHN files: {found_phn_files}")
    for phn in target_phonemes:
        print(f"Phoneme '{phn}': {len(segments[phn])} segments found.")
    return segments


def extract_mfcc_segment(wav_path, start, end, sample_rate=16000, n_mfcc=16):
    waveform, sr = torchaudio.load(wav_path)
    if sr != sample_rate:
        resample = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resample(waveform)
    segment = waveform[:, start:end]
    # 适配短片段
    n_fft = 32
    win_length = 32
    hop_length = 8
    # 使用全局 N_MFCC
    n_mfcc = N_MFCC
    if segment.shape[1] < n_fft:
        # 片段太短，返回空tensor
        return torch.zeros(0, n_mfcc)
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "win_length": win_length,
            "hop_length": hop_length,
            "center": False,
            "n_mels": n_mfcc,
        },
    )(segment)
    return mfcc.squeeze(0).T  # [T, n_mfcc]


def pad_or_truncate(x, target_len):
    if x.shape[0] < target_len:
        pad = torch.zeros(target_len - x.shape[0], x.shape[1], device=x.device)
        return torch.cat([x, pad], dim=0)
    else:
        return x[:target_len]


def augment_sequence(seg1, seg2, T_max, noise_std, phoneme_len=3):
    seg1 = pad_or_truncate(seg1, phoneme_len)
    seg2 = pad_or_truncate(seg2, phoneme_len)
    augmented = [seg1]
    delay_len = random.randint(0, T_max)
    if delay_len > 0:
        noise = torch.randn(delay_len, seg1.shape[1]) * noise_std
        augmented.append(noise)
    augmented.append(seg2)
    return torch.cat(augmented, dim=0)


def generate_ramping_target(lengths, class_id, num_classes):
    O_init = 0.25
    O_1 = 0.5
    O_2 = 1.0
    O_sile = 0.0

    total_len = sum(lengths)
    t1 = lengths[0]
    t2 = total_len
    target = torch.full((total_len, num_classes), O_init)  # 初始全部 Oinit

    # 当前序列对应的两个输出神经元
    seq = SEQUENCES[class_id]
    # neuron_a: 当前组合的索引
    neuron_a = class_id
    # neuron_b: 反向组合的索引
    try:
        neuron_b = SEQUENCES.index((seq[1], seq[0]))
    except ValueError:
        neuron_b = None

    # 第一阶段（第一项出现时）：两个相关 neuron ramp up 到 O_1，其他为 Osile
    target[:t1, :] = O_sile
    target[:t1, neuron_a] = O_1
    if neuron_b is not None:
        target[:t1, neuron_b] = O_1

    # 第二阶段（第二项出现时）：第一个 neuron ramp up 到 O_2，其他为 Osile
    target[t1:t2, :] = O_sile
    target[t1:t2, neuron_a] = O_2

    # 初始状态（可选）：target[:1, :] = O_init
    return target


# ----------------------------
# Dataset Class
# ----------------------------
class TIMITPhonemeSequenceDataset(Dataset):
    def __init__(self, segments, sequences, n_samples_per_class, augment_times=1):
        self.samples = []
        self.labels = []
        self.scaler = StandardScaler()

        for idx, (p1, p2) in enumerate(sequences):
            for _ in range(n_samples_per_class):
                seg1 = random.choice(segments[p1])
                seg2 = random.choice(segments[p2])

                mfcc1 = extract_mfcc_segment(*seg1, n_mfcc=N_MFCC)
                mfcc2 = extract_mfcc_segment(*seg2, n_mfcc=N_MFCC)

                if mfcc1.shape[0] < 5 or mfcc2.shape[0] < 5:
                    continue

                for _ in range(augment_times):
                    seq = augment_sequence(
                        mfcc1, mfcc2, T_MAX, NOISE_STD, phoneme_len=PHONEME_LEN
                    )
                    self.samples.append(seq)
                    self.labels.append((idx, [mfcc1.shape[0], mfcc2.shape[0]]))

        # Normalize features
        all_data = torch.cat(self.samples, dim=0).numpy()
        self.scaler.fit(all_data)
        self.samples = [
            torch.tensor(self.scaler.transform(x.numpy()), dtype=torch.float32)
            for x in self.samples
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        class_id, lengths = self.labels[idx]
        y = generate_ramping_target(lengths, class_id, len(SEQUENCES))
        return x, y


# Collate function for dynamic padding
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

        # 参数初始化
        w_rec_std = 0.2 / np.sqrt(hidden_dim)
        self.W_in = nn.Linear(input_dim, hidden_dim)  # 输入层 Linear
        self.W_rec = nn.Parameter(
            torch.normal(
                torch.zeros(hidden_dim, hidden_dim),
                w_rec_std * torch.ones(hidden_dim, hidden_dim),
            )
        )
        self.I_b = nn.Parameter(torch.zeros(hidden_dim))
        self.W_out = nn.Linear(hidden_dim, output_dim)  # 输出层 N->K

    def forward(self, input_seq):
        B, T, _ = input_seq.shape
        device = input_seq.device

        x = torch.zeros(B, self.hidden_dim, device=device)
        r = torch.tanh(x)
        outputs = []

        for t in range(T):
            I_t = input_seq[:, t, :]  # [B, N]

            dx = (
                -x
                + torch.matmul(r, self.W_rec.T)  # recurrent 输入
                + self.W_in(I_t)  # 输入层 Linear
                + self.I_b
            ) / self.tau  # 时间常数离散化

            x = x + dx
            r = torch.tanh(x)

            y_t = self.W_out(r)
            outputs.append(y_t.unsqueeze(1))
        return torch.cat(outputs, dim=1)  # [B, T, K]


# ----------------------------
# Training
# ----------------------------
def train():
    # 训练集
    segments_train = find_phoneme_segments(TRAIN_DATA_DIR, PRIMITIVE_PHONEMES)
    dataset_train = TIMITPhonemeSequenceDataset(
        segments_train, SEQUENCES, SAMPLE_PER_CLASS, augment_times=AUGMENT_TIMES
    )
    loader_train = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    # 测试集
    DATA_ROOT_TEST = f"{Path(__file__).parent.parent}/data/TIMIT/TEST/"
    segments_test = find_phoneme_segments(DATA_ROOT_TEST, PRIMITIVE_PHONEMES)
    dataset_test = TIMITPhonemeSequenceDataset(
        segments_test, SEQUENCES, TEST_SAMPLE_PER_CLASS, augment_times=AUGMENT_TIMES
    )
    loader_test = DataLoader(
        dataset_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    model = CustomContinuousRNN(N_MFCC, HIDDEN_DIM, len(SEQUENCES), TAU).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y in tqdm(
            loader_train,
            desc=f"Epoch {epoch+1}/{EPOCHS}",
            dynamic_ncols=True,
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
            # 分类准确率评估
            pred = out[:, -1, :].argmax(dim=1)
            target = y[:, -1, :].argmax(dim=1)
            correct += (pred == target).sum().item()
            total += x.size(0)
    print(f"Test set average loss: {test_loss / len(loader_test):.4f}")
    print(f"Test set accuracy: {correct / total * 100:.2f}%")

    # ----------- 保存模型 -----------
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(
        MODEL_DIR, f"rnn_mfcc{N_MFCC}_hid{HIDDEN_DIM}_tau{TAU}_ep{EPOCHS}_lr{LR:.4g}.pt"
    )
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    train()
