import glob
import os
import random
import time
from pathlib import Path

# TIMIT音素特征提取
import librosa
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ----------------------------
# 全局配置参数（统一管理）
# ----------------------------
TIMIT_DIR = (
    f"{Path(__file__).parent.parent}/data/TIMIT/TRAIN/"  # 请修改为你的TIMIT数据集路径
)
MODEL_DIR = f"{Path(__file__).parent.parent}/model/"
CSV_DIR = f"{Path(__file__).parent.parent}/csv/"
MODEL_NAME = "rnn_mfcc3_hid50_spc256_sec3_2_20250715_154931.pt"
PRETRAINED_MODEL_DIR = f"{MODEL_DIR}/{MODEL_NAME}"
USE_PRETRAINED = True  # 是否使用预训练模型
FREEZE = False  # 是否冻结RNN层参数，仅训练全连接层

# synthetic primitives & sequences
SAMPLE_PER_CLASS = 512  # 训练集每类样本数
TEST_SAMPLE_PER_CLASS = 1024  # 测试集每类样本数
T_MAX = 30  # 随机噪声最大长度
PHONEME_LEN = 3  # 每个音素持续时间
NOISE_STD = 0.01  # 噪声标准差
N_MFCC = 16  # 输入维度
HIDDEN_DIM = 50  # RNN隐藏层神经元数
TAU = 2  # RNN时间常数
BATCH_SIZE = 16  # 批大小
EPOCHS = 100  # 训练轮数
LR = 10 ** random.uniform(-4, -2)  # 学习率（随机采样）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 设备


# TIMIT音素MFCC均值提取
def extract_phoneme_mfcc_mean(timit_root, phoneme_list, n_mfcc=3):
    """
    从TIMIT数据集中提取指定音素的MFCC均值特征
    返回: {phoneme: mfcc_mean}
    """
    phoneme_mfccs = {p: [] for p in phoneme_list}
    wav_files = glob.glob(os.path.join(timit_root, "**/*.WAV"), recursive=True)
    for wav_path in wav_files:
        phn_path = wav_path.replace(".WAV", ".PHN")
        if not os.path.exists(phn_path):
            continue
        y, sr = librosa.load(wav_path, sr=None)
        with open(phn_path, "r") as f:
            for line in f:
                start, end, phn = line.strip().split()
                if phn in phoneme_list:
                    start, end = int(start), int(end)
                    seg = y[start:end]
                    if len(seg) < 10:
                        continue
                    mfcc = librosa.feature.mfcc(
                        y=seg,
                        sr=sr,
                        n_mfcc=n_mfcc,
                        n_fft=min(2048, len(seg)),
                        n_mels=20,
                    )
                    mfcc_mean = mfcc.mean(axis=1)
                    phoneme_mfccs[phn].append(mfcc_mean)
    # 取每个音素所有片段的均值
    phoneme_means = {}
    for p in phoneme_list:
        arr = np.stack(phoneme_mfccs[p], axis=0)
        phoneme_means[p] = arr.mean(axis=0)
    return phoneme_means


# ----------------------------
# Synthetic Helper Functions
# ----------------------------
def generate_synthetic_sequence(seq, T_max, noise_std, phoneme_len=3, fixed=False):
    # 格式：T_MAX空窗 + seg1 + T噪声 + seg2 + T_MAX空窗
    seg1 = torch.tensor(seq[0], dtype=torch.float32).unsqueeze(0).repeat(phoneme_len, 1)
    seg2 = torch.tensor(seq[1], dtype=torch.float32).unsqueeze(0).repeat(phoneme_len, 1)

    T_noise = 0
    if fixed:
        T_noise = T_max
    else:
        T_noise = random.randint(0, T_max)

    total_len = T_max + phoneme_len + T_noise + phoneme_len + T_max
    x = torch.randn(total_len, N_MFCC) * noise_std
    x[T_max : T_max + phoneme_len, :] = seg1
    x[
        T_max + phoneme_len + T_noise : T_max + phoneme_len + T_noise + phoneme_len, :
    ] = seg2
    lengths = [phoneme_len, phoneme_len]  # 每个音素长度
    return x, lengths, T_noise


# 三项音素序列生成（支持2项和3项）
def generate_synthetic_sequence_3item(seq, T_max, noise_std, phoneme_len=3, fixed=False):
    # seq: (a, b, c) 或 (a, b)
    segs = [torch.tensor(s, dtype=torch.float32).unsqueeze(0).repeat(phoneme_len, 1) for s in seq]
    T_noise = T_max if fixed else random.randint(0, T_max)
    if len(segs) == 3:
        total_len = T_max + phoneme_len + T_noise + phoneme_len + T_noise + phoneme_len + T_MAX
        x = torch.randn(total_len, N_MFCC) * noise_std
        x[T_MAX : T_MAX + phoneme_len, :] = segs[0]
        x[T_MAX + phoneme_len + T_noise : T_MAX + phoneme_len + T_noise + phoneme_len, :] = segs[1]
        x[T_MAX + phoneme_len + T_noise + phoneme_len + T_noise : T_MAX + phoneme_len + T_noise + phoneme_len + T_noise + phoneme_len, :] = segs[2]
        lengths = [phoneme_len, phoneme_len, phoneme_len]
    elif len(segs) == 2:
        total_len = T_MAX + phoneme_len + T_noise + phoneme_len + T_MAX
        x = torch.randn(total_len, N_MFCC) * noise_std
        x[T_MAX : T_MAX + phoneme_len, :] = segs[0]
        x[T_MAX + phoneme_len + T_noise : T_MAX + phoneme_len + T_noise + phoneme_len, :] = segs[1]
        lengths = [phoneme_len, phoneme_len]
    else:
        raise ValueError("seq must be length 2 or 3")
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
        return x, y


class SyntheticSequenceDataset3Item(Dataset):
    def __init__(self, sequences, n_samples_per_class, T_max, noise_std, phoneme_len=3, fixed=False):
        self.samples = []
        self.labels = []
        self.sequences = sequences
        self.phoneme_len = phoneme_len
        for idx, seq in enumerate(sequences):
            for _ in range(n_samples_per_class):
                x, lengths, T_noise = generate_synthetic_sequence_3item(seq, T_max, noise_std, phoneme_len, fixed)
                self.samples.append(x)
                self.labels.append((idx, lengths, T_noise))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        x = self.samples[idx]
        class_id, lengths, T_noise = self.labels[idx]
        y = generate_ramping_target_3item(lengths, class_id, len(self.sequences), T_noise)
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

# ramping target函数（支持三项序列）
def generate_ramping_target_3item(lengths, class_id, num_classes, T_noise):
    O_init = 0.25
    O_1 = 0.5
    O_2 = 0.75
    O_3 = 1.0
    O_sile = 0.0
    if len(lengths) == 3:
        phoneme_len = lengths[0]
        T_noise = T_noise
        total_len = T_MAX + phoneme_len + T_noise + phoneme_len + T_noise + phoneme_len + T_MAX
        target = torch.full((total_len, num_classes), O_init)
        # 第一项区间
        idx1_start = T_MAX
        idx1_end = idx1_start + phoneme_len + T_noise
        # 第二项区间
        idx2_start = idx1_end
        idx2_end = idx2_start + phoneme_len + T_noise
        # 第三项区间
        idx3_start = idx2_end
        idx3_end = idx3_start + phoneme_len + T_MAX
        # 第一阶段
        target[idx1_start:idx1_end, :] = O_sile
        target[idx1_start:idx1_end, 0] = O_1
        # 第二阶段
        target[idx2_start:idx2_end, :] = O_sile
        target[idx2_start:idx2_end, 1] = O_2
        # 第三阶段
        target[idx3_start:idx3_end, :] = O_sile
        target[idx3_start:idx3_end, 2] = O_3
    elif len(lengths) == 2:
        phoneme_len = lengths[0]
        total_len = T_MAX + phoneme_len + T_noise + phoneme_len + T_MAX
        target = torch.full((total_len, num_classes), O_init)
        idx1_start = T_MAX
        idx1_end = idx1_start + phoneme_len + T_noise
        idx2_start = idx1_end
        idx2_end = idx2_start + phoneme_len + T_MAX
        target[idx1_start:idx1_end, :] = O_sile
        target[idx1_start:idx1_end, 0] = O_1
        target[idx2_start:idx2_end, :] = O_sile
        target[idx2_start:idx2_end, 1] = O_2
    else:
        raise ValueError("lengths must be 2 or 3")
    return target


# ----------------------------
# Training
# ----------------------------
def train():
    # TIMIT音素序列构造
    phoneme_list = ["pcl", "tcl", "pau"]
    print("提取TIMIT音素MFCC均值...")
    phoneme_means = extract_phoneme_mfcc_mean(TIMIT_DIR, phoneme_list, n_mfcc=N_MFCC)
    a = phoneme_means["pcl"]
    b = phoneme_means["tcl"]
    c = phoneme_means["pau"]
    SEQUENCES_TIMIT = [
        (a, b, a),      # pcl-tcl-pcl
        (a, b, c),      # pcl-tcl-pau
        (a, c, a),      # pcl-pau-pcl
        (a, c, b),      # pcl-pau-tcl
        (b, c),         # tcl-pau
        (b, a),         # tcl-pcl
    ]
    dataset_train = SyntheticSequenceDataset3Item(SEQUENCES_TIMIT, SAMPLE_PER_CLASS, T_MAX, NOISE_STD, phoneme_len=PHONEME_LEN)
    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dataset_test = SyntheticSequenceDataset3Item(SEQUENCES_TIMIT, TEST_SAMPLE_PER_CLASS, T_MAX, NOISE_STD, phoneme_len=PHONEME_LEN)
    loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    # ----------- Transfer Learning -----------
    model = CustomContinuousRNN(N_MFCC, HIDDEN_DIM, len(SEQUENCES_TIMIT), TAU).to(DEVICE)
    if USE_PRETRAINED:
        print(f"加载预训练模型: {PRETRAINED_MODEL_DIR}")
        pretrained = torch.load(PRETRAINED_MODEL_DIR, map_location=DEVICE)
        with torch.no_grad():
            if "W_rec" in pretrained:
                model.W_rec.copy_(pretrained["W_rec"])
            if "I_b" in pretrained:
                model.I_b.copy_(pretrained["I_b"])
        print("已将RNN层权重与预训练模型保持一致，其余层按当前输入维度初始化。")
    else:
        print("未使用预训练模型，RNN参数将随机初始化。")
    if FREEZE:
        print("冻结RNN层参数，仅训练W_in和W_out。")
        for name, param in model.named_parameters():
            if name not in ["W_in.weight", "W_in.bias", "W_out.weight", "W_out.bias"]:
                param.requires_grad = False
    else:
        print("所有参数均可训练。")
        for param in model.parameters():
            param.requires_grad = True
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    loss_fn = nn.MSELoss()
    import csv
    history = []
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y in tqdm(loader_train, desc=f"Epoch {epoch+1}/{EPOCHS}", dynamic_ncols=False, leave=True):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        train_loss = total_loss / len(loader_train)
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
                batch_size = x.size(0)
                for i in range(batch_size):
                    target_seq = y[i]
                    out_seq = out[i]
                    if target_seq.shape[0] > 2 * T_MAX + 3 * PHONEME_LEN:
                        # 三项序列
                        phoneme_len = PHONEME_LEN
                        T_noise = (target_seq.shape[0] - (T_MAX + 3 * phoneme_len + T_MAX)) // 2
                        # 第一项区间
                        idx1_start = T_MAX
                        idx1_end = idx1_start + phoneme_len + T_noise
                        pred1 = out_seq[idx1_start:idx1_end].mean(dim=0).argmax().item()
                        target1 = target_seq[idx1_start:idx1_end].mean(dim=0).argmax().item()
                        correct += int(pred1 == target1)
                        total += 1
                        # 第二项区间
                        idx2_start = idx1_end
                        idx2_end = idx2_start + phoneme_len + T_noise
                        pred2 = out_seq[idx2_start:idx2_end].mean(dim=0).argmax().item()
                        target2 = target_seq[idx2_start:idx2_end].mean(dim=0).argmax().item()
                        correct += int(pred2 == target2)
                        total += 1
                        # 第三项区间
                        idx3_start = idx2_end
                        idx3_end = idx3_start + phoneme_len + T_MAX
                        pred3 = out_seq[idx3_start:idx3_end].mean(dim=0).argmax().item()
                        target3 = target_seq[idx3_start:idx3_end].mean(dim=0).argmax().item()
                        correct += int(pred3 == target3)
                        total += 1
                    else:
                        # 两项序列
                        phoneme_len = PHONEME_LEN
                        T_noise = target_seq.shape[0] - (T_MAX + 2 * phoneme_len + T_MAX)
                        # 第一项区间
                        idx1_start = T_MAX
                        idx1_end = idx1_start + phoneme_len + T_noise
                        pred1 = out_seq[idx1_start:idx1_end].mean(dim=0).argmax().item()
                        target1 = target_seq[idx1_start:idx1_end].mean(dim=0).argmax().item()
                        correct += int(pred1 == target1)
                        total += 1
                        # 第二项区间
                        idx2_start = idx1_end
                        idx2_end = idx2_start + phoneme_len + T_MAX
                        pred2 = out_seq[idx2_start:idx2_end].mean(dim=0).argmax().item()
                        target2 = target_seq[idx2_start:idx2_end].mean(dim=0).argmax().item()
                        correct += int(pred2 == target2)
                        total += 1
        test_loss_avg = test_loss / len(loader_test)
        test_acc = correct / total * 100
        print(f"Epoch {epoch+1} loss: {train_loss:.4f}, test_loss: {test_loss_avg:.4f}, test_acc: {test_acc:.2f}%, lr: {scheduler.get_last_lr()[0]:.6f}")
        history.append([epoch + 1, train_loss, test_loss_avg, test_acc, scheduler.get_last_lr()[0]])
    csv_path = os.path.join(CSV_DIR, f"train_history_newtest_rnn_mfcc{N_MFCC}_hid{HIDDEN_DIM}_spc{SAMPLE_PER_CLASS}_pretrained{USE_PRETRAINED}_freeze{FREEZE}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_loss", "test_acc", "lr"])
        writer.writerows(history)
    print(f"训练过程已保存到: {csv_path}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    current_time = time.strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f"newtest_rnn_mfcc{N_MFCC}_hid{HIDDEN_DIM}_spc{SAMPLE_PER_CLASS}_pretrained{USE_PRETRAINED}_freeze{FREEZE}_sec3_3_{current_time}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    train()
