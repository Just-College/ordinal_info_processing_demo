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
TIMIT_DIR = f"{Path(__file__).parent.parent}/data/TIMIT/TRAIN/"
MODEL_DIR = f"{Path(__file__).parent.parent}/model/"

# synthetic primitives & sequences
SAMPLE_PER_CLASS = 512  # 训练集每类样本数
TEST_SAMPLE_PER_CLASS = 256  # 测试集每类样本数
T_MAX = 60  # 随机噪声最大长度
PHONEME_LEN = 3  # 每个音素持续时间
NOISE_STD = 0.01  # 噪声标准差
N_MFCC = 16  # 输入维度
HIDDEN_DIM = 128  # RNN隐藏层神经元数
TAU = 2  # RNN时间常数
BATCH_SIZE = 16  # 批大小
EPOCHS = 200  # 训练轮数
LR = 10 ** random.uniform(-4, -2)  # 学习率（随机采样）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 设备


# 关键词分割与MFCC提取（按音节分割）
def extract_keyword_mfcc_by_syllable(
    timit_root, keywords, n_mfcc=16, samples_per_word=10
):
    """
    从TIMIT中筛选关键词并按音节分割，提取每个音节的MFCC均值
    返回: {word: [[chunk1_mfcc, chunk2_mfcc], ...]}
    """
    syllable_map = {
        "wash": [["w", "ao"], ["sh"]],
        "water": [["w", "ao"], ["t", "er"]],
        "year": [["y", "ih"], ["r"]],
        "had": [["h"], ["ae", "d"]],
    }
    result = {w: [] for w in keywords}
    wav_files = glob.glob(os.path.join(timit_root, "**/*.WAV"), recursive=True)
    sr = 16000
    n_fft = int(0.025 * sr)  # 25ms窗口
    hop_length = int(0.01 * sr)  # 10ms步长
    for word in keywords:
        candidate_files = []
        for wav_path in wav_files:
            wrd_path = wav_path.replace(".WAV", ".WRD")
            txt_path = wav_path.replace(".WAV", ".TXT")
            found = False
            for meta_path in [wrd_path, txt_path]:
                if os.path.exists(meta_path):
                    with open(meta_path, "r") as f:
                        content = f.read().lower()
                        if word in content:
                            found = True
                            break
            if found:
                candidate_files.append(wav_path)
        if len(candidate_files) < samples_per_word:
            print(f"警告：{word} 仅找到 {len(candidate_files)} 个样本")
            continue
        selected_files = random.sample(candidate_files, samples_per_word)
        for wav_path in selected_files:
            phn_path = wav_path.replace(".WAV", ".PHN")
            if not os.path.exists(phn_path):
                continue
            y, sr = librosa.load(wav_path, sr=sr)
            phn_intervals = []
            with open(phn_path, "r") as f:
                for line in f:
                    start, end, phn = line.strip().split()
                    phn_intervals.append((int(start), int(end), phn))
            chunks = []
            for syllable in syllable_map[word]:
                segs = []
                for s in syllable:
                    for start, end, phn in phn_intervals:
                        if phn == s:
                            segs.append(y[start:end])
                if segs:
                    seg = np.concatenate(segs)
                    # librosa警告修正：n_fft不能大于seg长度
                    n_fft_eff = min(n_fft, len(seg))
                    if n_fft_eff < 32:
                        continue  # 跳过过短片段
                    mfcc = librosa.feature.mfcc(
                        y=seg,
                        sr=sr,
                        n_mfcc=n_mfcc,
                        n_fft=n_fft_eff,
                        hop_length=hop_length,
                        n_mels=20,
                    )
                    # 沿时间维度归一化
                    mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (
                        mfcc.std(axis=1, keepdims=True) + 1e-8
                    )
                    mfcc_mean = mfcc.mean(axis=1)
                    chunks.append(mfcc_mean)
            if len(chunks) == 2:
                result[word].append(chunks)
    return result


# warp函数（np.interp实现，F范围0.5~2）
def warp_mfcc_sequence(mfcc_seq, F):
    """
    对MFCC序列进行时间轴warp，F为warp系数（0.5~2）
    mfcc_seq: shape (T, D)
    返回warp后的序列 shape (T_new, D)
    """
    T, D = mfcc_seq.shape
    T_new = max(1, int(T * F))
    x_old = np.arange(T)
    x_new = np.linspace(0, T - 1, T_new)
    warped = np.zeros((T_new, D))
    for d in range(D):
        warped[:, d] = np.interp(x_new, x_old, mfcc_seq[:, d])
    return warped


# ----------------------------
# Helper Functions
# ----------------------------
# 关键词任务专用序列生成（训练集插入间隔，测试集warp）
def generate_keyword_sequence(
    seq, T_max, noise_std, phoneme_len=3, T_noise=None, F=1.0, mode="train"
):
    # seq: [chunk1_mfcc, chunk2_mfcc]，每个为(n_mfcc,)
    # mode: "train"插入间隔，"test"进行warp
    if mode == "train":
        seg1 = (
            torch.tensor(seq[0], dtype=torch.float32)
            .unsqueeze(0)
            .repeat(phoneme_len, 1)
        )
        seg2 = (
            torch.tensor(seq[1], dtype=torch.float32)
            .unsqueeze(0)
            .repeat(phoneme_len, 1)
        )
        if T_noise is None:
            T_noise = random.randint(0, T_max)
        total_len = T_max + phoneme_len + T_noise + phoneme_len + T_max
        x = torch.randn(total_len, N_MFCC) * noise_std
        x[T_max : T_max + phoneme_len, :] = seg1
        x[
            T_max + phoneme_len + T_noise : T_max + phoneme_len + T_noise + phoneme_len,
            :,
        ] = seg2
        lengths = [phoneme_len, phoneme_len]
        return x, lengths, T_noise
    elif mode == "test":
        seg1 = (
            torch.tensor(seq[0], dtype=torch.float32)
            .unsqueeze(0)
            .repeat(phoneme_len, 1)
        )
        seg2 = (
            torch.tensor(seq[1], dtype=torch.float32)
            .unsqueeze(0)
            .repeat(phoneme_len, 1)
        )
        x = torch.cat([seg1, seg2], dim=0).numpy()
        x_warp = warp_mfcc_sequence(x, F)
        x_warp = torch.tensor(x_warp, dtype=torch.float32)
        lengths = [phoneme_len, phoneme_len]
        return x_warp, lengths, F


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
def generate_ramping_target(lengths, keyword_id, num_keywords, T_noise):
    O_init = 0.25
    O_1 = 1.0
    O_2 = 2.0
    O_sile = 0.0
    phoneme_len = lengths[0]
    total_len = T_MAX + phoneme_len + T_noise + phoneme_len + T_MAX
    target = torch.full((total_len, num_keywords), O_init)
    idx1_start = T_MAX
    idx1_end = idx1_start + phoneme_len + T_noise
    idx2_start = idx1_end
    idx2_end = idx2_start + phoneme_len + T_MAX
    if keyword_id in [0, 1]:
        target[idx1_start:idx1_end, 0] = O_1
        target[idx1_start:idx1_end, 1] = O_1
    else:
        target[idx1_start:idx1_end, 2] = O_1
        target[idx1_start:idx1_end, 3] = O_1
    target[idx2_start:idx2_end, :] = O_sile
    target[idx2_start:idx2_end, keyword_id] = O_2
    return target


class KeywordDataset(Dataset):
    def __init__(self, sequences, keyword_ids, n_samples, mode="train"):
        self.samples = []
        self.labels = []
        for idx, seq in enumerate(sequences):
            kid = keyword_ids[idx]
            for _ in range(n_samples):
                if mode == "train":
                    x, lengths, T_noise = generate_keyword_sequence(
                        seq, T_MAX, NOISE_STD, PHONEME_LEN, mode="train"
                    )
                    y = generate_ramping_target(lengths, kid, 4, T_noise)
                else:
                    # 测试集固定噪声区间
                    x, lengths, T_noise = generate_keyword_sequence(
                        seq, T_MAX, NOISE_STD, PHONEME_LEN, T_noise=T_MAX, mode="train"
                    )
                    y = generate_ramping_target(lengths, kid, 4, T_noise)
                self.samples.append(x)
                self.labels.append(y)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


# ----------------------------
# Training
# ----------------------------
def train_keyword_spotting():
    keywords = ["wash", "water", "year", "had"]
    print("提取关键词音节MFCC均值...")
    keyword_mfccs = extract_keyword_mfcc_by_syllable(
        TIMIT_DIR, keywords, n_mfcc=N_MFCC, samples_per_word=20
    )
    SEQUENCES_KEYWORD = []
    keyword_id_map = []
    for i, word in enumerate(keywords):
        for chunks in keyword_mfccs[word]:
            SEQUENCES_KEYWORD.append(chunks)
            keyword_id_map.append(i)
    dataset_train = KeywordDataset(
        SEQUENCES_KEYWORD, keyword_id_map, SAMPLE_PER_CLASS, mode="train"
    )
    dataset_test = KeywordDataset(
        SEQUENCES_KEYWORD, keyword_id_map, TEST_SAMPLE_PER_CLASS, mode="test"
    )
    loader_train = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    loader_test = DataLoader(
        dataset_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )
    model = CustomContinuousRNN(N_MFCC, HIDDEN_DIM, len(keywords), TAU).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    loss_fn = nn.MSELoss()
    print("开始训练关键词检测模型...")
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
                pred = out.mean(dim=1).argmax(dim=1)
                target = y.mean(dim=1).argmax(dim=1)
                correct += (pred == target).sum().item()
                total += x.size(0)
        print(
            f"Epoch {epoch+1} | TrainLoss: {total_loss/len(loader_train):.4f} | "
            f"TestLoss: {test_loss/len(loader_test):.4f} | Acc: {correct/total*100:.2f}%"
        )
    # 保存模型
    os.makedirs(MODEL_DIR, exist_ok=True)
    current_time = time.strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(
        MODEL_DIR,
        f"keyword_rnn_mfcc{N_MFCC}_hid{HIDDEN_DIM}_spc{SAMPLE_PER_CLASS}_sec4_{current_time}.pt",
    )
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    train_keyword_spotting()
