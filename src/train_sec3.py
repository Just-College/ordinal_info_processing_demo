# 仅保留参数管理和任务入口
import glob
import os
import random
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import SyntheticSequenceDataset, collate_fn
from models import CustomContinuousRNN
from train_utils import train_model

# 全局参数
TIMIT_DIR = (
    f"{Path(__file__).parent.parent}/data/TIMIT/TRAIN/"
)
MODEL_DIR = f"{Path(__file__).parent.parent}/model/"
CSV_DIR = f"{Path(__file__).parent.parent}/csv/"
MODEL_NAME = "GOOD_rnn_mfcc3_hid50_tau2_ep80_20250713_230504.pt"
PRETRAINED_MODEL_DIR = f"{MODEL_DIR}/{MODEL_NAME}"
USE_PRETRAINED = True
FREEZE = False
SAMPLE_PER_CLASS = 1024
TEST_SAMPLE_PER_CLASS = 216
T_MAX = 30
PHONEME_LEN = 3
NOISE_STD = 0.01
N_MFCC = 16
HIDDEN_DIM = 50
TAU = 2
BATCH_SIZE = 16
EPOCHS = 100
LR = 10 ** random.uniform(-4, -2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_phoneme_mfcc_mean(timit_root, phoneme_list, n_mfcc=3):
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
    phoneme_means = {}
    for p in phoneme_list:
        arr = np.stack(phoneme_mfccs[p], axis=0)
        phoneme_means[p] = arr.mean(axis=0)
    return phoneme_means


def main():
    phoneme_list = ["pcl", "tcl", "pau"]
    print("提取TIMIT音素MFCC均值...")
    phoneme_means = extract_phoneme_mfcc_mean(TIMIT_DIR, phoneme_list, n_mfcc=N_MFCC)
    a = phoneme_means["pcl"]
    b = phoneme_means["tcl"]
    c = phoneme_means["pau"]
    SEQUENCES_TIMIT = [(a, b), (a, c), (b, a), (b, c)]
    dataset_train = SyntheticSequenceDataset(
        SEQUENCES_TIMIT, SAMPLE_PER_CLASS, T_MAX, NOISE_STD, phoneme_len=PHONEME_LEN
    )
    loader_train = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    dataset_test = SyntheticSequenceDataset(
        SEQUENCES_TIMIT,
        TEST_SAMPLE_PER_CLASS,
        T_MAX,
        NOISE_STD,
        phoneme_len=PHONEME_LEN,
    )
    loader_test = DataLoader(
        dataset_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )
    model = CustomContinuousRNN(N_MFCC, HIDDEN_DIM, len(SEQUENCES_TIMIT), TAU).to(
        DEVICE
    )
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
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    loss_fn = torch.nn.MSELoss()
    params_str = f"mfcc{N_MFCC}_hid{HIDDEN_DIM}_spc{SAMPLE_PER_CLASS}_pretrained{USE_PRETRAINED}_freeze{FREEZE}_sec3"
    train_model(
        model,
        loader_train,
        loader_test,
        optimizer,
        scheduler,
        loss_fn,
        DEVICE,
        EPOCHS,
        CSV_DIR,
        MODEL_DIR,
        params_str,
    )


if __name__ == "__main__":
    main()
