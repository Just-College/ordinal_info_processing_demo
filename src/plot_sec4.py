from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from train_sec4 import (DEVICE, HIDDEN_DIM, N_MFCC, TAU, CustomContinuousRNN,
                        KeywordDataset, collate_fn,
                        extract_keyword_mfcc_by_syllable)

MODEL_DIR = f"{Path(__file__).parent.parent}/model/"
MODEL_NAME = "keyword_rnn_mfcc16_hid128_spc512_sec4_20250716_063114.pt"
CHECKPOINT = f"{MODEL_DIR}/{MODEL_NAME}"

# 关键词与数据集参数
keywords = ["wash", "water", "year", "had"]
SAMPLE_PER_CLASS = 10

# 构造测试集（与训练脚本一致，固定噪声区间）
TIMIT_DIR = f"{Path(__file__).parent.parent}/data/TIMIT/TRAIN/"
keyword_mfccs = extract_keyword_mfcc_by_syllable(
    TIMIT_DIR, keywords, n_mfcc=N_MFCC, samples_per_word=20
)
SEQUENCES_KEYWORD = []
keyword_id_map = []
for i, word in enumerate(keywords):
    for chunks in keyword_mfccs[word]:
        SEQUENCES_KEYWORD.append(chunks)
        keyword_id_map.append(i)
dataset_test = KeywordDataset(
    SEQUENCES_KEYWORD, keyword_id_map, SAMPLE_PER_CLASS, mode="test"
)
dataloader = DataLoader(
    dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn
)

# 加载模型
model = CustomContinuousRNN(N_MFCC, HIDDEN_DIM, len(keywords), TAU).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

# 1. 绘制模型在某一条测试数据下的输出曲线
for idx, (x, y) in enumerate(dataloader):
    x = x.to(DEVICE)
    with torch.no_grad():
        out = model(x)  # [1, T, 4]
    out_np = out.squeeze(0).cpu().numpy()  # [T, 4]
    y_np = y.squeeze(0).cpu().numpy()  # [T, 4]
    plt.figure(figsize=(10, 5))
    t_axis = np.arange(out_np.shape[0])
    for i in range(out_np.shape[1]):
        plt.plot(t_axis, out_np[:, i], label=f"Neuron {i} Output", linewidth=2)
        plt.plot(t_axis, y_np[:, i], "--", label=f"Neuron {i} Target", alpha=0.7)
    plt.xlabel("Time step")
    plt.ylabel("Activation")
    plt.title(f"Model Output vs. Target (Test Sample #{idx})")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(f"output_curve_test_sample{idx}.png", dpi=200)
    plt.show()
    break  # 只画一条测试数据

# 2. 分析PCA数量（累计方差解释曲线）
all_outputs = []
for x, y in dataloader:
    x = x.to(DEVICE)
    with torch.no_grad():
        out = model(x)
    all_outputs.append(out.squeeze(0).cpu().numpy())  # [T, 4]
all_outputs = np.concatenate(all_outputs, axis=0)  # [N*T, 4]
max_components = min(all_outputs.shape[1], 10)
pca = PCA(n_components=max_components)
pca.fit(all_outputs)
explained_var = pca.explained_variance_ratio_
plt.figure(figsize=(7, 4))
plt.plot(
    np.arange(1, max_components + 1),
    np.cumsum(explained_var),
    marker="o",
    label="sklearn PCA",
)
plt.xlabel("PCA Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Components vs. Cumulative Variance (Keyword Task)")
plt.grid(True)
plt.tight_layout()
plt.savefig("pca_variance_curve_keyword.png", dpi=200)
plt.show()

# 3. 从每一个类别里随机选一个样本，输出轨迹投射到前三个主成分空间
# 先用PCA得到主成分
pca = PCA(n_components=3)
pca.fit(all_outputs)
components = pca.components_  # [3, 4]
mean_output = np.mean(all_outputs, axis=0, keepdims=True)

# 按类别分组索引
class_indices = {cid: [] for cid in range(len(keywords))}
for idx, (_, y) in enumerate(dataloader):
    # 获取类别id（标签最大值所在列）
    label_id = int(y.mean(dim=1).argmax(dim=1)[0].item())
    class_indices[label_id].append(idx)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
for class_id in range(len(keywords)):
    indices = class_indices[class_id]
    if len(indices) == 0:
        print(f"警告：class_id={class_id}无样本可用，跳过。")
        continue
    idx = np.random.choice(indices)
    # 重新获取该样本的输出
    x, y = list(dataloader)[idx]
    x = x.to(DEVICE)
    with torch.no_grad():
        out = model(x)
    activs = out.squeeze(0).cpu().numpy()  # [T, 4]
    # 零均值化
    activs_centered = activs - mean_output
    traj = np.dot(activs_centered, components.T)  # [T, 3]
    ax.plot(
        traj[:, 0],
        traj[:, 1],
        traj[:, 2],
        color=colors[class_id % len(colors)],
        alpha=0.9,
        label=f"Class {class_id} ({keywords[class_id]})",
    )
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title(
    "RNN Output Trajectories (One Sample per Class) Projected onto First 3 PCA Components"
)
ax.legend(loc="best")
plt.tight_layout()
plt.savefig("output_pca_3d_traj_per_class_keyword.png", dpi=200)
plt.show()
