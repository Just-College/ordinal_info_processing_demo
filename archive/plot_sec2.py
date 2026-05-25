# ----------------------------
# Sec.2 visualization: output behavior, PCA variance, and 3D tree trajectories
# ----------------------------
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from train_sec2_synthetic import (
    DEVICE,
    HIDDEN_DIM,
    N_MFCC,
    PHONEME_LEN,
    T_MAX,
    TAU,
    SimpleRNN,
    SyntheticSequenceDataset,
)

MODEL_DIR = f"{Path(__file__).parent.parent}/model/"
FIG_DIR = f"{Path(__file__).parent.parent}/figure/"
MODEL_NAME = "rnn_mfcc3_hid64_spc128_sec2_seed1107.pt"
CHECKPOINT = f"{MODEL_DIR}/{MODEL_NAME}"

SAMPLE_PER_CLASS = 256
N_SAMPLES_PER_CLASS = 25
SELECT_CLASS_ID = 2  # None means random class.
NOISE_STD = 0.0
NODE_WINDOW = 5
ENABLE_PLOT_SHOW = False

SYN_PRIMITIVES = [
    np.array([1, 1, 0], dtype=np.float32),
    np.array([1, 0, 1], dtype=np.float32),
    np.array([0, 1, 1], dtype=np.float32),
]

# Must match train_sec2.py exactly.
SEQUENCES_SYN = [
    (SYN_PRIMITIVES[0], SYN_PRIMITIVES[1]),  # ab
    (SYN_PRIMITIVES[0], SYN_PRIMITIVES[2]),  # ac
    (SYN_PRIMITIVES[1], SYN_PRIMITIVES[0]),  # ba
    (SYN_PRIMITIVES[1], SYN_PRIMITIVES[2]),  # bc
]
SEQUENCE_NAMES = ["ab", "ac", "ba", "bc"]

Path(FIG_DIR).mkdir(exist_ok=True)


def save_and_maybe_show(out_path, dpi, show):
    plt.savefig(out_path, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def resolve_checkpoint():
    if os.path.exists(CHECKPOINT):
        return CHECKPOINT

    candidates = sorted(
        Path(MODEL_DIR).glob("rnn_mfcc3_hid*_spc*_sec2_*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No sec2 checkpoint found in {MODEL_DIR}. Train src/train_sec2.py first."
        )
    checkpoint = str(candidates[0])
    print(f"Configured checkpoint not found. Using latest checkpoint: {checkpoint}")
    return checkpoint


def load_state_dict(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def forward_with_rates(model, input_seq):
    B, T, _ = input_seq.shape
    x = torch.zeros(B, model.hidden_dim, device=input_seq.device)
    r = torch.tanh(x)
    outputs = []
    rates = []

    for t in range(T):
        input_t = input_seq[:, t, :]
        dx = (
            -x + torch.matmul(r, model.W_rec.T) + model.W_in(input_t) + model.I_b
        ) / model.tau
        x = x + dx
        r = torch.tanh(x)
        outputs.append(model.W_out(r).unsqueeze(1))
        rates.append(r.unsqueeze(1))

    return torch.cat(outputs, dim=1), torch.cat(rates, dim=1)


def manual_pca(activations, n_components=3):
    x = activations - np.mean(activations, axis=0, keepdims=True)
    cov = np.dot(x.T, x) / (x.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    explained_var = eigvals / np.sum(eigvals)
    projected = np.dot(x, eigvecs[:, :n_components])
    return projected, explained_var, eigvals, eigvecs[:, :n_components]


def save_pca_variance(activations, show=False):
    max_components = min(activations.shape[1], 10)

    sklearn_pca = PCA(n_components=max_components)
    sklearn_pca.fit(activations)
    sklearn_cumulative = np.cumsum(sklearn_pca.explained_variance_ratio_)

    _, manual_var, _, _ = manual_pca(activations, n_components=max_components)
    manual_cumulative = np.cumsum(manual_var[:max_components])

    plt.figure(figsize=(7, 4))
    xs = np.arange(1, max_components + 1)
    plt.plot(xs, sklearn_cumulative, marker="o", label="sklearn PCA")
    plt.plot(xs, manual_cumulative, marker="s", label="manual PCA")
    plt.xlabel("PCA Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("RNN Activity PCA Variance")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = f"{FIG_DIR}/pca_variance_curve.png"
    save_and_maybe_show(out_path, dpi=200, show=show)
    print(f"Saved PCA variance curve: {out_path}")
    print(
        f"Top 3 cumulative explained variance: {sklearn_cumulative[min(2, max_components - 1)]:.4f}"
    )


def save_output_curve(outputs, targets, class_id, show=False):
    mean_outputs = outputs.mean(axis=0)
    std_outputs = outputs.std(axis=0)
    mean_targets = targets.mean(axis=0)
    std_targets = targets.std(axis=0)
    t_axis = np.arange(mean_outputs.shape[0])

    plt.figure(figsize=(10, 5))
    for neuron_id in range(mean_outputs.shape[1]):
        plt.plot(
            t_axis,
            mean_outputs[:, neuron_id],
            label=f"Neuron {neuron_id} output",
            linewidth=2,
        )
        plt.fill_between(
            t_axis,
            mean_outputs[:, neuron_id] - std_outputs[:, neuron_id],
            mean_outputs[:, neuron_id] + std_outputs[:, neuron_id],
            alpha=0.18,
        )
        plt.plot(
            t_axis,
            mean_targets[:, neuron_id],
            "--",
            label=f"Neuron {neuron_id} target",
            alpha=0.7,
        )
        plt.fill_between(
            t_axis,
            mean_targets[:, neuron_id] - std_targets[:, neuron_id],
            mean_targets[:, neuron_id] + std_targets[:, neuron_id],
            alpha=0.08,
        )

    second_start = T_MAX + PHONEME_LEN + T_MAX
    plt.axvspan(T_MAX, T_MAX + PHONEME_LEN, color="orange", alpha=0.2, label="Item 1")
    plt.axvspan(
        second_start,
        second_start + PHONEME_LEN,
        color="green",
        alpha=0.2,
        label="Item 2",
    )
    plt.xlabel("Time step")
    plt.ylabel("Activation")
    plt.title(
        f"Output vs target for class {class_id} ({SEQUENCE_NAMES[class_id]}), n={outputs.shape[0]}"
    )
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    out_path = f"{FIG_DIR}/multi_sample_output_curve_class{class_id}.png"
    save_and_maybe_show(out_path, dpi=200, show=show)
    print(f"Saved output curve: {out_path}")


def save_3d_trajectories(rates_by_class, pca, show=False):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    root_start = max(T_MAX - NODE_WINDOW, 0)

    for class_id, rates in rates_by_class.items():
        mean_rates = rates.mean(axis=0)
        traj = pca.transform(mean_rates[root_start:])
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            traj[:, 2],
            color=colors[class_id % len(colors)],
            linewidth=2,
            label=f"Class {class_id} ({SEQUENCE_NAMES[class_id]})",
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("RNN Activity Trajectories Projected onto First 3 PCs")
    ax.legend(loc="best")
    plt.tight_layout()
    out_path = f"{FIG_DIR}/rnn_activity_pca_3d_traj_per_class.png"
    save_and_maybe_show(out_path, dpi=200, show=show)
    print(f"Saved 3D trajectory plot: {out_path}")


def tree_node_estimates(rates_by_class):
    first_item_end = T_MAX + PHONEME_LEN + T_MAX
    sequence_end = T_MAX + PHONEME_LEN + T_MAX + PHONEME_LEN + T_MAX

    root = np.mean(
        [
            rates[:, T_MAX - NODE_WINDOW : T_MAX, :].mean(axis=(0, 1))
            for rates in rates_by_class.values()
        ],
        axis=0,
    )
    branch_a = np.mean(
        [
            rates_by_class[class_id][
                :, first_item_end - NODE_WINDOW : first_item_end, :
            ].mean(axis=(0, 1))
            for class_id in [0, 1]
        ],
        axis=0,
    )
    branch_b = np.mean(
        [
            rates_by_class[class_id][
                :, first_item_end - NODE_WINDOW : first_item_end, :
            ].mean(axis=(0, 1))
            for class_id in [2, 3]
        ],
        axis=0,
    )
    leaves = [
        rates_by_class[class_id][:, sequence_end - NODE_WINDOW : sequence_end, :].mean(
            axis=(0, 1)
        )
        for class_id in range(len(SEQUENCES_SYN))
    ]

    names = ["root", "a", "b", "ab", "ac", "ba", "bc"]
    nodes = np.vstack([root, branch_a, branch_b, *leaves])
    return names, nodes


def draw_segment(ax, start, end, **kwargs):
    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], **kwargs)


def save_figure2e(rates_by_class, pca, show=False):
    fig = plt.figure(figsize=(7.4, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    colors = ["#d62728", "#2ca02c", "#ff7f0e", "#1f77b4"]
    root_start = max(T_MAX - NODE_WINDOW, 0)

    for class_id in range(len(SEQUENCES_SYN)):
        mean_rates = rates_by_class[class_id].mean(axis=0)
        traj = pca.transform(mean_rates[root_start:])
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            traj[:, 2],
            color=colors[class_id],
            linewidth=2.4,
            label=f"$S_{class_id + 1}$ ({SEQUENCE_NAMES[class_id]})",
        )
        ax.scatter(
            traj[0, 0], traj[0, 1], traj[0, 2], color=colors[class_id], s=18, alpha=0.75
        )

    node_names, nodes = tree_node_estimates(rates_by_class)
    projected_nodes = pca.transform(nodes)

    # Draw the abstract tree skeleton over the averaged trajectories.
    skeleton_edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
    for start_idx, end_idx in skeleton_edges:
        draw_segment(
            ax,
            projected_nodes[start_idx],
            projected_nodes[end_idx],
            color="black",
            linewidth=1.0,
            alpha=0.35,
            linestyle="--",
        )

    ax.scatter(
        projected_nodes[:, 0],
        projected_nodes[:, 1],
        projected_nodes[:, 2],
        marker="x",
        s=95,
        color="black",
        linewidths=2.2,
        label="tree nodes",
    )
    for name, point in zip(node_names, projected_nodes):
        ax.text(point[0], point[1], point[2], f" {name}", fontsize=9)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("Figure 2E reproduction: tree-structured RNN trajectories")
    ax.view_init(elev=24, azim=-56)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    out_path = f"{FIG_DIR}/figure2E_tree_structure.png"

    save_and_maybe_show(out_path, dpi=300, show=show)
    print(f"Saved Figure 2E reproduction: {out_path}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    dataset = SyntheticSequenceDataset(
        SEQUENCES_SYN,
        SAMPLE_PER_CLASS,
        T_MAX,
        NOISE_STD,
        phoneme_len=PHONEME_LEN,
        fixed=True,
    )

    model = SimpleRNN(N_MFCC, HIDDEN_DIM, len(SEQUENCES_SYN), tau=TAU).to(DEVICE)
    checkpoint = resolve_checkpoint()
    model.load_state_dict(load_state_dict(checkpoint))
    model.eval()

    outputs_by_class = {class_id: [] for class_id in range(len(SEQUENCES_SYN))}
    rates_by_class = {class_id: [] for class_id in range(len(SEQUENCES_SYN))}
    targets_by_class = {class_id: [] for class_id in range(len(SEQUENCES_SYN))}

    with torch.no_grad():
        for idx in range(len(dataset)):
            x, target, class_id, _ = dataset[idx]
            if len(outputs_by_class[class_id]) >= N_SAMPLES_PER_CLASS:
                continue

            output, rates = forward_with_rates(model, x.unsqueeze(0).to(DEVICE))
            outputs_by_class[class_id].append(output.squeeze(0).cpu().numpy())
            rates_by_class[class_id].append(rates.squeeze(0).cpu().numpy())
            targets_by_class[class_id].append(target.cpu().numpy())

            if all(
                len(values) >= N_SAMPLES_PER_CLASS
                for values in outputs_by_class.values()
            ):
                break

    missing = {
        class_id: N_SAMPLES_PER_CLASS - len(values)
        for class_id, values in outputs_by_class.items()
        if len(values) < N_SAMPLES_PER_CLASS
    }
    if missing:
        print(f"Warning: insufficient samples for classes: {missing}")

    outputs_by_class = {
        class_id: np.stack(values, axis=0)
        for class_id, values in outputs_by_class.items()
        if values
    }
    rates_by_class = {
        class_id: np.stack(values, axis=0)
        for class_id, values in rates_by_class.items()
        if values
    }
    targets_by_class = {
        class_id: np.stack(values, axis=0)
        for class_id, values in targets_by_class.items()
        if values
    }

    selected_class_id = (
        np.random.choice(list(outputs_by_class.keys()))
        if SELECT_CLASS_ID is None
        else SELECT_CLASS_ID
    )
    save_output_curve(
        outputs_by_class[selected_class_id],
        targets_by_class[selected_class_id],
        selected_class_id,
        show=ENABLE_PLOT_SHOW,
    )

    all_rates = np.concatenate(
        [rates.reshape(-1, rates.shape[-1]) for rates in rates_by_class.values()],
        axis=0,
    )
    save_pca_variance(all_rates, show=ENABLE_PLOT_SHOW)

    pca = PCA(n_components=3)
    pca.fit(all_rates)
    save_3d_trajectories(rates_by_class, pca, show=ENABLE_PLOT_SHOW)
    save_figure2e(rates_by_class, pca, show=ENABLE_PLOT_SHOW)


if __name__ == "__main__":
    main()
