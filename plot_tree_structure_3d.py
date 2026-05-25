import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from datasets import (
    SYNTHETIC_SEQUENCE_NAMES,
    SYNTHETIC_SEQUENCES,
    SyntheticSequenceDataset,
    collate_fn,
)
from models import CustomContinuousRNN
from train_utils import FIGURE_DIR, MODEL_DIR, ensure_output_dirs, load_checkpoint


CHECKPOINT = MODEL_DIR / "synthetic_latest.pt"
SAMPLES_PER_CLASS = 50
BATCH_SIZE = 64
NODE_WINDOW = 5


def collect_class_rates(model, dataset, loader, device):
    rates_by_class = {class_id: [] for class_id in range(len(SYNTHETIC_SEQUENCES))}
    cursor = 0
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            batch_size = x.shape[0]
            _, rates, _ = model(x.to(device), return_states=True)
            rates = rates.cpu().numpy()
            for batch_idx in range(batch_size):
                class_id = dataset.labels[cursor + batch_idx][0]
                rates_by_class[class_id].append(rates[batch_idx])
            cursor += batch_size
    return {class_id: np.stack(values, axis=0) for class_id, values in rates_by_class.items()}


def mean_node(rate_trials, start, end):
    return rate_trials[:, start:end, :].mean(axis=(0, 1))


def main():
    ensure_output_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict, metadata = load_checkpoint(CHECKPOINT, device)

    hidden_dim = int(metadata.get("hidden_dim", 50))
    input_dim = int(metadata.get("input_dim", 3))
    output_dim = int(metadata.get("output_dim", len(SYNTHETIC_SEQUENCES)))
    tau = float(metadata.get("tau", 2.0))
    t_max = int(metadata.get("t_max", 30))
    phoneme_len = int(metadata.get("phoneme_len", 3))
    noise_std = float(metadata.get("noise_std", 0.01))

    model = CustomContinuousRNN(input_dim, hidden_dim, output_dim, tau=tau).to(device)
    model.load_state_dict(state_dict)

    dataset = SyntheticSequenceDataset(
        SYNTHETIC_SEQUENCES,
        SAMPLES_PER_CLASS,
        t_max,
        noise_std,
        phoneme_len=phoneme_len,
        fixed=True,
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )
    rates_by_class = collect_class_rates(model, dataset, loader, device)

    all_rates = np.concatenate(
        [rates.reshape(-1, rates.shape[-1]) for rates in rates_by_class.values()],
        axis=0,
    )
    pca = PCA(n_components=3)
    pca.fit(all_rates)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    colors = ["tab:red", "tab:green", "tab:orange", "tab:blue"]

    projected_trajectories = {}
    for class_id, rate_trials in rates_by_class.items():
        mean_traj = rate_trials.mean(axis=0)
        projected = pca.transform(mean_traj)
        projected_trajectories[class_id] = projected
        ax.plot(
            projected[:, 0],
            projected[:, 1],
            projected[:, 2],
            color=colors[class_id],
            linewidth=2,
            label=f"S{class_id + 1} ({SYNTHETIC_SEQUENCE_NAMES[class_id]})",
        )

    first_end = t_max + phoneme_len + t_max
    leaf_start = first_end + phoneme_len + t_max - NODE_WINDOW
    leaf_end = first_end + phoneme_len + t_max

    root = np.mean(
        [
            mean_node(rates, 0, min(NODE_WINDOW, t_max))
            for rates in rates_by_class.values()
        ],
        axis=0,
    )
    branch_a = np.mean(
        [mean_node(rates_by_class[i], first_end - NODE_WINDOW, first_end) for i in [0, 1]],
        axis=0,
    )
    branch_b = np.mean(
        [mean_node(rates_by_class[i], first_end - NODE_WINDOW, first_end) for i in [2, 3]],
        axis=0,
    )
    leaf_nodes = [
        mean_node(rates_by_class[i], leaf_start, leaf_end)
        for i in range(len(SYNTHETIC_SEQUENCES))
    ]
    node_names = ["root", "a*", "b*", "ab", "ac", "ba", "bc"]
    nodes = np.vstack([root, branch_a, branch_b, *leaf_nodes])
    projected_nodes = pca.transform(nodes)

    ax.scatter(
        projected_nodes[:, 0],
        projected_nodes[:, 1],
        projected_nodes[:, 2],
        marker="x",
        s=90,
        color="black",
        linewidths=2,
        label="Tree nodes",
    )
    for name, point in zip(node_names, projected_nodes):
        ax.text(point[0], point[1], point[2], name, fontsize=9)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("Learned 3D Tree Structure of RNN Activity")
    ax.legend(loc="best")
    plt.tight_layout()

    out_fig = FIGURE_DIR / "tree_structure_3d.png"
    plt.savefig(out_fig, dpi=220)
    print(f"Saved 3D tree figure: {out_fig}")


if __name__ == "__main__":
    main()
