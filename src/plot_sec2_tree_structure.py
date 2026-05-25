import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from datasets import SyntheticSequenceDataset, collate_fn
from models import SimpleRNN

MODEL_DIR = f"{Path(__file__).parent.parent}/model/"
FIG_DIR = f"{Path(__file__).parent.parent}/figure/"
MODEL_NAME = "rnn_mfcc3_hid64_spc1024_sec2_seed1107.pt"
CHECKPOINT = f"{MODEL_DIR}/{MODEL_NAME}"

T_MAX = 30
PHONEME_LEN = 3
N_MFCC = 3
HIDDEN_DIM = 64
TAU = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLE_PER_CLASS = 256
N_SAMPLES_PER_CLASS = 25
NOISE_STD = 0.0
NODE_WINDOW = 5
ENABLE_PLOT_SHOW = False

OPT_STEPS = 2000
LBFGS_STEPS = 200
FIXED_POINT_LR = 1e-2
RESIDUAL_RMS_TOL = 1e-3
NEAR_NODE_RATIO_TOL = 0.5

SYN_PRIMITIVES = [
    np.array([1, 1, 0], dtype=np.float32),
    np.array([1, 0, 1], dtype=np.float32),
    np.array([0, 1, 1], dtype=np.float32),
]

SEQUENCES_SYN = [
    (SYN_PRIMITIVES[0], SYN_PRIMITIVES[1]),  # ab
    (SYN_PRIMITIVES[0], SYN_PRIMITIVES[2]),  # ac
    (SYN_PRIMITIVES[1], SYN_PRIMITIVES[0]),  # ba
    (SYN_PRIMITIVES[1], SYN_PRIMITIVES[2]),  # bc
]
SEQUENCE_NAMES = ["ab", "ac", "ba", "bc"]


OUT_FIG = f"{FIG_DIR}/figure2E_tree_with_stable_fixed_points.png"


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


def draw_segment(ax, start, end, **kwargs):
    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], **kwargs)


def forward_with_rates_and_states(model, input_seq):
    batch_size, timesteps, _ = input_seq.shape
    x = torch.zeros(batch_size, model.hidden_dim, device=input_seq.device)
    r = torch.tanh(x)
    rates = []
    states = []

    for t in range(timesteps):
        input_t = input_seq[:, t, :]
        dx = (
            -x + torch.matmul(r, model.W_rec.T) + model.W_in(input_t) + model.I_b
        ) / model.tau
        x = x + dx
        r = torch.tanh(x)
        rates.append(r.unsqueeze(1))
        states.append(x.unsqueeze(1))

    return torch.cat(rates, dim=1), torch.cat(states, dim=1)


def collect_activity_by_class(model):
    dataset = SyntheticSequenceDataset(
        SEQUENCES_SYN,
        SAMPLE_PER_CLASS,
        T_MAX,
        NOISE_STD,
        phoneme_len=PHONEME_LEN,
        fixed=True,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    rates_by_class = {class_id: [] for class_id in range(len(SEQUENCES_SYN))}
    states_by_class = {class_id: [] for class_id in range(len(SEQUENCES_SYN))}

    with torch.no_grad():
        for x, _, _, class_ids, _, _ in loader:
            rates, states = forward_with_rates_and_states(model, x.to(DEVICE))
            rates = rates.cpu()
            states = states.cpu()
            for sample_idx, class_id in enumerate(class_ids.tolist()):
                if len(rates_by_class[class_id]) >= N_SAMPLES_PER_CLASS:
                    continue
                rates_by_class[class_id].append(rates[sample_idx].numpy())
                states_by_class[class_id].append(states[sample_idx].numpy())

            if all(
                len(values) >= N_SAMPLES_PER_CLASS for values in rates_by_class.values()
            ):
                break

    rates_by_class = {
        class_id: np.stack(values, axis=0)
        for class_id, values in rates_by_class.items()
        if values
    }
    states_by_class = {
        class_id: np.stack(values, axis=0)
        for class_id, values in states_by_class.items()
        if values
    }
    return rates_by_class, states_by_class


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
    names = ["root", "a", "b", *SEQUENCE_NAMES]
    nodes = np.vstack([root, branch_a, branch_b, *leaves])
    return names, nodes


def node_state_estimates(states_by_class):
    first_item_end = T_MAX + PHONEME_LEN + T_MAX
    sequence_end = T_MAX + PHONEME_LEN + T_MAX + PHONEME_LEN + T_MAX

    root = np.mean(
        [
            states[:, T_MAX - NODE_WINDOW : T_MAX, :].mean(axis=(0, 1))
            for states in states_by_class.values()
        ],
        axis=0,
    )
    branch_a = np.mean(
        [
            states_by_class[class_id][
                :, first_item_end - NODE_WINDOW : first_item_end, :
            ].mean(axis=(0, 1))
            for class_id in [0, 1]
        ],
        axis=0,
    )
    branch_b = np.mean(
        [
            states_by_class[class_id][
                :, first_item_end - NODE_WINDOW : first_item_end, :
            ].mean(axis=(0, 1))
            for class_id in [2, 3]
        ],
        axis=0,
    )
    leaves = [
        states_by_class[class_id][:, sequence_end - NODE_WINDOW : sequence_end, :].mean(
            axis=(0, 1)
        )
        for class_id in range(len(SEQUENCES_SYN))
    ]
    names = ["root", "a", "b", *SEQUENCE_NAMES]
    states = np.vstack([root, branch_a, branch_b, *leaves])
    return names, states


def nearest_node_distances(node_states):
    distances = []
    for idx, state in enumerate(node_states):
        others = np.delete(node_states, idx, axis=0)
        distances.append(np.linalg.norm(others - state, axis=1).min())
    return distances


def vector_field(model, x, input_value):
    r = torch.tanh(x)
    return -x + torch.matmul(r, model.W_rec.T) + model.W_in(input_value) + model.I_b


def optimize_fixed_point(model, initial_state):
    zero_input = torch.zeros(N_MFCC, device=DEVICE)
    initial = torch.tensor(initial_state, dtype=torch.float32, device=DEVICE)
    x = initial.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=FIXED_POINT_LR)

    with torch.no_grad():
        initial_residual = vector_field(model, initial, zero_input).norm().item()

    for _ in range(OPT_STEPS):
        optimizer.zero_grad()
        residual = vector_field(model, x, zero_input)
        loss = torch.mean(residual.square())
        loss.backward()
        optimizer.step()

    lbfgs = torch.optim.LBFGS(
        [x], lr=1.0, max_iter=LBFGS_STEPS, line_search_fn="strong_wolfe"
    )

    def closure():
        lbfgs.zero_grad()
        residual = vector_field(model, x, zero_input)
        loss = torch.mean(residual.square())
        loss.backward()
        return loss

    lbfgs.step(closure)

    with torch.no_grad():
        residual = vector_field(model, x, zero_input)
        final_residual = residual.norm().item()
        final_residual_rms = torch.sqrt(torch.mean(residual.square())).item()
        distance = (x - initial).norm().item()
        rate_distance = (torch.tanh(x) - torch.tanh(initial)).norm().item()
        output = model.W_out(torch.tanh(x)).detach().cpu().numpy()
        fixed_state = x.detach().cpu().numpy()

    return {
        "initial_residual": initial_residual,
        "final_residual": final_residual,
        "final_residual_rms": final_residual_rms,
        "state_distance": distance,
        "rate_distance": rate_distance,
        "fixed_state": fixed_state,
        "output": output,
    }


def stability_metrics(model, fixed_state):
    x = torch.tensor(fixed_state, dtype=torch.float32, device=DEVICE)
    r = torch.tanh(x)
    gain = 1.0 - r.square()
    jac_f = -torch.eye(model.hidden_dim, device=DEVICE) + model.W_rec * gain.unsqueeze(
        0
    )
    jac_step = torch.eye(model.hidden_dim, device=DEVICE) + jac_f / model.tau
    eig_f = torch.linalg.eigvals(jac_f).detach().cpu().numpy()
    eig_step = torch.linalg.eigvals(jac_step).detach().cpu().numpy()
    spectral_radius = float(np.abs(eig_step).max())
    max_real = float(np.real(eig_f).max())
    return {
        "step_spectral_radius": spectral_radius,
        "continuous_max_real_eig": max_real,
        "discrete_stable": spectral_radius < 1.0,
        "continuous_stable": max_real < 0.0,
    }


def find_fixed_points(model, states_by_class):
    node_names, node_states = node_state_estimates(states_by_class)
    nearest_distances = nearest_node_distances(node_states)
    rows = []
    fixed_states = []

    for node_name, node_state, nearest_distance in zip(
        node_names, node_states, nearest_distances
    ):
        result = optimize_fixed_point(model, node_state)
        stability = stability_metrics(model, result["fixed_state"])
        near_ratio = result["state_distance"] / max(nearest_distance, 1e-12)
        fixed_point_found = (
            result["final_residual_rms"] <= RESIDUAL_RMS_TOL
            and near_ratio <= NEAR_NODE_RATIO_TOL
        )
        stable = (
            fixed_point_found
            and stability["discrete_stable"]
            and stability["continuous_stable"]
        )
        rows.append(
            {
                "node": node_name,
                "stable": stable,
                "final_residual_rms": result["final_residual_rms"],
                "distance_over_nearest_node_distance": near_ratio,
                "step_spectral_radius": stability["step_spectral_radius"],
                "continuous_max_real_eig": stability["continuous_max_real_eig"],
            }
        )
        fixed_states.append(result["fixed_state"])

    return rows, np.stack(fixed_states, axis=0)


def plot_tree_with_fixed_points(rates_by_class, fixed_rows, fixed_states):
    all_rates = np.concatenate(
        [rates.reshape(-1, rates.shape[-1]) for rates in rates_by_class.values()],
        axis=0,
    )
    pca = PCA(n_components=3)
    pca.fit(all_rates)

    fig = plt.figure(figsize=(8.0, 6.8))
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
            linewidth=2.2,
            alpha=0.9,
            label=f"$S_{class_id + 1}$ ({SEQUENCE_NAMES[class_id]})",
        )

    node_names, rate_nodes = tree_node_estimates(rates_by_class)
    projected_nodes = pca.transform(rate_nodes)
    skeleton_edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
    for start_idx, end_idx in skeleton_edges:
        draw_segment(
            ax,
            projected_nodes[start_idx],
            projected_nodes[end_idx],
            color="black",
            linewidth=1.0,
            alpha=0.3,
            linestyle="--",
        )

    for name, point in zip(node_names, projected_nodes):
        ax.text(point[0], point[1], point[2], f" {name}", fontsize=8)

    fixed_rates = np.tanh(fixed_states)
    projected_fixed = pca.transform(fixed_rates)
    stable_mask = np.array([row["stable"] for row in fixed_rows], dtype=bool)
    if stable_mask.any():
        stable_points = projected_fixed[stable_mask]
        ax.scatter(
            stable_points[:, 0],
            stable_points[:, 1],
            stable_points[:, 2],
            marker="X",
            s=95,
            color="black",
            linewidths=1.0,
            label="stable fixed points",
            zorder=10,
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("Sec.2 trajectories with stable fixed points")
    ax.view_init(elev=24, azim=-56)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    save_and_maybe_show(OUT_FIG, dpi=300, show=ENABLE_PLOT_SHOW)
    print(f"Saved trajectory/fixed-point plot: {OUT_FIG}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    checkpoint = resolve_checkpoint()
    print(f"Checkpoint: {checkpoint}")

    model = SimpleRNN(N_MFCC, HIDDEN_DIM, len(SEQUENCES_SYN), tau=TAU).to(DEVICE)
    model.load_state_dict(load_state_dict(checkpoint))
    model.eval()

    rates_by_class, states_by_class = collect_activity_by_class(model)
    fixed_rows, fixed_states = find_fixed_points(model, states_by_class)
    plot_tree_with_fixed_points(rates_by_class, fixed_rows, fixed_states)

    print("node stable final_rms near_ratio spectral_radius max_real")
    for row in fixed_rows:
        print(
            f"{row['node']:>4} "
            f"{row['stable']} "
            f"{row['final_residual_rms']:.6g} "
            f"{row['distance_over_nearest_node_distance']:.3f} "
            f"{row['step_spectral_radius']:.6g} "
            f"{row['continuous_max_real_eig']:.6g}"
        )


if __name__ == "__main__":
    main()
