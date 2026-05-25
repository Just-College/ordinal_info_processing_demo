import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets import SyntheticSequenceDataset
from models import SimpleRNN

REPO_ROOT = Path(__file__).parent.parent
MODEL_DIR = REPO_ROOT / "model"
FIG_DIR = REPO_ROOT / "figure"
MODEL_NAME = "rnn_mfcc3_hid64_spc1024_sec2_seed1107.pt"
CHECKPOINT = MODEL_DIR / MODEL_NAME

T_MAX = 30
PHONEME_LEN = 3
N_MFCC = 3
HIDDEN_DIM = 64
TAU = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1107

SAMPLE_PER_CLASS = 256
N_SAMPLES_PER_CLASS = 25
SELECT_CLASS_ID = 2
NOISE_STD = 0.0
ENABLE_PLOT_SHOW = False

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_and_maybe_show(out_path, dpi, show):
    plt.savefig(out_path, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def resolve_checkpoint():
    if CHECKPOINT.exists():
        return CHECKPOINT

    candidates = sorted(
        MODEL_DIR.glob("rnn_mfcc3_hid*_spc*_sec2_*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No sec2 checkpoint found in {MODEL_DIR}. Train src/train_sec2_synthetic.py first."
        )
    checkpoint = candidates[0]
    print(f"Configured checkpoint not found. Using latest checkpoint: {checkpoint}")
    return checkpoint


def load_state_dict(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def collect_outputs(model):
    dataset = SyntheticSequenceDataset(
        SEQUENCES_SYN,
        SAMPLE_PER_CLASS,
        T_MAX,
        NOISE_STD,
        phoneme_len=PHONEME_LEN,
        fixed=True,
    )
    outputs_by_class = {class_id: [] for class_id in range(len(SEQUENCES_SYN))}
    targets_by_class = {class_id: [] for class_id in range(len(SEQUENCES_SYN))}

    with torch.no_grad():
        for idx in range(len(dataset)):
            x, target, class_id, _ = dataset[idx]
            if len(outputs_by_class[class_id]) >= N_SAMPLES_PER_CLASS:
                continue

            output = model(x.unsqueeze(0).to(DEVICE))
            outputs_by_class[class_id].append(output.squeeze(0).cpu().numpy())
            targets_by_class[class_id].append(target.cpu().numpy())

            if all(
                len(values) >= N_SAMPLES_PER_CLASS
                for values in outputs_by_class.values()
            ):
                break

    outputs_by_class = {
        class_id: np.stack(values, axis=0)
        for class_id, values in outputs_by_class.items()
        if values
    }
    targets_by_class = {
        class_id: np.stack(values, axis=0)
        for class_id, values in targets_by_class.items()
        if values
    }
    return outputs_by_class, targets_by_class


def plot_output_curve(outputs, targets, class_id, show=False):
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

    ax = plt.gca()
    ax.spines[["top", "right"]].set_visible(False)

    plt.xlabel("Timestep")
    plt.ylabel("Outputs")
    plt.title(f"Model output for class {class_id} ({SEQUENCE_NAMES[class_id]})")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    out_path = FIG_DIR / f"sec2_model_output_class{class_id}.png"
    save_and_maybe_show(out_path, dpi=200, show=show)
    print(f"Saved model output curve: {out_path}")


def main():
    set_seed(SEED)
    os.makedirs(FIG_DIR, exist_ok=True)

    model = SimpleRNN(N_MFCC, HIDDEN_DIM, len(SEQUENCES_SYN), tau=TAU).to(DEVICE)
    checkpoint = resolve_checkpoint()
    print(f"Checkpoint: {checkpoint}")
    model.load_state_dict(load_state_dict(checkpoint))
    model.eval()

    outputs_by_class, targets_by_class = collect_outputs(model)
    if SELECT_CLASS_ID not in outputs_by_class:
        raise ValueError(f"Class {SELECT_CLASS_ID} was not collected.")
    plot_output_curve(
        outputs_by_class[SELECT_CLASS_ID],
        targets_by_class[SELECT_CLASS_ID],
        SELECT_CLASS_ID,
        show=ENABLE_PLOT_SHOW,
    )


if __name__ == "__main__":
    main()
