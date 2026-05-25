import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from datasets import SYNTHETIC_SEQUENCES, SyntheticSequenceDataset, collate_fn
from models import CustomContinuousRNN
from train_utils import FIGURE_DIR, MODEL_DIR, ensure_output_dirs, load_checkpoint


CHECKPOINT = MODEL_DIR / "synthetic_latest.pt"
SAMPLES_PER_CLASS = 250
BATCH_SIZE = 64
N_COMPONENTS = 10


def collect_rates(model, loader, device):
    all_rates = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            _, rates, _ = model(x, return_states=True)
            all_rates.append(rates.cpu().numpy().reshape(-1, rates.shape[-1]))
    return np.concatenate(all_rates, axis=0)


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
    rates = collect_rates(model, loader, device)

    n_components = min(N_COMPONENTS, rates.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(rates)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    out_csv = FIGURE_DIR / "pca_explained_variance.csv"
    np.savetxt(
        out_csv,
        np.column_stack([np.arange(1, n_components + 1), explained, cumulative]),
        delimiter=",",
        header="component,explained_variance_ratio,cumulative_explained_variance",
        comments="",
    )

    plt.figure(figsize=(7, 4))
    plt.bar(np.arange(1, n_components + 1), explained, alpha=0.55, label="Single PC")
    plt.plot(
        np.arange(1, n_components + 1),
        cumulative,
        marker="o",
        color="tab:red",
        label="Cumulative",
    )
    plt.xlabel("PC")
    plt.ylabel("Explained variance ratio")
    plt.title("RNN Activity PCA Explained Variance")
    plt.ylim(0, 1.05)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_fig = FIGURE_DIR / "pca_explained_variance.png"
    plt.savefig(out_fig, dpi=200)

    print(f"Top 3 cumulative explained variance: {cumulative[min(2, len(cumulative)-1)]:.4f}")
    print(f"Saved PCA table: {out_csv}")
    print(f"Saved PCA figure: {out_fig}")


if __name__ == "__main__":
    main()
