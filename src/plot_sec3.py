import csv
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).parent.parent
FIGURE_DIR = REPO_ROOT / "figure"

CSV_FILE = REPO_ROOT / "resources" / "figure3b_transfer_history_seed1107.csv"
OUTPUT_FILE = FIGURE_DIR / "figure3B_transfer_test_accuracy.png"
ENABLE_PLOT_SHOW = True

MODE_LABELS = {
    "scratch_all": "w/o transfer",
    "sec2_pretrained_freeze_wrec": "with transfer",
}

MODE_COLORS = {"scratch_all": "#1f77b4", "sec2_pretrained_freeze_wrec": "#ff7f0e"}


def load_history():
    csv_path = Path(CSV_FILE)
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing history CSV: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required = {"mode", "epoch", "test_acc"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")
    return csv_path, rows


def plot_figure3b():
    csv_path, rows = load_history()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5.2, 3.6))
    for mode in MODE_LABELS:
        mode_rows = sorted(
            [row for row in rows if row["mode"] == mode],
            key=lambda row: int(row["epoch"]),
        )
        if not mode_rows:
            print(f"Warning: mode {mode!r} not found in {csv_path}")
            continue
        plt.plot(
            [int(row["epoch"]) for row in mode_rows],
            [float(row["test_acc"]) for row in mode_rows],
            label=MODE_LABELS[mode],
            color=MODE_COLORS[mode],
            linewidth=2.2,
        )

    ax = plt.gca()
    ax.spines[["top", "right"]].set_visible(False)

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Transfer Learning")
    plt.ylim(0, 105)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"Loaded history: {csv_path}")
    print(f"Saved figure: {OUTPUT_FILE}")
    if ENABLE_PLOT_SHOW:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    plot_figure3b()
