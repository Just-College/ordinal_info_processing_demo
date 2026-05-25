import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CSV_DIR = f"{Path(__file__).parent.parent}/csv/"
PARTIAL = True  # 是否为部分利用模型
SMOOTH = True  # 是否开启滑动平均
SMOOTH_WINDOW = 5  # 平滑窗口大小

if PARTIAL:
    csv_files = {
        "pretrained: False, freeze: False": os.path.join(
            CSV_DIR,
            "train_history_newtest_rnn_mfcc16_hid50_spc512_pretrainedFalse_freezeFalse_20250715_162724.csv",
        ),
        "pretrained: False, freeze: True": os.path.join(
            CSV_DIR,
            "train_history_newtest_rnn_mfcc16_hid50_spc512_pretrainedFalse_freezeTrue_20250715_163423.csv",
        ),
        "pretrained: True, freeze: True": os.path.join(
            CSV_DIR,
            "train_history_newtest_rnn_mfcc16_hid50_spc512_pretrainedTrue_freezeTrue_20250715_161615.csv",
        ),
        "pretrained: True, freeze: False": os.path.join(
            CSV_DIR,
            "train_history_newtest_rnn_mfcc16_hid50_spc512_pretrainedTrue_freezeFalse_20250716_095537.csv",
        ),
    }
else:
    csv_files = {
        "pretrained: False, freeze: False": os.path.join(
            CSV_DIR, "train_history_pretrainedFalse_freezeFalse_20250714_214052.csv"
        ),
        "pretrained: False, freeze: True": os.path.join(
            CSV_DIR, "train_history_pretrainedFalse_freezeTrue_20250714_213056.csv"
        ),
        "pretrained: True, freeze: True": os.path.join(
            CSV_DIR, "train_history_pretrainedTrue_freezeTrue_20250714_212048.csv"
        ),
        "pretrained: True, freeze: False": os.path.join(
            CSV_DIR, "train_history_pretrainedTrue_freezeFalse_20250716_093811.csv"
        ),
    }

plt.figure(figsize=(10, 6))
for label, file_path in csv_files.items():
    if not os.path.exists(file_path):
        print(f"警告：{file_path} 不存在，跳过。")
        continue
    df = pd.read_csv(file_path)
    # 假定csv有 'epoch' 和 'test_acc' 两列
    if "epoch" not in df.columns or "test_acc" not in df.columns:
        print(f"警告：{file_path} 缺少必要列，跳过。")
        continue
    if SMOOTH:
        acc_smooth = (
            df["test_acc"]
            .rolling(window=SMOOTH_WINDOW, min_periods=1, center=True)
            .mean()
        )
        plt.plot(df["epoch"], acc_smooth, label=label + " (smooth)", linewidth=2)
    else:
        plt.plot(df["epoch"], df["test_acc"], label=label, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy (%)")
plt.title("Test Accuracy Curve for Different Training Modes")
plt.legend()
plt.grid(True)
plt.tight_layout()

if PARTIAL:
    plt.savefig("test_acc_compare_modes_partial.png", dpi=200)
else:
    plt.savefig("test_acc_compare_modes_complete.png", dpi=200)

plt.show()
