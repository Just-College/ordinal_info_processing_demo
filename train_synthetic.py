import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import (
    SYNTHETIC_SEQUENCE_NAMES,
    SYNTHETIC_SEQUENCES,
    SyntheticSequenceDataset,
    collate_fn,
)
from models import CustomContinuousRNN
from train_utils import set_seed, train_model


SAMPLES_PER_CLASS = 256
TEST_SAMPLES_PER_CLASS = 216
EPOCHS = 100
BATCH_SIZE = 16
LR = 1e-3
HIDDEN_DIM = 50
TAU = 2.0
T_MAX = 30
PHONEME_LEN = 3
NOISE_STD = 0.01
SEED = 1


def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = SyntheticSequenceDataset(
        SYNTHETIC_SEQUENCES,
        SAMPLES_PER_CLASS,
        T_MAX,
        NOISE_STD,
        phoneme_len=PHONEME_LEN,
        fixed=False,
    )
    test_dataset = SyntheticSequenceDataset(
        SYNTHETIC_SEQUENCES,
        TEST_SAMPLES_PER_CLASS,
        T_MAX,
        NOISE_STD,
        phoneme_len=PHONEME_LEN,
        fixed=True,
    )
    loader_train = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    loader_test = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    model = CustomContinuousRNN(
        input_dim=3,
        hidden_dim=HIDDEN_DIM,
        output_dim=len(SYNTHETIC_SEQUENCES),
        tau=TAU,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    loss_fn = nn.MSELoss()

    metadata = {
        "task": "synthetic",
        "sequence_names": SYNTHETIC_SEQUENCE_NAMES,
        "input_dim": 3,
        "hidden_dim": HIDDEN_DIM,
        "output_dim": len(SYNTHETIC_SEQUENCES),
        "tau": TAU,
        "t_max": T_MAX,
        "phoneme_len": PHONEME_LEN,
        "noise_std": NOISE_STD,
        "seed": SEED,
    }
    _, checkpoint_path, latest_path, csv_path = train_model(
        model,
        loader_train,
        loader_test,
        optimizer,
        scheduler,
        loss_fn,
        device,
        EPOCHS,
        T_MAX,
        PHONEME_LEN,
        run_name="synthetic",
        checkpoint_metadata=metadata,
    )
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Updated latest checkpoint: {latest_path}")
    print(f"Saved training history: {csv_path}")


if __name__ == "__main__":
    main()
