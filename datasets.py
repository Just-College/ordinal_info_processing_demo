import random

import numpy as np
import torch
from torch.utils.data import Dataset


SYNTHETIC_PRIMITIVES = {
    "a": np.array([1.0, 1.0, 0.0], dtype=np.float32),
    "b": np.array([1.0, 0.0, 1.0], dtype=np.float32),
    "c": np.array([0.0, 1.0, 1.0], dtype=np.float32),
}

SYNTHETIC_SEQUENCE_NAMES = ["ab", "ac", "ba", "bc"]
SYNTHETIC_SEQUENCES = [
    (SYNTHETIC_PRIMITIVES["a"], SYNTHETIC_PRIMITIVES["b"]),
    (SYNTHETIC_PRIMITIVES["a"], SYNTHETIC_PRIMITIVES["c"]),
    (SYNTHETIC_PRIMITIVES["b"], SYNTHETIC_PRIMITIVES["a"]),
    (SYNTHETIC_PRIMITIVES["b"], SYNTHETIC_PRIMITIVES["c"]),
]


def make_transfer_primitives(input_dim=16, seed=7):
    rng = np.random.default_rng(seed)
    primitives = rng.normal(size=(3, input_dim)).astype(np.float32)
    primitives = primitives / np.linalg.norm(primitives, axis=1, keepdims=True)
    return {
        "p1": primitives[0],
        "p2": primitives[1],
        "p3": primitives[2],
    }


def make_transfer_sequences(input_dim=16, seed=7):
    primitives = make_transfer_primitives(input_dim=input_dim, seed=seed)
    return [
        (primitives["p1"], primitives["p2"]),
        (primitives["p1"], primitives["p3"]),
        (primitives["p2"], primitives["p1"]),
        (primitives["p2"], primitives["p3"]),
    ]


def generate_ramping_target(lengths, class_id, num_classes, t_noise, t_max=30):
    o_init = 0.25
    o_1 = 0.5
    o_2 = 1.0
    o_silent = 0.0
    phoneme_len = lengths[0]
    total_len = t_max + phoneme_len + t_noise + phoneme_len + t_max

    target = torch.full((total_len, num_classes), o_init)
    idx1_start = t_max
    idx1_end = idx1_start + phoneme_len + t_noise
    idx2_start = idx1_end
    idx2_end = idx2_start + phoneme_len + t_max

    if class_id in [0, 1]:
        target[idx1_start:idx1_end, 0] = o_1
        target[idx1_start:idx1_end, 1] = o_1
        target[idx1_start:idx1_end, 2:] = o_silent
    else:
        target[idx1_start:idx1_end, 2] = o_1
        target[idx1_start:idx1_end, 3] = o_1
        target[idx1_start:idx1_end, :2] = o_silent

    target[idx2_start:idx2_end, :] = o_silent
    target[idx2_start:idx2_end, class_id] = o_2
    return target


class SyntheticSequenceDataset(Dataset):
    def __init__(
        self,
        sequences,
        n_samples_per_class,
        t_max,
        noise_std,
        phoneme_len=3,
        fixed=False,
    ):
        self.samples = []
        self.labels = []
        self.sequences = sequences
        self.t_max = t_max
        self.noise_std = noise_std
        self.phoneme_len = phoneme_len

        for class_id, sequence in enumerate(sequences):
            for _ in range(n_samples_per_class):
                x, lengths, t_noise = self.generate_sequence(sequence, fixed=fixed)
                self.samples.append(x)
                self.labels.append((class_id, lengths, t_noise))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        class_id, lengths, t_noise = self.labels[idx]
        y = generate_ramping_target(
            lengths,
            class_id,
            len(self.sequences),
            t_noise,
            t_max=self.t_max,
        )
        return x, y

    def generate_sequence(self, sequence, fixed=False):
        seg1 = (
            torch.tensor(sequence[0], dtype=torch.float32)
            .unsqueeze(0)
            .repeat(self.phoneme_len, 1)
        )
        seg2 = (
            torch.tensor(sequence[1], dtype=torch.float32)
            .unsqueeze(0)
            .repeat(self.phoneme_len, 1)
        )
        t_noise = self.t_max if fixed else random.randint(0, self.t_max)
        total_len = self.t_max + self.phoneme_len + t_noise + self.phoneme_len + self.t_max

        x = torch.randn(total_len, seg1.shape[1]) * self.noise_std
        x[self.t_max : self.t_max + self.phoneme_len, :] = seg1
        second_start = self.t_max + self.phoneme_len + t_noise
        x[second_start : second_start + self.phoneme_len, :] = seg2
        return x, [self.phoneme_len, self.phoneme_len], t_noise


def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    xs_padded = torch.stack(
        [torch.nn.functional.pad(x, (0, 0, 0, max_len - x.shape[0])) for x in xs]
    )
    ys_padded = torch.stack(
        [torch.nn.functional.pad(y, (0, 0, 0, max_len - y.shape[0])) for y in ys]
    )
    return xs_padded, ys_padded
