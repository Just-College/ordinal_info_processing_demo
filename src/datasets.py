import random

import torch
from torch.utils.data import Dataset


def generate_sequence(seq, t_max, noise_std, phoneme_len=3, fixed=False):
    seg1 = torch.tensor(seq[0], dtype=torch.float32).unsqueeze(0).repeat(phoneme_len, 1)
    seg2 = torch.tensor(seq[1], dtype=torch.float32).unsqueeze(0).repeat(phoneme_len, 1)
    t_noise = t_max if fixed else random.randint(0, t_max)
    total_len = t_max + phoneme_len + t_noise + phoneme_len + t_max
    x = torch.randn(total_len, seg1.shape[1]) * noise_std
    x[t_max : t_max + phoneme_len, :] = seg1
    second_start = t_max + phoneme_len + t_noise
    x[second_start : second_start + phoneme_len, :] = seg2
    return x, [phoneme_len, phoneme_len], t_noise


def generate_ramping_target(lengths, class_id, num_classes, t_noise, t_max):
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


class SequenceDataset(Dataset):
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
        self.phoneme_len = phoneme_len
        for class_id, seq in enumerate(sequences):
            for _ in range(n_samples_per_class):
                x, lengths, t_noise = generate_sequence(
                    seq, t_max, noise_std, phoneme_len=phoneme_len, fixed=fixed
                )
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
            self.t_max,
        )
        return x, y, class_id, t_noise


class SyntheticSequenceDataset(SequenceDataset):
    def __init__(
        self,
        sequences,
        n_samples_per_class,
        T_max,
        noise_std,
        phoneme_len=3,
        fixed=False,
    ):
        super().__init__(
            sequences,
            n_samples_per_class,
            T_max,
            noise_std,
            phoneme_len=phoneme_len,
            fixed=fixed,
        )


class TimitSequenceDataset(SequenceDataset):
    def __init__(
        self,
        sequences,
        n_samples_per_class,
        t_max,
        noise_std,
        phoneme_len=3,
        fixed=False,
    ):
        super().__init__(
            sequences,
            n_samples_per_class,
            t_max,
            noise_std,
            phoneme_len=phoneme_len,
            fixed=fixed,
        )


def collate_fn(batch):
    xs, ys, class_ids, t_noises = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    xs_padded = torch.stack(
        [torch.nn.functional.pad(x, (0, 0, 0, max_len - x.shape[0])) for x in xs]
    )
    ys_padded = torch.stack(
        [torch.nn.functional.pad(y, (0, 0, 0, max_len - y.shape[0])) for y in ys]
    )
    mask = torch.zeros(len(xs), max_len, dtype=torch.bool)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    return (
        xs_padded,
        ys_padded,
        mask,
        torch.tensor(class_ids, dtype=torch.long),
        torch.tensor(t_noises, dtype=torch.long),
        lengths,
    )
