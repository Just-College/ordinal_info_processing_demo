import random

import torch
from torch.utils.data import Dataset


class SyntheticSequenceDataset(Dataset):
    def __init__(self, sequences, n_samples_per_class, T_max, noise_std, phoneme_len=3, fixed=False):
        self.samples = []
        self.labels = []
        self.sequences = sequences
        self.phoneme_len = phoneme_len
        for idx, seq in enumerate(sequences):
            for _ in range(n_samples_per_class):
                x, lengths, T_noise = self.generate_synthetic_sequence(seq, T_max, noise_std, phoneme_len, fixed)
                self.samples.append(x)
                self.labels.append((idx, lengths, T_noise))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        class_id, lengths, T_noise = self.labels[idx]
        y = generate_ramping_target(lengths, class_id, len(self.sequences), T_noise)
        return x, y

    @staticmethod
    def generate_synthetic_sequence(seq, T_max, noise_std, phoneme_len=3, fixed=False):
        seg1 = torch.tensor(seq[0], dtype=torch.float32).unsqueeze(0).repeat(phoneme_len, 1)
        seg2 = torch.tensor(seq[1], dtype=torch.float32).unsqueeze(0).repeat(phoneme_len, 1)
        T_noise = T_max if fixed else random.randint(0, T_max)
        total_len = T_max + phoneme_len + T_noise + phoneme_len + T_max
        x = torch.randn(total_len, seg1.shape[1]) * noise_std
        x[T_max : T_max + phoneme_len, :] = seg1
        x[T_max + phoneme_len + T_noise : T_max + phoneme_len + T_noise + phoneme_len, :] = seg2
        lengths = [phoneme_len, phoneme_len]
        return x, lengths, T_noise

def generate_ramping_target(lengths, class_id, num_classes, T_noise, T_MAX=30):
    O_init = 0.25
    O_1 = 0.5
    O_2 = 1.0
    O_sile = 0.0
    phoneme_len = lengths[0]
    total_len = T_MAX + phoneme_len + T_noise + phoneme_len + T_MAX
    target = torch.full((total_len, num_classes), O_init)
    idx1_start = T_MAX
    idx1_end = idx1_start + phoneme_len + T_noise
    idx2_start = idx1_end
    idx2_end = idx2_start + phoneme_len + T_MAX
    if class_id in [0, 1]:
        target[idx1_start:idx1_end, 0] = O_1
        target[idx1_start:idx1_end, 1] = O_1
        target[idx1_start:idx1_end, 2:] = O_sile
    else:
        target[idx1_start:idx1_end, 2] = O_1
        target[idx1_start:idx1_end, 3] = O_1
        target[idx1_start:idx1_end, :2] = O_sile
    target[idx2_start:idx2_end, :] = O_sile
    target[idx2_start:idx2_end, class_id] = O_2
    return target

def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    xs_padded = torch.stack([torch.nn.functional.pad(x, (0, 0, 0, max_len - x.shape[0])) for x in xs])
    ys_padded = torch.stack([torch.nn.functional.pad(y, (0, 0, 0, max_len - y.shape[0])) for y in ys])
    return xs_padded, ys_padded
