import argparse
import glob
import os
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

from load_sec3_data import SEC3_FEATURE_FILE, SEC3_PHONEMES

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_TIMIT_DIR = REPO_ROOT / "data" / "TIMIT" / "TRAIN"


def extract_phoneme_mfcc_mean(timit_root, phoneme_list=SEC3_PHONEMES, n_mfcc=16):
    timit_root = Path(timit_root)
    phoneme_mfccs = {phoneme: [] for phoneme in phoneme_list}
    wav_files = glob.glob(str(timit_root / "**" / "*.WAV"), recursive=True)
    if not wav_files:
        raise FileNotFoundError(f"No .WAV files found under {timit_root}")

    for wav_path in tqdm(wav_files, desc="Extracting TIMIT MFCC", leave=False):
        phn_path = wav_path.replace(".WAV", ".PHN")
        if not os.path.exists(phn_path):
            continue

        y, sr = librosa.load(wav_path, sr=None)
        with open(phn_path, "r", encoding="utf-8") as f:
            for line in f:
                start, end, phoneme = line.strip().split()
                if phoneme not in phoneme_mfccs:
                    continue

                segment = y[int(start) : int(end)]
                if len(segment) < 10:
                    continue

                mfcc = librosa.feature.mfcc(
                    y=segment,
                    sr=sr,
                    n_mfcc=n_mfcc,
                    n_fft=min(2048, len(segment)),
                    n_mels=20,
                )
                phoneme_mfccs[phoneme].append(mfcc.mean(axis=1).astype(np.float32))

    phoneme_means = {}
    for phoneme in phoneme_list:
        if not phoneme_mfccs[phoneme]:
            raise ValueError(f"No samples found for phoneme {phoneme!r}")
        phoneme_means[phoneme] = np.stack(phoneme_mfccs[phoneme], axis=0).mean(axis=0)
    return phoneme_means


def save_sec3_timit_features(timit_root, output_path=SEC3_FEATURE_FILE, n_mfcc=16):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    phoneme_means = extract_phoneme_mfcc_mean(timit_root, n_mfcc=n_mfcc)
    np.savez_compressed(output_path, **phoneme_means)
    print(f"Saved Sec.3 TIMIT features: {output_path}")
    for phoneme, value in phoneme_means.items():
        print(f"  {phoneme}: shape={value.shape}, dtype={value.dtype}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the small Sec.3 TIMIT phoneme feature bundle."
    )
    parser.add_argument("--timit-dir", default=str(DEFAULT_TIMIT_DIR))
    parser.add_argument("--output", default=str(SEC3_FEATURE_FILE))
    parser.add_argument("--n-mfcc", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    save_sec3_timit_features(args.timit_dir, args.output, args.n_mfcc)


if __name__ == "__main__":
    main()
