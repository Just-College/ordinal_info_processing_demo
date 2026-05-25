from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
SEC3_FEATURE_FILE = REPO_ROOT / "resources" / "timit_phoneme_mfcc_means_mfcc16.npz"
SEC3_PHONEMES = ("pcl", "tcl", "pau")


def load_sec3_phoneme_mfcc_means(path=SEC3_FEATURE_FILE, phoneme_list=SEC3_PHONEMES):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing packaged Sec.3 TIMIT features: {path}. "
            "Run src/build_sec3_timit_features.py once on a machine with TIMIT, "
            "or include this .npz file in the release package."
        )

    data = np.load(path)
    missing = [phoneme for phoneme in phoneme_list if phoneme not in data]
    if missing:
        raise ValueError(f"{path} missing phoneme entries: {missing}")

    return {phoneme: data[phoneme].astype(np.float32) for phoneme in phoneme_list}


def build_sec3_timit_sequences(path=SEC3_FEATURE_FILE):
    phoneme_means = load_sec3_phoneme_mfcc_means(path)
    pcl = phoneme_means["pcl"]
    tcl = phoneme_means["tcl"]
    pau = phoneme_means["pau"]
    return [(pcl, tcl), (pcl, pau), (tcl, pcl), (tcl, pau)]
