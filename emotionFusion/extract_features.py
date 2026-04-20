"""
extract_features.py
-------------------
Dataset loading, label mapping, data augmentation, and feature extraction
for EmotionFusion — implemented exactly as described in the paper:

  Vankalas & Dhingra, "EmotionFusion: A Deep Learning Approach to
  Speech Emotion Recognition", Savitribai Phule Pune University.

Feature vector layout (2380-D):
  [ ZCR (108) | RMS (108) | MFCC_flat (2160) | HuBERT_sentiment (4) ]

Key design decisions taken directly from the paper:
  - Audio padded / trimmed to 2.5 s → 55125 samples @ 22050 Hz
    → exactly 108 frames with hop_length=512, center=True
  - ZCR and RMSE kept frame-level (NOT mean-pooled)
  - 20 MFCCs per frame, flattened across all 108 frames → 2160
  - HuBERT used as a 4-class SENTIMENT CLASSIFIER (neu/hap/ang/sad),
    output one-hot encoded → (4,)   ← critical difference from v1
"""

import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
FEATURES_DIR  = BASE_DIR / "features"
FEATURES_FILE = FEATURES_DIR / "features.pkl"

# ── Sampling rates ─────────────────────────────────────────────────────────
SR_LIBROSA = 22050    # acoustic feature extraction
SR_HUBERT  = 16000    # HuBERT expects 16 kHz

# ── Fixed audio length (paper: gives exactly 108 frames) ──────────────────
# 2.5 s × 22050 Hz = 55125 samples
# ceil(55125 / 512) = 108 frames  ✓
AUDIO_DURATION = 2.5
TARGET_SAMPLES = int(AUDIO_DURATION * SR_LIBROSA)   # 55125
N_FRAMES       = 108    # expected frame count after pad/trim
HOP_LENGTH     = 512
N_MFCC         = 20

# Feature dimensions — must match the paper diagram
DIM_ZCR        = N_FRAMES            # 108
DIM_RMS        = N_FRAMES            # 108
DIM_MFCC       = N_MFCC * N_FRAMES   # 2160
DIM_SENTIMENT  = 4                   # HuBERT 4-class one-hot
FEATURE_DIM    = DIM_ZCR + DIM_RMS + DIM_MFCC + DIM_SENTIMENT  # 2380

# ── Unified emotion label set ──────────────────────────────────────────────
EMOTION_LABELS = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# ── Dataset-specific label maps ────────────────────────────────────────────
CREMA_MAP = {
    "ANG": "anger", "DIS": "disgust", "FEA": "fear",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad",
}

RAVDESS_MAP = {
    "01": "neutral", "02": "neutral",   # calm → neutral
    "03": "happy",   "04": "sad",
    "05": "anger",   "06": "fear",
    "07": "disgust", "08": "surprise",
}

TESS_MAP = {
    "angry": "anger",  "disgust": "disgust", "fear": "fear",
    "happy": "happy",  "neutral": "neutral", "sad": "sad",
    "ps":    "surprise",   # pleasant surprise → surprise
}

SAVEE_MAP = {
    "a": "anger", "d": "disgust", "f": "fear",
    "h": "happy", "n": "neutral", "sa": "sad", "su": "surprise",
}

HINDI_MAP = {
    "angry": "anger",   "disgust": "disgust", "fear": "fear",
    "happy": "happy",   "neutral": "neutral", "sad": "sad",
    "surprise": "surprise", "sarcasm": "disgust",
}


# ══════════════════════════════════════════════════════════════════════════
# 1.  DATASET LOADERS
# ══════════════════════════════════════════════════════════════════════════

def load_cremad(root: Path) -> list:
    samples = []
    path = root / "CREMA-D" / "AudioWAV"
    if not path.exists():
        print(f"[WARN] CREMA-D not found at {path}"); return samples
    for wav in path.glob("*.wav"):
        parts = wav.stem.split("_")
        emotion = CREMA_MAP.get(parts[2]) if len(parts) >= 3 else None
        if emotion:
            samples.append((str(wav), emotion))
    print(f"[INFO] CREMA-D: {len(samples)} samples"); return samples


def load_ravdess(root: Path) -> list:
    samples = []
    path = root / "RAVDESS"
    if not path.exists():
        print(f"[WARN] RAVDESS not found at {path}"); return samples
    for wav in path.rglob("*.wav"):
        parts = wav.stem.split("-")
        emotion = RAVDESS_MAP.get(parts[2]) if len(parts) >= 3 else None
        if emotion:
            samples.append((str(wav), emotion))
    print(f"[INFO] RAVDESS: {len(samples)} samples"); return samples


def load_tess(root: Path) -> list:
    samples = []
    path = root / "TESS"
    if not path.exists():
        print(f"[WARN] TESS not found at {path}"); return samples
    for wav in path.rglob("*.wav"):
        folder_emotion = wav.parent.name.lower().split("_")[-1]
        emotion = TESS_MAP.get(folder_emotion)
        if emotion:
            samples.append((str(wav), emotion))
    print(f"[INFO] TESS: {len(samples)} samples"); return samples


def load_savee(root: Path) -> list:
    samples = []
    path = root / "SAVEE"
    if not path.exists():
        print(f"[WARN] SAVEE not found at {path}"); return samples
    for wav in path.rglob("*.wav"):
        stem = wav.stem.split("_")[-1]
        prefix = re.match(r"^([a-z]{1,2})", stem)
        if not prefix:
            continue
        code    = prefix.group(1)
        emotion = SAVEE_MAP.get(code, SAVEE_MAP.get(code[0]))
        if emotion:
            samples.append((str(wav), emotion))
    print(f"[INFO] SAVEE: {len(samples)} samples"); return samples


def load_hindi(root: Path) -> list:
    samples = []
    path = root / "Hindi"
    if not path.exists():
        print(f"[WARN] Hindi SER not found at {path}"); return samples
    for wav in path.rglob("*.wav"):
        emotion = HINDI_MAP.get(wav.parent.name.lower())
        if emotion:
            samples.append((str(wav), emotion))
    print(f"[INFO] Hindi SER: {len(samples)} samples"); return samples


def load_all_datasets() -> pd.DataFrame:
    all_samples = []
    all_samples.extend(load_cremad(DATA_DIR))
    all_samples.extend(load_ravdess(DATA_DIR))
    all_samples.extend(load_tess(DATA_DIR))
    all_samples.extend(load_savee(DATA_DIR))
    all_samples.extend(load_hindi(DATA_DIR))
    df = pd.DataFrame(all_samples, columns=["path", "emotion"])
    print(f"\n[INFO] Total original samples: {len(df)}")
    print(df["emotion"].value_counts())
    return df


# ══════════════════════════════════════════════════════════════════════════
# 2.  DATA AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════

def add_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    signal_power = np.mean(signal ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    return signal + np.random.normal(0, np.sqrt(noise_power), len(signal))


def pitch_shift(signal: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    return librosa.effects.pitch_shift(signal, sr=sr, n_steps=semitones)


def augment_sample(signal: np.ndarray, sr: int) -> list:
    """Return 3 augmented variants: noise, pitch, noise+pitch."""
    snr   = np.random.uniform(10, 20)
    steps = np.random.uniform(-3, 3)
    return [
        add_noise(signal, snr),
        pitch_shift(signal, sr, steps),
        pitch_shift(add_noise(signal, snr), sr, steps),
    ]


# ══════════════════════════════════════════════════════════════════════════
# 3.  FEATURE EXTRACTION  (paper-exact)
# ══════════════════════════════════════════════════════════════════════════

def pad_or_trim(signal: np.ndarray, target: int = TARGET_SAMPLES) -> np.ndarray:
    """
    Trim or zero-pad signal to exactly `target` samples.
    This guarantees a fixed frame count for all downstream features.
    """
    if len(signal) > target:
        return signal[:target]
    return np.pad(signal, (0, target - len(signal)))


def extract_acoustic_features(signal: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract frame-level acoustic features exactly as described in the paper.

    Returns a 1-D vector of shape (2376,):
        - ZCR  : (108,)  — frame-level, NOT mean-pooled
        - RMSE : (108,)  — frame-level, NOT mean-pooled
        - MFCC : (2160,) — 20 coefficients × 108 frames, flattened
    """
    # Pad/trim to fixed length so every sample yields exactly 108 frames
    signal = pad_or_trim(signal, TARGET_SAMPLES)

    # Zero Crossing Rate — shape (1, 108) → flatten to (108,)
    zcr  = librosa.feature.zero_crossing_rate(
        signal, hop_length=HOP_LENGTH
    ).flatten()[:N_FRAMES]

    # Root Mean Square Energy — shape (1, 108) → flatten to (108,)
    rmse = librosa.feature.rms(
        y=signal, hop_length=HOP_LENGTH
    ).flatten()[:N_FRAMES]

    # MFCCs — shape (20, 108) → flatten to (2160,)
    mfcc = librosa.feature.mfcc(
        y=signal, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH
    )[:, :N_FRAMES].flatten()

    # Concatenate → (2376,)
    return np.concatenate([zcr, rmse, mfcc]).astype(np.float32)


class HuBERTSentimentExtractor:
    """
    Uses superb/hubert-base-superb-er as a 4-CLASS CLASSIFIER.

    The paper feeds audio through HuBERT, takes the predicted sentiment
    class (one of: neu, hap, ang, sad), and one-hot encodes it → (4,).
    This is NOT a feature embedding — it is a discrete classification output.
    """

    def __init__(self, model_name: str = "superb/hubert-base-superb-er"):
        print(f"[INFO] Loading HuBERT classifier: {model_name} …")
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model     = HubertForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device)
        self.model.eval()
        self.num_labels = self.model.config.num_labels  # 4
        print(f"[INFO] HuBERT classifier ready on {self.device} "
              f"({self.num_labels} sentiment classes)")

    @torch.no_grad()
    def extract(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """
        Resample to 16 kHz → run HuBERT classifier → one-hot encode.
        Returns a (4,) float32 vector, e.g. [0, 0, 1, 0] for neutral.
        """
        if sr != SR_HUBERT:
            signal = librosa.resample(signal, orig_sr=sr, target_sr=SR_HUBERT)

        inputs = self.extractor(
            signal, sampling_rate=SR_HUBERT, return_tensors="pt", padding=True
        )
        logits    = self.model(
            inputs.input_values.to(self.device)
        ).logits                                   # (1, 4)
        pred_id   = int(logits.argmax(dim=-1).item())

        one_hot   = np.zeros(self.num_labels, dtype=np.float32)
        one_hot[pred_id] = 1.0
        return one_hot                             # (4,)


def extract_features_for_signal(
    signal: np.ndarray,
    sr: int,
    hubert: HuBERTSentimentExtractor,
) -> np.ndarray:
    """
    Build the complete 2380-D feature vector for one audio signal.

        acoustic  (2376,)  =  ZCR(108) + RMS(108) + MFCC_flat(2160)
        sentiment    (4,)  =  HuBERT 4-class one-hot
        fused       (2380,)
    """
    acoustic  = extract_acoustic_features(signal, sr)   # (2376,)
    sentiment = hubert.extract(signal, sr)               # (4,)
    return np.concatenate([acoustic, sentiment])         # (2380,)


# ══════════════════════════════════════════════════════════════════════════
# 4.  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def build_feature_dataset(df: pd.DataFrame) -> tuple:
    """
    For every sample: load audio, create 3 augmented variants,
    extract 2380-D feature vector for each (original + 3 augmented).

    Returns:
        X : np.ndarray  shape (N*4, 2380)
        y : list of str emotion labels
    """
    hubert  = HuBERTSentimentExtractor()
    X_list  = []
    y_list  = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        try:
            signal, sr = librosa.load(row["path"], sr=SR_LIBROSA, mono=True)
        except Exception as e:
            print(f"[WARN] Skipping {row['path']}: {e}")
            continue

        emotion = row["emotion"]

        # Original
        X_list.append(extract_features_for_signal(signal, SR_LIBROSA, hubert))
        y_list.append(emotion)

        # 3 augmented variants
        for aug in augment_sample(signal, SR_LIBROSA):
            X_list.append(extract_features_for_signal(aug, SR_LIBROSA, hubert))
            y_list.append(emotion)

    X = np.array(X_list, dtype=np.float32)
    print(f"\n[INFO] Feature matrix: {X.shape}  (expected dim={FEATURE_DIM})")
    assert X.shape[1] == FEATURE_DIM, \
        f"Feature dim mismatch: got {X.shape[1]}, expected {FEATURE_DIM}"

    unique, counts = np.unique(y_list, return_counts=True)
    for lbl, cnt in zip(unique, counts):
        print(f"  {lbl}: {cnt}")
    return X, y_list


def save_features(X: np.ndarray, y: list) -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    with open(FEATURES_FILE, "wb") as f:
        pickle.dump({"X": X, "y": y}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[INFO] Features saved → {FEATURES_FILE}")


def load_features() -> tuple:
    with open(FEATURES_FILE, "rb") as f:
        data = pickle.load(f)
    print(f"[INFO] Features loaded: shape={data['X'].shape}")
    return data["X"], data["y"]


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_all_datasets()
    if df.empty:
        print("[ERROR] No data found. Download datasets first (see README.md).")
        raise SystemExit(1)

    X, y = build_feature_dataset(df)
    save_features(X, y)
    print("\n[DONE] Feature extraction complete → run `python train.py` next.")
