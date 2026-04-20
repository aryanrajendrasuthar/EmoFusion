"""
predict.py
----------
Inference module for EmotionFusion.

Feature pipeline mirrors extract_features.py exactly (paper-exact):
    ZCR (108) + RMS (108) + MFCC_flat (2160) + HuBERT_sentiment (4) = 2380-D

Usage:
    from predict import predict_emotion
    label, confidence, all_probs = predict_emotion("audio.wav")

CLI:
    python predict.py path/to/audio.wav
"""

import sys
import os
import pickle
import warnings
import numpy as np
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import librosa
import torch
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
import tensorflow as tf

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
MODEL_PATH   = BASE_DIR / "models" / "emotionFusion_model.h5"
SCALER_PATH  = BASE_DIR / "models" / "scaler.pkl"
ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"

# ── Audio / feature constants (must match extract_features.py) ─────────────
SR_LIBROSA     = 22050
SR_HUBERT      = 16000
AUDIO_DURATION = 2.5
TARGET_SAMPLES = int(AUDIO_DURATION * SR_LIBROSA)   # 55125
N_FRAMES       = 108
HOP_LENGTH     = 512
N_MFCC         = 20
FEATURE_DIM    = 2380   # 108 + 108 + 2160 + 4

# ── Module-level singletons (loaded once on first call) ────────────────────
_model              = None
_scaler             = None
_encoder            = None
_hubert_extractor   = None
_hubert_model       = None
_hubert_device      = None
_hubert_num_labels  = None


def _load_model_artifacts():
    global _model, _scaler, _encoder
    for p in [MODEL_PATH, SCALER_PATH, ENCODER_PATH]:
        if not p.exists():
            raise FileNotFoundError(
                f"Required artefact not found: {p}\n"
                "Please run train.py before using predict.py."
            )
    if _model is None:
        _model = tf.keras.models.load_model(str(MODEL_PATH))
    if _scaler is None:
        with open(SCALER_PATH, "rb") as f:
            _scaler = pickle.load(f)
    if _encoder is None:
        with open(ENCODER_PATH, "rb") as f:
            _encoder = pickle.load(f)


def _load_hubert():
    global _hubert_extractor, _hubert_model, _hubert_device, _hubert_num_labels
    if _hubert_model is None:
        model_name          = "superb/hubert-base-superb-er"
        _hubert_device      = "cuda" if torch.cuda.is_available() else "cpu"
        _hubert_extractor   = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        _hubert_model       = HubertForSequenceClassification.from_pretrained(
            model_name
        ).to(_hubert_device)
        _hubert_model.eval()
        _hubert_num_labels  = _hubert_model.config.num_labels   # 4


# ══════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION  (mirrors extract_features.py — kept in sync)
# ══════════════════════════════════════════════════════════════════════════

def _pad_or_trim(signal: np.ndarray) -> np.ndarray:
    if len(signal) > TARGET_SAMPLES:
        return signal[:TARGET_SAMPLES]
    return np.pad(signal, (0, TARGET_SAMPLES - len(signal)))


def _extract_acoustic(signal: np.ndarray, sr: int) -> np.ndarray:
    """Frame-level ZCR(108) + RMS(108) + MFCC_flat(2160) = (2376,)."""
    signal = _pad_or_trim(signal)

    zcr  = librosa.feature.zero_crossing_rate(
        signal, hop_length=HOP_LENGTH
    ).flatten()[:N_FRAMES]

    rmse = librosa.feature.rms(
        y=signal, hop_length=HOP_LENGTH
    ).flatten()[:N_FRAMES]

    mfcc = librosa.feature.mfcc(
        y=signal, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH
    )[:, :N_FRAMES].flatten()

    return np.concatenate([zcr, rmse, mfcc]).astype(np.float32)


@torch.no_grad()
def _extract_hubert_sentiment(signal: np.ndarray, sr: int) -> np.ndarray:
    """HuBERT 4-class sentiment classifier → one-hot (4,)."""
    _load_hubert()
    if sr != SR_HUBERT:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=SR_HUBERT)

    inputs  = _hubert_extractor(
        signal, sampling_rate=SR_HUBERT, return_tensors="pt", padding=True
    )
    logits  = _hubert_model(inputs.input_values.to(_hubert_device)).logits
    pred_id = int(logits.argmax(dim=-1).item())

    one_hot           = np.zeros(_hubert_num_labels, dtype=np.float32)
    one_hot[pred_id]  = 1.0
    return one_hot


def _build_feature_vector(signal: np.ndarray, sr: int) -> np.ndarray:
    """Concatenate acoustic (2376,) + sentiment (4,) → (2380,)."""
    acoustic  = _extract_acoustic(signal, sr)
    sentiment = _extract_hubert_sentiment(signal, sr)
    return np.concatenate([acoustic, sentiment])


# ══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════

def predict_emotion(audio_path: str) -> tuple:
    """
    Predict the emotion in a speech audio file.

    Parameters
    ----------
    audio_path : str
        Path to any audio file readable by librosa (.wav, .mp3, etc.)

    Returns
    -------
    emotion    : str   — one of {anger, disgust, fear, happy, neutral, sad, surprise}
    confidence : float — probability of predicted class (0–1)
    all_probs  : dict  — {emotion: probability} for all 7 classes
    """
    _load_model_artifacts()
    _load_hubert()

    signal, sr = librosa.load(str(audio_path), sr=SR_LIBROSA, mono=True)
    features   = _build_feature_vector(signal, SR_LIBROSA)          # (2380,)

    features_scaled = _scaler.transform(features.reshape(1, -1))    # (1, 2380)
    features_input  = features_scaled.reshape(1, FEATURE_DIM, 1)    # (1, 2380, 1)

    proba          = _model.predict(features_input, verbose=0)[0]   # (7,)
    predicted_idx  = int(np.argmax(proba))
    predicted_label = _encoder.classes_[predicted_idx]
    confidence      = float(proba[predicted_idx])
    all_probs       = {cls: float(p) for cls, p in zip(_encoder.classes_, proba)}

    return predicted_label, confidence, all_probs


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_audio.wav>")
        sys.exit(1)

    audio_file = sys.argv[1]
    if not Path(audio_file).exists():
        print(f"[ERROR] File not found: {audio_file}")
        sys.exit(1)

    print(f"\nAnalysing: {audio_file}")
    print("-" * 50)
    emotion, confidence, all_probs = predict_emotion(audio_file)
    print(f"Predicted Emotion : {emotion.upper()}")
    print(f"Confidence        : {confidence * 100:.2f}%")
    print("\nAll class probabilities:")
    for label, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {label:<10} {prob * 100:5.2f}%  {bar}")
