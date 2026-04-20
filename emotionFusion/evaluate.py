"""
evaluate.py
-----------
Loads the trained model and held-out test data, then produces:
  - Classification report (precision / recall / F1 per class)
  - Confusion matrix heatmap  → outputs/confusion_matrix.png
  - Train & validation accuracy/loss curves → outputs/training_curves.png

Run after train.py has completed successfully.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
MODELS_DIR    = BASE_DIR / "models"
MODEL_PATH    = MODELS_DIR / "emotionFusion_model.h5"
ENCODER_PATH  = MODELS_DIR / "label_encoder.pkl"
HISTORY_PATH  = BASE_DIR / "features" / "training_history.pkl"
TEST_DATA_PATH = BASE_DIR / "features" / "test_data.pkl"
OUTPUTS_DIR   = BASE_DIR / "outputs"

EMOTION_LABELS = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# ══════════════════════════════════════════════════════════════════════════
# 1.  LOAD ARTEFACTS
# ══════════════════════════════════════════════════════════════════════════

def load_artefacts():
    """Load model, label encoder, training history, and test set."""
    for p in [MODEL_PATH, ENCODER_PATH, HISTORY_PATH, TEST_DATA_PATH]:
        if not p.exists():
            raise FileNotFoundError(
                f"Required file not found: {p}\n"
                "Make sure train.py ran successfully."
            )

    model = tf.keras.models.load_model(str(MODEL_PATH))
    print(f"[INFO] Model loaded from {MODEL_PATH}")

    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)

    with open(HISTORY_PATH, "rb") as f:
        history = pickle.load(f)

    with open(TEST_DATA_PATH, "rb") as f:
        test_data = pickle.load(f)

    return model, le, history, test_data


# ══════════════════════════════════════════════════════════════════════════
# 2.  CLASSIFICATION REPORT
# ══════════════════════════════════════════════════════════════════════════

def evaluate_model(model, test_data: dict, le) -> tuple:
    """
    Run model inference on the test set and print metrics.

    Returns predicted integer labels and true integer labels.
    """
    X_test     = test_data["X_test"]
    y_test_raw = test_data["y_test_raw"]  # integer codes

    # Predict class probabilities, then argmax to get class indices
    y_proba = model.predict(X_test, batch_size=64, verbose=0)
    y_pred  = np.argmax(y_proba, axis=1)

    # Map integer codes back to emotion strings for the report
    class_names = le.classes_

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(y_test_raw, y_pred, target_names=class_names)
    print(report)

    # Compute and display overall accuracy
    accuracy = np.mean(y_pred == y_test_raw)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Save the report text to outputs/
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUTS_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(report)
        f.write(f"\nOverall Accuracy: {accuracy:.4f}\n")
    print(f"[INFO] Report saved → {report_path}")

    return y_pred, y_test_raw, class_names


# ══════════════════════════════════════════════════════════════════════════
# 3.  CONFUSION MATRIX HEATMAP
# ══════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           class_names: list) -> None:
    """Plot and save a seaborn heatmap of the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    # Normalise rows to show recall per class (values 0–1)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # ── Raw counts ────────────────────────────────────────────────────────
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix – Raw Counts", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Predicted Label", fontsize=12)
    axes[0].set_ylabel("True Label", fontsize=12)
    axes[0].tick_params(axis="x", rotation=45)

    # ── Normalised (recall) ────────────────────────────────────────────────
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix – Normalised (Recall)", fontsize=14,
                       fontweight="bold")
    axes[1].set_xlabel("Predicted Label", fontsize=12)
    axes[1].set_ylabel("True Label", fontsize=12)
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    save_path = OUTPUTS_DIR / "confusion_matrix.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════
# 4.  TRAINING CURVES
# ══════════════════════════════════════════════════════════════════════════

def plot_training_curves(history: dict) -> None:
    """Plot accuracy and loss curves for training and validation sets."""
    epochs = range(1, len(history["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Accuracy ──────────────────────────────────────────────────────────
    axes[0].plot(epochs, history["accuracy"],     label="Train Accuracy",
                 color="steelblue", linewidth=2)
    axes[0].plot(epochs, history["val_accuracy"], label="Val Accuracy",
                 color="darkorange", linewidth=2, linestyle="--")
    axes[0].set_title("Model Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # ── Loss ──────────────────────────────────────────────────────────────
    axes[1].plot(epochs, history["loss"],     label="Train Loss",
                 color="steelblue", linewidth=2)
    axes[1].plot(epochs, history["val_loss"], label="Val Loss",
                 color="darkorange", linewidth=2, linestyle="--")
    axes[1].set_title("Model Loss", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Loss", fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("EmotionFusion – Training History", fontsize=16,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = OUTPUTS_DIR / "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Training curves saved → {save_path}")


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load everything
    model, le, history, test_data = load_artefacts()

    # 2. Print classification report + compute predictions
    y_pred, y_true, class_names = evaluate_model(model, test_data, le)

    # 3. Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names)

    # 4. Plot training curves
    plot_training_curves(history)

    print("\n[DONE] Evaluation complete.  Plots saved to outputs/")
