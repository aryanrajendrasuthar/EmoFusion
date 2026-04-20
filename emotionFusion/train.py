"""
train.py
--------
Loads pre-extracted features, builds the 1-D CNN model,
trains it, and saves the best weights to models/emotionFusion_model.h5.

Run after extract_features.py has completed successfully.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Suppress TF info/warning logs (keep errors)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
FEATURES_FILE = BASE_DIR / "features" / "features.pkl"
MODELS_DIR    = BASE_DIR / "models"
MODEL_PATH    = MODELS_DIR / "emotionFusion_model.h5"
SCALER_PATH   = MODELS_DIR / "scaler.pkl"
ENCODER_PATH  = MODELS_DIR / "label_encoder.pkl"
HISTORY_PATH  = BASE_DIR / "features" / "training_history.pkl"

# ── Hyperparameters ────────────────────────────────────────────────────────
BATCH_SIZE   = 64
MAX_EPOCHS   = 50
LEARNING_RATE = 0.001
TEST_SIZE    = 0.20
RANDOM_SEED  = 42

NUM_CLASSES  = 7
FEATURE_DIM  = 2380  # 108 ZCR + 108 RMS + 2160 MFCC + 4 HuBERT-sentiment (paper-exact)


# ══════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING & PREPARATION
# ══════════════════════════════════════════════════════════════════════════

def load_features() -> tuple:
    """Load feature matrix and label list from the cache file."""
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURES_FILE}\n"
            "Please run extract_features.py first."
        )
    with open(FEATURES_FILE, "rb") as f:
        data = pickle.load(f)
    X, y = data["X"], data["y"]
    print(f"[INFO] Loaded features: X={X.shape}, y={len(y)} samples")
    return X, y


def prepare_data(X: np.ndarray, y: list) -> tuple:
    """
    Encode labels, scale features, split into train/test sets,
    and one-hot encode the target.

    Returns:
        X_train, X_test : np.ndarray, shape (N, 318, 1)
        y_train, y_test : np.ndarray, one-hot encoded, shape (N, 7)
        scaler          : fitted StandardScaler
        le              : fitted LabelEncoder
    """
    # ── Label encoding ────────────────────────────────────────────────────
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)   # integer codes 0–6
    print(f"[INFO] Emotion classes: {list(le.classes_)}")

    # ── Stratified train/test split ───────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        stratify=y_encoded,
        random_state=RANDOM_SEED,
    )
    print(f"[INFO] Train: {len(X_train)}  Test: {len(X_test)}")

    # ── Feature standardisation (fit only on training data) ───────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Reshape for Conv1D: (samples, features, 1) ───────────────────────
    X_train = X_train.reshape(-1, FEATURE_DIM, 1)
    X_test  = X_test.reshape(-1, FEATURE_DIM, 1)

    # ── One-hot encode targets ────────────────────────────────────────────
    y_train_oh = keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test_oh  = keras.utils.to_categorical(y_test,  num_classes=NUM_CLASSES)

    return X_train, X_test, y_train_oh, y_test_oh, scaler, le, y_test


# ══════════════════════════════════════════════════════════════════════════
# 2.  MODEL DEFINITION
# ══════════════════════════════════════════════════════════════════════════

def build_model(input_shape: tuple = (FEATURE_DIM, 1)) -> keras.Model:
    """
    Build the 1-D CNN architecture described in the spec.

    Architecture:
        5 × (Conv1D → BatchNorm → ReLU → MaxPool) blocks
        Flatten → Dense(512, relu) → Dropout(0.3) → Dense(7, softmax)
    """
    model = keras.Sequential(name="EmotionFusion_CNN")

    # ── Block 1: 512 filters ──────────────────────────────────────────────
    model.add(layers.Conv1D(512, kernel_size=5, padding="same",
                            input_shape=input_shape, name="conv1"))
    model.add(layers.BatchNormalization(name="bn1"))
    model.add(layers.Activation("relu", name="relu1"))
    model.add(layers.MaxPooling1D(pool_size=5, strides=2, name="pool1"))

    # ── Block 2: 512 filters ──────────────────────────────────────────────
    model.add(layers.Conv1D(512, kernel_size=5, padding="same", name="conv2"))
    model.add(layers.BatchNormalization(name="bn2"))
    model.add(layers.Activation("relu", name="relu2"))
    model.add(layers.MaxPooling1D(pool_size=5, strides=2, name="pool2"))

    # ── Block 3: 256 filters ──────────────────────────────────────────────
    model.add(layers.Conv1D(256, kernel_size=5, padding="same", name="conv3"))
    model.add(layers.BatchNormalization(name="bn3"))
    model.add(layers.Activation("relu", name="relu3"))
    model.add(layers.MaxPooling1D(pool_size=5, strides=2, name="pool3"))

    # ── Block 4: 256 filters ──────────────────────────────────────────────
    model.add(layers.Conv1D(256, kernel_size=5, padding="same", name="conv4"))
    model.add(layers.BatchNormalization(name="bn4"))
    model.add(layers.Activation("relu", name="relu4"))
    model.add(layers.MaxPooling1D(pool_size=5, strides=2, name="pool4"))

    # ── Block 5: 128 filters ──────────────────────────────────────────────
    model.add(layers.Conv1D(128, kernel_size=5, padding="same", name="conv5"))
    model.add(layers.BatchNormalization(name="bn5"))
    model.add(layers.Activation("relu", name="relu5"))
    model.add(layers.MaxPooling1D(pool_size=5, strides=2, name="pool5"))

    # ── Classifier head ───────────────────────────────────────────────────
    model.add(layers.Flatten(name="flatten"))
    model.add(layers.Dense(512, activation="relu", name="dense1"))
    model.add(layers.Dropout(0.3, name="dropout"))
    model.add(layers.Dense(NUM_CLASSES, activation="softmax", name="output"))

    return model


# ══════════════════════════════════════════════════════════════════════════
# 3.  TRAINING
# ══════════════════════════════════════════════════════════════════════════

def train(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> keras.callbacks.History:
    """
    Compile and train the model with the specified callbacks.

    Returns the Keras History object for later plotting.
    """
    # Compile with RMSprop and categorical cross-entropy
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,   # revert to best epoch weights on stop
        verbose=1,
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,      # halve the learning rate
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )

    # Checkpoint – save best weights during training
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1,
    )

    return history


def save_artifacts(scaler: StandardScaler, le: LabelEncoder,
                   history: keras.callbacks.History) -> None:
    """Persist scaler, label encoder, and training history for evaluate.py."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    # Save history dict so evaluate.py can plot curves without re-training
    history_dir = BASE_DIR / "features"
    history_dir.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "wb") as f:
        pickle.dump(history.history, f)

    print(f"[INFO] Model saved    → {MODEL_PATH}")
    print(f"[INFO] Scaler saved   → {SCALER_PATH}")
    print(f"[INFO] Encoder saved  → {ENCODER_PATH}")
    print(f"[INFO] History saved  → {HISTORY_PATH}")


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Load features
    X, y = load_features()

    # 2. Prepare data splits, scaling, encoding
    X_train, X_test, y_train_oh, y_test_oh, scaler, le, y_test_raw = prepare_data(X, y)

    # 3. Build model
    model = build_model()

    # 4. Train
    history = train(model, X_train, y_train_oh, X_test, y_test_oh)

    # 5. Persist artefacts
    save_artifacts(scaler, le, history)

    # Save the raw test labels alongside features for evaluate.py
    test_data = {"X_test": X_test, "y_test_raw": y_test_raw, "y_test_oh": y_test_oh}
    with open(BASE_DIR / "features" / "test_data.pkl", "wb") as f:
        pickle.dump(test_data, f)

    print("\n[DONE] Training complete.  Run `python evaluate.py` next.")
