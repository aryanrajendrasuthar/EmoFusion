# EmotionFusion — Speech Emotion Recognition System

EmotionFusion classifies emotions from speech audio into **7 categories**:  
`anger · disgust · fear · happy · neutral · sad · surprise`

It fuses hand-crafted acoustic features (ZCR, RMSE, MFCCs + deltas) with  
deep HuBERT embeddings and trains a 1-D CNN classifier.

---

## Project Structure

```
emotionFusion/
├── data/               # raw dataset folders (populated via Kaggle CLI)
├── features/           # cached .pkl feature files
├── models/             # saved .h5 model + scaler + encoder
├── outputs/            # plots and classification report
├── extract_features.py # Step 1 – dataset loading, augmentation, feature extraction
├── train.py            # Step 2 – model definition and training
├── evaluate.py         # Step 3 – evaluation, confusion matrix, training curves
├── predict.py          # Inference helper
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1 · Install dependencies

```bash
pip install -r requirements.txt
```

### 2 · Download datasets via Kaggle CLI

Make sure your `~/.kaggle/kaggle.json` API key is in place, then:

```bash
# CREMA-D
kaggle datasets download ejlok1/cremad -p data/CREMA-D --unzip

# RAVDESS
kaggle datasets download uwrfkaggler/ravdess-emotional-speech-audio -p data/RAVDESS --unzip

# TESS
kaggle datasets download ejlok1/toronto-emotional-speech-set-tess -p data/TESS --unzip

# SAVEE
kaggle datasets download ejlok1/surrey-audiovisual-expressed-emotion-savee -p data/SAVEE --unzip

# Hindi SER
kaggle datasets download vishlb/speech-emotion-recognition-hindi -p data/Hindi --unzip
```

Expected `data/` layout after extraction:

```
data/
├── CREMA-D/AudioWAV/        *.wav files
├── RAVDESS/Actor_*/         *.wav files
├── TESS/<emotion_folders>/  *.wav files
├── SAVEE/<speaker_folders>/ *.wav files
└── Hindi/<emotion_folders>/ *.wav files
```

### 3 · Extract features  *(~hours for full dataset — run once)*

```bash
python extract_features.py
```

Produces `features/features.pkl` (~4× original dataset after augmentation).

### 4 · Train the model

```bash
python train.py
```

Saves:
- `models/emotionFusion_model.h5`
- `models/scaler.pkl`
- `models/label_encoder.pkl`
- `features/training_history.pkl`
- `features/test_data.pkl`

### 5 · Evaluate

```bash
python evaluate.py
```

Produces:
- `outputs/classification_report.txt`
- `outputs/confusion_matrix.png`
- `outputs/training_curves.png`

### 6 · Inference

```bash
python predict.py /path/to/audio.wav
```

Or in Python:

```python
from predict import predict_emotion

emotion, confidence, all_probs = predict_emotion("speech.wav")
print(f"Emotion: {emotion}  ({confidence*100:.1f}%)")
```

---

## Architecture

```
Input (318, 1)
  │
  ├─ Conv1D(512, k=5) → BN → ReLU → MaxPool(5, s=2)
  ├─ Conv1D(512, k=5) → BN → ReLU → MaxPool(5, s=2)
  ├─ Conv1D(256, k=5) → BN → ReLU → MaxPool(5, s=2)
  ├─ Conv1D(256, k=5) → BN → ReLU → MaxPool(5, s=2)
  ├─ Conv1D(128, k=5) → BN → ReLU → MaxPool(5, s=2)
  │
  ├─ Flatten
  ├─ Dense(512, relu) → Dropout(0.3)
  └─ Dense(7, softmax)
```

**Feature vector (318-D)**:

| Component                          | Dim |
|------------------------------------|-----|
| Zero Crossing Rate (mean)          | 1   |
| RMSE (mean)                        | 1   |
| MFCCs × 20 (mean)                  | 20  |
| MFCC Δ × 20 (mean)                 | 20  |
| MFCC ΔΔ × 20 (mean)               | 20  |
| HuBERT pooled hidden state         | 256 |
| **Total**                          | **318** |

---

## Training Configuration

| Setting              | Value                        |
|----------------------|------------------------------|
| Optimizer            | RMSprop (lr=0.001)           |
| Loss                 | Categorical cross-entropy    |
| Batch size           | 64                           |
| Max epochs           | 50                           |
| Early stopping       | patience=5 on val_loss       |
| LR reduction         | ×0.5 every 3 plateau epochs  |
| Train / Test split   | 80 / 20 (stratified)         |
| Feature scaling      | StandardScaler (fit on train)|

---

## Datasets

| Dataset   | Language | Kaggle slug                                    |
|-----------|----------|------------------------------------------------|
| CREMA-D   | English  | ejlok1/cremad                                  |
| RAVDESS   | English  | uwrfkaggler/ravdess-emotional-speech-audio     |
| TESS      | English  | ejlok1/toronto-emotional-speech-set-tess       |
| SAVEE     | English  | ejlok1/surrey-audiovisual-expressed-emotion-savee |
| Hindi SER | Hindi    | vishlb/speech-emotion-recognition-hindi        |

---

## Notes

- GPU is highly recommended for HuBERT inference during feature extraction.
- Feature extraction is the slowest step (~1–3 s per file × 4 augmentations).
  Once `features/features.pkl` exists, it will not be recomputed.
- The HuBERT model (`superb/hubert-base-superb-er`) is downloaded automatically
  from HuggingFace on first run and cached by the `transformers` library.
