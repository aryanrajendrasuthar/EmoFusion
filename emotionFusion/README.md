# EmotionFusion — Speech Emotion Recognition System

EmotionFusion classifies emotions from speech audio into **7 categories**:  
`anger · disgust · fear · happy · neutral · sad · surprise`

It fuses hand-crafted acoustic features (ZCR, RMSE, MFCCs) with  
HuBERT sentiment classification and trains a 1-D CNN classifier.

**Paper:** *EmotionFusion: A Deep Learning Approach to Speech Emotion Recognition*  
Shubham Vankalas & Vandana Dhingra — Savitribai Phule Pune University  
**Reported Accuracy: 96.56%**

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
├── app.py              # Modern Gradio web UI with real-time mic analysis
├── requirements.txt
└── README.md
```

---

## Setting Up on a New Device

### 1 · Clone the repository

```bash
git clone https://github.com/aryanrajendrasuthar/EmoFusion.git
cd EmoFusion
```

### 2 · Create virtual environment with Python 3.11

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

> **Note:** TensorFlow requires Python 3.11 or lower. Python 3.12+ is not supported.

### 3 · Install dependencies

```bash
pip install -r emotionFusion/requirements.txt
```

### 4 · Download datasets via Kaggle CLI

Get your Kaggle API token from [kaggle.com](https://www.kaggle.com) → Account → Settings → API → Generate New Token, then:

```bash
export KAGGLE_TOKEN=your_token_here

cd emotionFusion

../.venv/bin/kaggle datasets download ejlok1/cremad -p data/CREMA-D --unzip
../.venv/bin/kaggle datasets download uwrfkaggler/ravdess-emotional-speech-audio -p data/RAVDESS --unzip
../.venv/bin/kaggle datasets download ejlok1/toronto-emotional-speech-set-tess -p data/TESS --unzip
../.venv/bin/kaggle datasets download ejlok1/surrey-audiovisual-expressed-emotion-savee -p data/SAVEE --unzip
../.venv/bin/kaggle datasets download vishlb/speech-emotion-recognition-hindi -p data/Hindi --unzip
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

### 5 · Run the pipeline

```bash
# Step 1 — Extract features (runs once, takes a few hours)
../.venv/bin/python extract_features.py

# Step 2 — Train the model
../.venv/bin/python train.py

# Step 3 — Evaluate and generate plots
../.venv/bin/python evaluate.py

# Step 4 — Launch the web UI
../.venv/bin/python app.py
```

Then open **http://localhost:7860** in your browser.

---

## Architecture

```
Input (2380, 1)
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

**Feature vector (2380-D)** — paper-exact implementation:

| Component                              | Dim    |
|----------------------------------------|--------|
| Zero Crossing Rate (108 frames)        | 108    |
| RMSE (108 frames)                      | 108    |
| MFCCs × 20 coefficients × 108 frames  | 2160   |
| HuBERT 4-class sentiment (one-hot)     | 4      |
| **Total**                              | **2380** |

---

## Training Configuration

| Setting              | Value                         |
|----------------------|-------------------------------|
| Optimizer            | RMSprop (lr=0.001)            |
| Loss                 | Categorical cross-entropy     |
| Batch size           | 64                            |
| Max epochs           | 50                            |
| Early stopping       | patience=5 on val_loss        |
| LR reduction         | ×0.5 every 3 plateau epochs   |
| Train / Test split   | 80 / 20 (stratified)          |
| Feature scaling      | StandardScaler (fit on train) |
| Audio fixed length   | 2.5 seconds (55125 samples)   |

---

## Datasets

| Dataset   | Language | Kaggle slug                                        |
|-----------|----------|----------------------------------------------------|
| CREMA-D   | English  | ejlok1/cremad                                      |
| RAVDESS   | English  | uwrfkaggler/ravdess-emotional-speech-audio         |
| TESS      | English  | ejlok1/toronto-emotional-speech-set-tess           |
| SAVEE     | English  | ejlok1/surrey-audiovisual-expressed-emotion-savee  |
| Hindi SER | Hindi    | vishlb/speech-emotion-recognition-hindi            |

---

## Inference

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

## Notes

- GPU is highly recommended for HuBERT inference during feature extraction.
- Feature extraction is the slowest step (~1–3 s per file × 4 augmentations).
  Once `features/features.pkl` exists, it will not be recomputed.
- The HuBERT model (`superb/hubert-base-superb-er`) is downloaded automatically
  from HuggingFace on first run and cached by the `transformers` library.
- Data files (`data/`, `models/`, `outputs/`) are excluded from git — you must
  download datasets and re-run the pipeline on each new device.
