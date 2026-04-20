"""
app.py
------
Modern Gradio web interface for EmotionFusion.

Features:
  - Tab 1: Upload any .wav file → instant emotion analysis
  - Tab 2: Live microphone → real-time emotion detection every ~2 seconds

Run with:
    ../.venv/bin/python app.py
Then open http://localhost:7860 in your browser.
"""

import os
import tempfile
import numpy as np
import gradio as gr
import soundfile as sf
from pathlib import Path
from collections import deque

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Import EmotionFusion prediction pipeline
from predict import predict_emotion, _load_model_artifacts, _load_hubert

# Pre-load both models at startup so first prediction is instant
print("[EmotionFusion] Loading models — please wait…")
_load_model_artifacts()
_load_hubert()
print("[EmotionFusion] Ready!")

# ── Emotion display config ─────────────────────────────────────────────────
EMOTIONS = {
    "anger":    {"emoji": "😠", "color": "#E74C3C", "label": "Anger"},
    "disgust":  {"emoji": "🤢", "color": "#8E44AD", "label": "Disgust"},
    "fear":     {"emoji": "😨", "color": "#9B59B6", "label": "Fear"},
    "happy":    {"emoji": "😊", "color": "#F1C40F", "label": "Happiness"},
    "neutral":  {"emoji": "😐", "color": "#95A5A6", "label": "Neutral"},
    "sad":      {"emoji": "😢", "color": "#3498DB", "label": "Sadness"},
    "surprise": {"emoji": "😲", "color": "#E67E22", "label": "Surprise"},
}

MIN_SECONDS  = 2.0   # minimum audio length before running real-time prediction
STREAM_SR    = 16000 # Gradio streams mic audio at 16 kHz

# ── Custom CSS ─────────────────────────────────────────────────────────────
CSS = """
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: #0d0d1a !important;
    color: #e0e0e0 !important;
}

/* ── Header ── */
.ef-header {
    text-align: center;
    padding: 2rem 1rem 1rem;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid #1e3a5f;
}
.ef-header h1 {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00d2ff, #7b2ff7, #ff6b6b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem;
}
.ef-header p {
    color: #8899aa;
    font-size: 1rem;
    margin: 0;
}

/* ── Results card ── */
.ef-result-card {
    background: #12122a;
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid #1e2a40;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.ef-main-emotion {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1.2rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    font-weight: 700;
}
.ef-emoji  { font-size: 3rem; line-height: 1; }
.ef-elabel { font-size: 1.6rem; letter-spacing: 0.1em; text-transform: uppercase; }
.ef-conf   { margin-left: auto; font-size: 2rem; opacity: 0.9; }

/* ── Probability bars ── */
.ef-bars { display: flex; flex-direction: column; gap: 0.65rem; }
.ef-bar-row {
    display: grid;
    grid-template-columns: 130px 1fr 56px;
    align-items: center;
    gap: 0.7rem;
}
.ef-bar-name  { font-size: 0.88rem; color: #b0b8c8; }
.ef-bar-track {
    height: 10px;
    background: #1e2a40;
    border-radius: 99px;
    overflow: hidden;
}
.ef-bar-fill  { height: 100%; border-radius: 99px; transition: width 0.5s ease; }
.ef-bar-pct   { font-size: 0.82rem; color: #7888a0; text-align: right; }

/* ── History ── */
.ef-history {
    background: #12122a;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    border: 1px solid #1e2a40;
    max-height: 260px;
    overflow-y: auto;
}
.ef-history h3 { margin: 0 0 0.8rem; font-size: 0.9rem; color: #556677; text-transform: uppercase; letter-spacing: 0.08em; }
.ef-hist-item  { display: flex; align-items: center; gap: 0.6rem; padding: 0.4rem 0; border-bottom: 1px solid #1a2233; font-size: 0.9rem; }
.ef-hist-item:last-child { border-bottom: none; }

/* ── Status pill ── */
.ef-status {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 99px;
    font-size: 0.82rem;
    font-weight: 600;
    background: #1a2a1a;
    color: #4caf50;
    border: 1px solid #2a4a2a;
    margin-bottom: 1rem;
}
.ef-status.idle { background: #1a1a2a; color: #556677; border-color: #2a2a3a; }

/* ── Gradio overrides ── */
.gradio-container .tab-nav button {
    background: transparent !important;
    color: #6688aa !important;
    border: none !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    padding: 0.7rem 1.4rem !important;
}
.gradio-container .tab-nav button.selected {
    color: #00d2ff !important;
    border-bottom: 2px solid #00d2ff !important;
}
.gradio-container .block { background: #12122a !important; border-color: #1e2a40 !important; }
.gradio-container label span { color: #8899aa !important; }
button.primary { background: linear-gradient(135deg, #00d2ff, #7b2ff7) !important; border: none !important; font-weight: 600 !important; }
button.primary:hover { opacity: 0.88 !important; }
"""

# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def build_results_html(emotion: str, confidence: float, all_probs: dict) -> str:
    """Render the styled results card as an HTML string."""
    cfg  = EMOTIONS[emotion]
    col  = cfg["color"]

    # Main emotion banner
    html = f"""
    <div class="ef-result-card">
      <div class="ef-main-emotion" style="background:{col}22; border:2px solid {col}55;">
        <span class="ef-emoji">{cfg['emoji']}</span>
        <span class="ef-elabel" style="color:{col};">{cfg['label']}</span>
        <span class="ef-conf" style="color:{col};">{confidence*100:.1f}%</span>
      </div>
      <div class="ef-bars">
    """

    # Probability bars — sorted by probability descending
    for em, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        c  = EMOTIONS[em]["color"]
        lbl = f"{EMOTIONS[em]['emoji']} {EMOTIONS[em]['label']}"
        pct = prob * 100
        html += f"""
        <div class="ef-bar-row">
          <span class="ef-bar-name">{lbl}</span>
          <div class="ef-bar-track">
            <div class="ef-bar-fill" style="width:{pct:.1f}%; background:{c};"></div>
          </div>
          <span class="ef-bar-pct">{pct:.1f}%</span>
        </div>"""

    html += "</div></div>"
    return html


def build_history_html(history: list) -> str:
    """Render the live-analysis history list as HTML."""
    if not history:
        return '<div class="ef-history"><h3>History</h3><span style="color:#445566;font-size:0.85rem;">Predictions will appear here…</span></div>'

    items = ""
    for ts, em, conf in reversed(history[-12:]):   # show last 12
        cfg = EMOTIONS[em]
        items += f"""
        <div class="ef-hist-item">
          <span>{cfg['emoji']}</span>
          <span style="color:{cfg['color']};font-weight:600;">{cfg['label']}</span>
          <span style="margin-left:auto;color:#556677;">{conf*100:.0f}%</span>
          <span style="color:#334455;font-size:0.78rem;">{ts}</span>
        </div>"""

    return f'<div class="ef-history"><h3>Recent predictions</h3>{items}</div>'


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — FILE UPLOAD
# ══════════════════════════════════════════════════════════════════════════

def analyze_file(audio):
    """
    Called when the user uploads or records an audio file.
    Returns the styled HTML results card.
    """
    if audio is None:
        return '<p style="color:#445566;text-align:center;padding:2rem;">Upload or record an audio clip to analyse it.</p>'

    try:
        # Gradio returns a file path for uploaded audio
        emotion, confidence, all_probs = predict_emotion(audio)
        return build_results_html(emotion, confidence, all_probs)
    except Exception as e:
        return f'<p style="color:#E74C3C;padding:1rem;">Error: {e}</p>'


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — REAL-TIME MICROPHONE
# ══════════════════════════════════════════════════════════════════════════

def process_stream(audio_chunk, state):
    """
    Called for every incoming microphone chunk (~0.5 s).
    Accumulates audio until MIN_SECONDS, then runs prediction.

    state = {"buffer": [...], "samples": int, "history": [...], "count": int}
    """
    # Initialise state on first call
    if state is None:
        state = {"buffer": [], "samples": 0, "history": [], "count": 0}

    if audio_chunk is None:
        history_html = build_history_html(state["history"])
        idle_html = '<p style="color:#445566;text-align:center;padding:2rem;">Press <b>Record</b> to start live analysis.</p>'
        return state, idle_html, history_html

    sr, data = audio_chunk

    # Convert stereo → mono if needed
    if data.ndim == 2:
        data = data.mean(axis=1)

    # Normalise int16 → float32
    if data.dtype != np.float32:
        data = data.astype(np.float32) / 32768.0

    # Accumulate chunk into buffer
    state["buffer"].append(data)
    state["samples"] += len(data)

    # Not enough audio yet — show collecting status
    collected = state["samples"] / sr
    if collected < MIN_SECONDS:
        status_html = f'<div class="ef-status">🎙 Collecting… {collected:.1f}s / {MIN_SECONDS:.0f}s</div>'
        return state, status_html, build_history_html(state["history"])

    # ── Run prediction ────────────────────────────────────────────────────
    combined = np.concatenate(state["buffer"])

    # Save buffer to a temp .wav and call predict_emotion
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, combined, sr)
        tmp_path = tmp.name

    try:
        emotion, confidence, all_probs = predict_emotion(tmp_path)
    except Exception as e:
        os.unlink(tmp_path)
        err = f'<p style="color:#E74C3C;">Prediction error: {e}</p>'
        return state, err, build_history_html(state["history"])

    os.unlink(tmp_path)

    # Update history
    import datetime
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    state["history"].append((ts, emotion, confidence))
    state["count"] += 1

    # Reset buffer for next window
    state["buffer"]  = []
    state["samples"] = 0

    result_html  = build_results_html(emotion, confidence, all_probs)
    history_html = build_history_html(state["history"])

    return state, result_html, history_html


def reset_stream():
    """Clear the stream state when the user stops recording."""
    empty = {"buffer": [], "samples": 0, "history": [], "count": 0}
    idle  = '<p style="color:#445566;text-align:center;padding:2rem;">Press <b>Record</b> to start live analysis.</p>'
    return empty, idle, build_history_html([])


# ══════════════════════════════════════════════════════════════════════════
# BUILD UI
# ══════════════════════════════════════════════════════════════════════════

with gr.Blocks(css=CSS, title="EmotionFusion") as demo:

    # ── Header ────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="ef-header">
      <h1>🎙 EmotionFusion</h1>
      <p>Speech Emotion Recognition · Anger · Disgust · Fear · Happiness · Neutral · Sadness · Surprise</p>
    </div>
    """)

    # ── Tabs ──────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ════════════════════════════════════════════════════════════════
        # TAB 1 — UPLOAD / RECORD
        # ════════════════════════════════════════════════════════════════
        with gr.Tab("📂  Analyse File"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="Upload or record a speech clip",
                    )
                    analyse_btn = gr.Button("Analyse Emotion", variant="primary")

                with gr.Column(scale=1):
                    file_result = gr.HTML(
                        value='<p style="color:#445566;text-align:center;padding:2rem;">Results will appear here.</p>'
                    )

            analyse_btn.click(fn=analyze_file, inputs=audio_input, outputs=file_result)
            # Also auto-analyse when a file is uploaded
            audio_input.change(fn=analyze_file, inputs=audio_input, outputs=file_result)

        # ════════════════════════════════════════════════════════════════
        # TAB 2 — REAL-TIME MICROPHONE
        # ════════════════════════════════════════════════════════════════
        with gr.Tab("🎤  Live Analysis"):
            gr.HTML('<p style="color:#667788;font-size:0.88rem;margin:0 0 1rem;">Speak into your microphone — EmotionFusion predicts your emotion every ~2 seconds automatically.</p>')

            with gr.Row():
                with gr.Column(scale=1):
                    mic_input = gr.Audio(
                        sources=["microphone"],
                        streaming=True,
                        type="numpy",
                        label="Live Microphone",
                    )
                    clear_btn = gr.Button("Clear History", variant="secondary")

                with gr.Column(scale=1):
                    stream_state   = gr.State(None)
                    live_result    = gr.HTML(
                        value='<p style="color:#445566;text-align:center;padding:2rem;">Press <b>Record</b> to start live analysis.</p>'
                    )
                    history_output = gr.HTML(value=build_history_html([]))

            # Stream audio chunks → process_stream
            mic_input.stream(
                fn=process_stream,
                inputs=[mic_input, stream_state],
                outputs=[stream_state, live_result, history_output],
            )

            # Clear button resets state + UI
            clear_btn.click(
                fn=reset_stream,
                inputs=[],
                outputs=[stream_state, live_result, history_output],
            )

    # ── Footer ────────────────────────────────────────────────────────────
    gr.HTML('<p style="text-align:center;color:#334455;font-size:0.78rem;margin-top:1.5rem;">EmotionFusion · 1D CNN + HuBERT · Trained on CREMA-D, RAVDESS, TESS, SAVEE, Hindi SER</p>')


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,          # set True to get a public gradio.live link
        show_error=True,
    )
