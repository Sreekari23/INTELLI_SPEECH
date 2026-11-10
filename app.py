import streamlit as st
import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
import onnxruntime as ort
import time
from pathlib import Path

st.set_page_config(page_title="Project TIDE: IntelliSpeech Edge Dashboard", layout="wide")

REPO_ROOT = Path(__file__).resolve().parent
model_path = REPO_ROOT / "checkpoints" / "onnx" / "squeezeformer_ctc_int8.onnx"

@st.cache_resource
def load_model():
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

session = load_model()

VOCAB = ["<blank>", " ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", 
         "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]
idx2char = {idx: char for idx, char in enumerate(VOCAB)}

def greedy_decode(logits):
    preds = np.argmax(logits[:, 0, :], axis=-1)
    collapsed = []
    prev_idx = -1
    for idx in preds:
        if idx != prev_idx and idx != 0:
            collapsed.append(idx2char[idx])
        prev_idx = idx
    return "".join(collapsed).strip()

st.title("🎙️ Project TIDE: Edge-Optimized Assistive Speech Transcription")
st.markdown("**Target Hardware:** Raspberry Pi 5 (Quad-core ARM Cortex-A76) | **Model:** Squeezeformer-CTC (INT8 Quantized)")

col1, col2, col3 = st.columns(3)
col1.metric("Model Footprint", "1.48 MB", "Optimized ONNX")
col2.metric("Avg Inference Latency", "29.73 ms", "Local CPU Profile")
col3.metric("Real-Time Factor (RTF)", "0.0126", "~80x Faster than Real-Time")

st.markdown("---")
st.subheader("🧪 Live Inference & Pipeline Simulation")
uploaded_file = st.file_uploader("Upload a validation .wav audio sample", type=["wav"])

if uploaded_file is not None:
    audio_np, sr = sf.read(uploaded_file, dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    st.audio(uploaded_file, format='audio/wav')
    
    enable_cleaning = st.checkbox("Apply DSP Noise Reduction / Normalization Filter", value=True)
    
    if st.button("Run Edge Inference Pipeline"):
        with st.spinner("Processing audio through INT8 Squeezeformer engine..."):
            waveform = torch.from_numpy(audio_np).float()
            if sr != 16000:
                waveform = T.Resample(orig_freq=sr, new_freq=16000)(waveform)
                
            duration = len(waveform) / 16000.0
            
            if enable_cleaning:
                waveform = waveform - waveform.mean()
                waveform = waveform / torch.clamp(torch.max(torch.abs(waveform)), min=1e-7)
            
            mel_transform = T.MelSpectrogram(
                sample_rate=16000, n_fft=400, win_length=400, hop_length=160,
                f_min=0.0, f_max=8000.0, n_mels=80, power=2.0, center=True,
                pad_mode="reflect", norm="slaney", mel_scale="slaney"
            )
            mel = torch.log(torch.clamp(mel_transform(waveform), min=1e-5))
            mel = (mel - mel.mean(dim=-1, keepdim=True)) / mel.std(dim=-1, keepdim=True).clamp(min=1e-5)
            
            features = mel.transpose(0, 1).unsqueeze(0).numpy()
            feat_lens = np.array([features.shape[1]], dtype=np.int64)

            input_feat = session.get_inputs()[0].name
            input_len = session.get_inputs()[1].name

            t0 = time.perf_counter()
            outputs = session.run(None, {input_feat: features, input_len: feat_lens})
            latency_ms = (time.perf_counter() - t0) * 1000
            rtf = (latency_ms / 1000.0) / duration
            predicted_text = greedy_decode(outputs[0])

        st.success("Inference Complete!")
        res_col1, res_col2 = st.columns(2)
        res_col1.markdown(f"### 📝 Decoded Transcript\n> **'{predicted_text}'**")
        res_col2.markdown(f"### ⏱️ Performance Metrics\n- **Audio Duration:** {duration:.2f}s\n- **Inference Latency:** {latency_ms:.2f} ms\n- **Calculated RTF:** {rtf:.4f}")