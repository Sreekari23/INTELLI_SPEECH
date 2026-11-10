import os
import glob
import json
import time
import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
import onnxruntime as ort
from pathlib import Path

# 1. Setup paths & Vocabulary
REPO_ROOT = Path(__file__).resolve().parent.parent
model_path = REPO_ROOT / "checkpoints" / "onnx" / "squeezeformer_ctc_int8.onnx"

VOCAB = ["<blank>", " ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", 
         "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]
idx2char = {idx: char for idx, char in enumerate(VOCAB)}

def greedy_decode(logits):
    # logits shape: [seq_len, batch_size, num_classes] -> take batch 0
    preds = np.argmax(logits[:, 0, :], axis=-1)
    collapsed = []
    prev_idx = -1
    for idx in preds:
        if idx != prev_idx and idx != 0:
            collapsed.append(idx2char[idx])
        prev_idx = idx
    return "".join(collapsed).strip()

def main():
    print(f"[*] Loading INT8 ONNX model from {model_path}...")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    
    # 2. Grab a random audio file from validation manifests to test
    val_manifests = glob.glob(str(REPO_ROOT / "**/val/manifest.json"), recursive=True)
    if not val_manifests:
        # Fallback to general search if structure varies locally
        val_manifests = glob.glob("**/manifest.json", recursive=True)
        
    if not val_manifests:
        print("[!] Could not locate a validation manifest automatically. Place a test .wav file path directly in the script.")
        return

    manifest_path = val_manifests[0]
    with open(manifest_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    records = raw.get("samples", raw.get("cases", [])) if isinstance(raw, dict) else raw
    
    # Pick the first record with a reference text
    sample_rec = records[0]
    base_dir = os.path.dirname(manifest_path)
    rel_path = sample_rec.get("file") or sample_rec.get("audio_filepath") or sample_rec.get("filename", "")
    wav_path = os.path.join(base_dir, rel_path)
    if not os.path.exists(wav_path):
        wav_path = rel_path # try direct path
        
    reference_text = sample_rec.get("text") or sample_rec.get("transcript") or sample_rec.get("reference", "")
    print(f"[*] Test Audio File: {wav_path}")
    print(f"[*] Ground Truth Text: '{reference_text}'")

    # 3. Extract 80-channel Log-Mel features (matching training pipeline)
    audio_np, sr = sf.read(wav_path, dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    waveform = torch.from_numpy(audio_np).float()
    if sr != 16000:
        waveform = T.Resample(orig_freq=sr, new_freq=16000)(waveform)

    peak = torch.clamp(torch.max(torch.abs(waveform)), min=1e-7)
    waveform = waveform / peak
    
    mel_transform = T.MelSpectrogram(
        sample_rate=16000, n_fft=400, win_length=400, hop_length=160,
        f_min=0.0, f_max=8000.0, n_mels=80, power=2.0, center=True,
        pad_mode="reflect", norm="slaney", mel_scale="slaney"
    )
    mel = torch.log(torch.clamp(mel_transform(waveform), min=1e-5))
    mel = (mel - mel.mean(dim=-1, keepdim=True)) / mel.std(dim=-1, keepdim=True).clamp(min=1e-5)
    
    # Format input tensors: features [1, time_frames, 80], lengths [1]
    features = mel.transpose(0, 1).unsqueeze(0).numpy() # Shape: [1, T, 80]
    feat_lens = np.array([features.shape[1]], dtype=np.int64)

    # 4. Run ONNX Inference & Time it
    input_name_feat = session.get_inputs()[0].name
    input_name_len = session.get_inputs()[1].name

    print("[*] Running local ONNX Runtime inference...")
    t0 = time.perf_counter()
    outputs = session.run(None, {
        input_name_feat: features,
        input_name_len: feat_lens
    })
    inference_time = (time.perf_counter() - t0) * 1000

    logits = outputs[0] # Shape: [seq_len, batch_size, vocab_size]
    predicted_text = greedy_decode(logits)

    print("\n" + "="*40)
    print(f"🎤 Predicted Transcript : '{predicted_text}'")
    print(f"⏱️ Local Inference Latency: {inference_time:.2f} ms")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()