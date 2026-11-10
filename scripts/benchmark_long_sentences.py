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

def main():
    REPO_ROOT = Path(__file__).resolve().parent.parent
    model_path = REPO_ROOT / "checkpoints" / "onnx" / "squeezeformer_ctc_int8.onnx"
    
    print(f"[*] Loading INT8 ONNX model for long-sentence evaluation...")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    
    val_manifests = glob.glob(str(REPO_ROOT / "**/val/manifest.json"), recursive=True)
    if not val_manifests:
        print("[!] Validation manifest not found.")
        return
        
    with open(val_manifests[0], "r", encoding="utf-8") as f:
        raw = json.load(f)
    records = raw.get("samples", raw.get("cases", [])) if isinstance(raw, dict) else raw
    base_dir = os.path.dirname(val_manifests[0])

    # Filter records by text length to find longer sentences/phrases
    sorted_records = sorted(records, key=lambda x: len(x.get("text") or x.get("transcript") or x.get("reference", "")), reverse=True)
    
    print(f"\n{'='*70}")
    print(f"{'AUDIO FILE':<25} | {'DURATION':<8} | {'LATENCY':<10} | {'RTF':<8} | {'GROUND TRUTH'}")
    print(f"{'='*70}")

    mel_transform = T.MelSpectrogram(
        sample_rate=16000, n_fft=400, win_length=400, hop_length=160,
        f_min=0.0, f_max=8000.0, n_mels=80, power=2.0, center=True,
        pad_mode="reflect", norm="slaney", mel_scale="slaney"
    )

    input_name_feat = session.get_inputs()[0].name
    input_name_len = session.get_inputs()[1].name

    for rec in sorted_records[:5]:
        rel_path = rec.get("file") or rec.get("audio_filepath") or rec.get("filename", "")
        wav_path = os.path.join(base_dir, rel_path)
        if not os.path.exists(wav_path):
            wav_path = rel_path
            
        ref_text = rec.get("text") or rec.get("transcript") or rec.get("reference", "")
        if not os.path.exists(wav_path):
            continue

        audio_np, sr = sf.read(wav_path, dtype="float32")
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        waveform = torch.from_numpy(audio_np).float()
        if sr != 16000:
            waveform = T.Resample(orig_freq=sr, new_freq=16000)(waveform)
            
        duration = len(waveform) / 16000.0
        waveform = waveform / torch.clamp(torch.max(torch.abs(waveform)), min=1e-7)
        
        mel = torch.log(torch.clamp(mel_transform(waveform), min=1e-5))
        mel = (mel - mel.mean(dim=-1, keepdim=True)) / mel.std(dim=-1, keepdim=True).clamp(min=1e-5)
        
        features = mel.transpose(0, 1).unsqueeze(0).numpy()
        feat_lens = np.array([features.shape[1]], dtype=np.int64)

        t0 = time.perf_counter()
        outputs = session.run(None, {input_name_feat: features, input_name_len: feat_lens})
        latency_ms = (time.perf_counter() - t0) * 1000
        rtf = (latency_ms / 1000.0) / duration

        filename = os.path.basename(wav_path)
        print(f"{filename:<25} | {duration:6.2f}s  | {latency_ms:7.2f}ms  | {rtf:6.4f} | '{ref_text}'")

    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()

    