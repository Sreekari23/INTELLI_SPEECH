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

def main():
    REPO_ROOT = Path(__file__).resolve().parent.parent
    model_path = REPO_ROOT / "checkpoints" / "onnx" / "squeezeformer_ctc_int8.onnx"
    
    print(f"[*] Loading INT8 ONNX model for latency profiling...")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    
    # Locate sample audio
    val_manifests = glob.glob(str(REPO_ROOT / "**/val/manifest.json"), recursive=True)
    if not val_manifests:
        print("[!] Validation manifest not found.")
        return
        
    with open(val_manifests[0], "r", encoding="utf-8") as f:
        raw = json.load(f)
    records = raw.get("samples", raw.get("cases", [])) if isinstance(raw, dict) else raw
    
    sample_rec = records[0]
    base_dir = os.path.dirname(val_manifests[0])
    wav_path = os.path.join(base_dir, sample_rec.get("file") or sample_rec.get("audio_filepath") or sample_rec.get("filename", ""))
    if not os.path.exists(wav_path):
        wav_path = sample_rec.get("file") or sample_rec.get("audio_filepath") or sample_rec.get("filename", "")

    # Pre-process audio
    audio_np, sr = sf.read(wav_path, dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    waveform = torch.from_numpy(audio_np).float()
    if sr != 16000:
        waveform = T.Resample(orig_freq=sr, new_freq=16000)(waveform)
        
    audio_duration_sec = len(waveform) / 16000.0
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

    input_name_feat = session.get_inputs()[0].name
    input_name_len = session.get_inputs()[1].name

    # Warm-up run (discards JIT/allocation spike)
    print("[*] Performing warm-up run...")
    _ = session.run(None, {input_name_feat: features, input_name_len: feat_lens})

    # Benchmark loop (50 iterations)
    iterations = 50
    print(f"[*] Running latency benchmark across {iterations} iterations...")
    
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        session.run(None, {input_name_feat: features, input_name_len: feat_lens})
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    avg_lat = np.mean(latencies)
    p50_lat = np.percentile(latencies, 50)
    p95_lat = np.percentile(latencies, 95)
    rtf = (avg_lat / 1000.0) / audio_duration_sec

    print("\n" + "="*50)
    print(f"📊 INT8 ONNX LATENCY BENCHMARK REPORT")
    print("="*50)
    print(f" Audio Sample Duration : {audio_duration_sec:.2f} s")
    print(f" Average Latency       : {avg_lat:.2f} ms")
    print(f" Median (p50) Latency  : {p50_lat:.2f} ms")
    print(f" 95th Percentile (p95) : {p95_lat:.2f} ms")
    print(f" Real-Time Factor (RTF): {rtf:.4f} (Lower is better, <1.0 means faster than real-time)")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()