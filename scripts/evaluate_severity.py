import os, glob, json, time
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
    collapsed, prev_idx = [], -1
    for idx in preds:
        if idx != prev_idx and idx != 0:
            collapsed.append(idx2char[idx])
        prev_idx = idx
    return "".join(collapsed).strip()

def calculate_cer(reference, hypothesis):
    if len(reference) == 0: return len(hypothesis)
    d = [[i if j == 0 else 0 for j in range(len(hypothesis) + 1)] for i in range(len(reference) + 1)]
    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            if reference[i-1] == hypothesis[j-1]: d[i][j] = d[i-1][j-1]
            else: d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
    return d[len(reference)][len(hypothesis)] / len(reference)

def main():
    REPO_ROOT = Path(__file__).resolve().parent.parent
    model_path = REPO_ROOT / "checkpoints" / "onnx" / "squeezeformer_ctc_int8.onnx"
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    
    with open(glob.glob(str(REPO_ROOT / "**/val/manifest.json"), recursive=True)[0], "r", encoding="utf-8") as f:
        records = json.load(f)
    records = records.get("samples", records.get("cases", [])) if isinstance(records, dict) else records
    base_dir = os.path.dirname(glob.glob(str(REPO_ROOT / "**/val/manifest.json"), recursive=True)[0])

    mel_transform = T.MelSpectrogram(sample_rate=16000, n_fft=400, win_length=400, hop_length=160, f_min=0.0, f_max=8000.0, n_mels=80, power=2.0, center=True, pad_mode="reflect", norm="slaney", mel_scale="slaney")
    input_feat, input_len = session.get_inputs()[0].name, session.get_inputs()[1].name

    results = {"Control (Healthy)": {"lat": [], "cer": []}, "Dysarthric (Impaired)": {"lat": [], "cer": []}}

    for rec in records[:30]:  # Test a larger batch
        wav_path = os.path.join(base_dir, rec.get("file") or rec.get("audio_filepath", ""))
        if not os.path.exists(wav_path): wav_path = rec.get("file") or rec.get("audio_filepath", "")
        if not os.path.exists(wav_path): continue
        
        # Determine Severity based on TORGO naming conventions (FC/MC = Control, F/M = Dysarthric)
        filename = os.path.basename(wav_path)
        group = "Control (Healthy)" if filename.startswith(("FC", "MC")) else "Dysarthric (Impaired)"

        audio_np, sr = sf.read(wav_path, dtype="float32")
        if audio_np.ndim > 1: audio_np = audio_np.mean(axis=1)
        waveform = torch.from_numpy(audio_np).float()
        if sr != 16000: waveform = T.Resample(orig_freq=sr, new_freq=16000)(waveform)
        waveform = waveform / torch.clamp(torch.max(torch.abs(waveform)), min=1e-7)
        
        mel = torch.log(torch.clamp(mel_transform(waveform), min=1e-5))
        mel = (mel - mel.mean(dim=-1, keepdim=True)) / mel.std(dim=-1, keepdim=True).clamp(min=1e-5)
        features, feat_lens = mel.transpose(0, 1).unsqueeze(0).numpy(), np.array([mel.shape[1]], dtype=np.int64)

        t0 = time.perf_counter()
        outputs = session.run(None, {input_feat: features, input_len: feat_lens})
        lat_ms = (time.perf_counter() - t0) * 1000
        
        ref_text = (rec.get("text") or rec.get("transcript", "")).lower()
        pred_text = greedy_decode(outputs[0]).lower()
        
        results[group]["lat"].append(lat_ms)
        results[group]["cer"].append(calculate_cer(ref_text, pred_text))

    print("\n📊 SEVERITY-WISE RESEARCH METRICS:")
    for group, data in results.items():
        if data["lat"]:
            print(f"\n--- {group} ---")
            print(f" • Samples Evaluated : {len(data['lat'])}")
            print(f" • Average Latency   : {np.mean(data['lat']):.2f} ms")
            print(f" • Average CER       : {np.mean(data['cer'])*100:.2f}%")

if __name__ == "__main__":
    main()