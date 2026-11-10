import time
import queue
import numpy as np
import sounddevice as sd
import torch
import torchaudio.transforms as T
import onnxruntime as ort
from pathlib import Path

# Configuration
SAMPLE_RATE = 16000
BLOCK_DURATION_SEC = 0.25  # 250ms chunks for responsive VAD
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION_SEC)

REPO_ROOT = Path(__file__).resolve().parent.parent
model_path = REPO_ROOT / "checkpoints" / "onnx" / "squeezeformer_ctc_int8.onnx"

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
    print(f"[*] Loading INT8 ONNX model from {model_path}...")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_feat = session.get_inputs()[0].name
    input_len = session.get_inputs()[1].name

    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=400, win_length=400, hop_length=160,
        f_min=0.0, f_max=8000.0, n_mels=80, power=2.0, center=True,
        pad_mode="reflect", norm="slaney", mel_scale="slaney"
    )

    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[!] Stream Status: {status}", flush=True)
        q.put(indata.copy())

    print("\n" + "="*50)
    print("🎙️ PROJECT TIDE: VAD-GATED STREAMING INFERENCE")
    print("Speak a sentence/phrase clearly. Pause when finished.")
    print("Press Ctrl+C to stop.")
    print("="*50 + "\n")

    # State variables for Voice Activity Detection (VAD) & Utterance Collection
    speech_buffer = []
    is_speaking = False
    silence_counter = 0
    VAD_THRESHOLD = 0.015  # Energy threshold for speech vs background noise
    SILENCE_LIMIT = 4      # Number of silent blocks before triggering inference (~1 second)

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, blocksize=BLOCK_SIZE):
            while True:
                chunk = q.get().flatten()
                
                # Compute RMS Energy of chunk
                energy = np.sqrt(np.mean(chunk**2))

                if energy > VAD_THRESHOLD:
                    if not is_speaking:
                        print("[*] Speech detected... listening...", flush=True)
                        is_speaking = True
                        speech_buffer = []
                    silence_counter = 0
                    speech_buffer.append(chunk)
                else:
                    if is_speaking:
                        silence_counter += 1
                        speech_buffer.append(chunk) # Keep slight tail
                        # If silence limit reached, process the accumulated utterance
                        if silence_counter >= SILENCE_LIMIT:
                            print("[*] Pause detected. Running inference on utterance...", flush=True)
                            is_speaking = False
                            
                            # Combine utterance chunks
                            utterance = np.concatenate(speech_buffer)
                            speech_buffer = []

                            if len(utterance) > SAMPLE_RATE * 0.5: # Minimum 0.5 sec
                                waveform = torch.from_numpy(utterance).float()
                                waveform = waveform - waveform.mean()
                                waveform = waveform / torch.clamp(torch.max(torch.abs(waveform)), min=1e-7)

                                mel = torch.log(torch.clamp(mel_transform(waveform), min=1e-5))
                                mel = (mel - mel.mean(dim=-1, keepdim=True)) / mel.std(dim=-1, keepdim=True).clamp(min=1e-5)
                                
                                features = mel.transpose(0, 1).unsqueeze(0).numpy()
                                feat_lens = np.array([features.shape[1]], dtype=np.int64)

                                t0 = time.perf_counter()
                                outputs = session.run(None, {input_feat: features, input_len: feat_lens})
                                inf_time_ms = (time.perf_counter() - t0) * 1000

                                transcript = greedy_decode(outputs[0])
                                print(f"✨ [{inf_time_ms:.1f}ms] Recognized Transcript: '{transcript}'\n")

    except KeyboardInterrupt:
        print("\n[*] Stopping live audio stream. Shutting down gracefully...")

if __name__ == "__main__":
    main()
    