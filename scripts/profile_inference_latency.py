import os
import sys
import time
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import torchaudio.transforms as T

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.frontend.mel_extractor import LogMelFeatureExtractor
from src.models.squeezeformer_ctc import SqueezeformerCTC, VOCAB
from src.models.acoustic_1dcnn import AcousticScorer
from src.routing.confidence_gate import decide_routing
from src.fusion.beam_fusion import compute_fusion_scores
from sentence_transformers import SentenceTransformer


class InferenceProfiler:
    def __init__(self, device="cpu"):
        self.device = device
        print(f"[*] Loading models to {self.device} for profiling...")

        self.mel_extractor = LogMelFeatureExtractor()

        print("[*] Loading SqueezeformerCTC (Non-Autoregressive Edge ASR)...")
        self.asr = SqueezeformerCTC(input_dim=80, num_classes=len(VOCAB), encoder_dim=256, num_layers=8).to(self.device)
        ckpt_path = REPO_ROOT / "checkpoints" / "best_squeezeformer_ctc.pth"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.asr.load_state_dict(ckpt["model_state_dict"])
            print(f"    [+] Loaded checkpoint from {ckpt_path.name}")
        self.asr.eval()

        self.scorer = AcousticScorer(checkpoint_path=REPO_ROOT / "checkpoints" / "best_1dcnn_v2.pth", device=device)
        self.semantic_model = SentenceTransformer("distilbert-base-uncased", device=device)

        self.context_prompts = [
            "A natural fluent spoken English sentence or daily request.",
            "Common everyday conversation, objects, and household activities."
        ]
        self.ctx_embs = self.semantic_model.encode(self.context_prompts, convert_to_tensor=True, normalize_embeddings=True)

        self._warmup()

    def _warmup(self):
        print("[*] Warming up neural network kernels...")
        dummy_wav = torch.zeros(1, 16000 * 2).to(self.device)
        dummy_mel = torch.zeros(1, 200, 80).to(self.device)
        dummy_len = torch.LongTensor([200]).to(self.device)
        with torch.no_grad():
            logits, out_l = self.asr(dummy_mel, dummy_len)
            _ = self.asr.decode_beam(logits, out_l, beam_width=5)
        _ = self.scorer.score_candidates(dummy_wav, ["test candidate"], sample_rate=16000)
        _ = self.semantic_model.encode("test candidate", convert_to_tensor=True)
        print("[+] Warmup complete.\n")

    def profile_file(self, wav_path: str):
        stats = {}
        print(f"--- Profiling E2E Latency: {Path(wav_path).name} ---")

        # 1. Audio Preprocessing + 80-channel Log-Mel CMVN
        t_start = time.perf_counter()
        audio_np, sr = sf.read(wav_path, dtype="float32")
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        waveform = torch.from_numpy(np.asarray(audio_np, dtype=np.float32)).unsqueeze(0)
        if sr != 16000:
            waveform = T.Resample(orig_freq=sr, new_freq=16000)(waveform)
        waveform = waveform.to(self.device)

        mel = self.mel_extractor(waveform).squeeze(0)
        mean = mel.mean(dim=-1, keepdim=True)
        std = mel.std(dim=-1, keepdim=True).clamp(min=1e-5)
        mel = ((mel - mean) / std).transpose(0, 1).unsqueeze(0).contiguous()
        feat_len = torch.LongTensor([mel.size(1)]).to(self.device)

        t_audio = time.perf_counter()
        stats['1_Audio_Mel_Preproc'] = (t_audio - t_start) * 1000

        # 2. Squeezeformer CTC Forward Pass + 5-Beam Prefix Decode
        t_asr_start = time.perf_counter()
        with torch.no_grad():
            logits, out_lens = self.asr(mel, feat_len)
            asr_out = self.asr.decode_beam(logits, out_lens, beam_width=5)

        candidates = [c["transcript"] if c["transcript"].strip() else "a" for c in asr_out["candidates"]]
        logprobs = [c["sequence_log_prob"] for c in asr_out["candidates"]]
        conf_score = asr_out["confidence_metrics"]["p_norm_top1"]
        conf_margin = asr_out["confidence_metrics"]["delta_margin"]

        t_asr_end = time.perf_counter()
        stats['2_Squeezeformer_CTC'] = (t_asr_end - t_asr_start) * 1000

        # 3. Confidence Gate
        t_gate_start = time.perf_counter()
        gate = decide_routing(confidence_score=conf_score, confidence_margin=conf_margin, threshold=0.99)
        fast_path = gate["path"] == "FAST"
        t_gate_end = time.perf_counter()
        stats['3_Confidence_Gate'] = (t_gate_end - t_gate_start) * 1000

        if fast_path:
            stats['Total_Fast_Path'] = sum(stats.values())
            self._print_stats(stats, "FAST PATH")
            return

        exp_lp = np.exp(np.array(logprobs) - np.max(logprobs))
        asr_norm_scores = exp_lp / np.sum(exp_lp)

        # 4. Acoustic 1D-CNN Head
        t_acous_start = time.perf_counter()
        acoustic_res = self.scorer.score_candidates(waveform, candidates, sample_rate=16000)
        acoustic_by_text = {r["transcript"]: r for r in acoustic_res}
        acous_norm_scores = [float(acoustic_by_text.get(c, {"acoustic_score": 0.0})["acoustic_score"]) for c in candidates]
        t_acous_end = time.perf_counter()
        stats['4_Acoustic_1DCNN'] = (t_acous_end - t_acous_start) * 1000

        # 5. Semantic DistilBERT Head (Batched Encoding Optimization)
        t_sem_start = time.perf_counter()
        cand_embs = self.semantic_model.encode(candidates, convert_to_tensor=True, normalize_embeddings=True)
        cos_sims = torch.matmul(cand_embs, self.ctx_embs.T).mean(dim=-1).cpu().numpy()
        sem_scores = [round(float(np.clip(0.5 * (cs + 1.0), 0.0, 1.0)), 3) for cs in cos_sims]
        t_sem_end = time.perf_counter()
        stats['5_Semantic_DistilBERT'] = (t_sem_end - t_sem_start) * 1000

        # 6. Linear Z-Score Fusion
        t_fus_start = time.perf_counter()
        fusion_results = compute_fusion_scores(
            transcripts=candidates,
            asr_norm_scores=list(asr_norm_scores),
            acoustic_scores=acous_norm_scores,
            semantic_scores=sem_scores,
            weights={"asr": 0.4, "acoustic": 0.3, "semantic": 0.3}
        )
        t_fus_end = time.perf_counter()
        stats['6_Fusion'] = (t_fus_end - t_fus_start) * 1000

        stats['Total_Hard_Path'] = sum(stats.values())
        self._print_stats(stats, "HARD PATH")

    def _print_stats(self, stats, path_name):
        print(f"\n{path_name} Latency Breakdown (Target: < 1300ms on Pi 5 | Old Whisper E2E: 2287.26ms)")
        print("-" * 65)
        for k, v in stats.items():
            prefix = ">> " if "Total" in k else "   "
            print(f"{prefix}{k.replace('_', ' '):<25}: {v:>8.2f} ms")
        print("-" * 65 + "\n")

if __name__ == "__main__":
    profiler = InferenceProfiler(device="cpu")
    test_wav = str(REPO_ROOT / "data" / "sample_audio" / "test_sample.wav")
    if os.path.exists(test_wav):
        profiler.profile_file(test_wav)
    else:
        print(f"Please place a test .wav file at {test_wav} to run the profiler.")
