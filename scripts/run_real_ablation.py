import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
import time
import re
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from src.models.acoustic_1dcnn import Acoustic1DCNN, PhonemeTokenizer, LogMelSpectrogramTransform
from src.fusion.beam_fusion import BeamReranker
from src.routing.confidence_gate import ConfidenceGate
from jiwer import wer

def clean_txt(t: str) -> str:
    t = re.sub(r"[^A-Za-z\s']", "", t).strip().upper()
    return re.sub(r"\s+", " ", t)

def main():
    print("=" * 80)
    print(" PROJECT TIDE: REAL 16-SAMPLE 4-SEVERITY-TIER ABLATION BENCHMARK")
    print("=" * 80)

    manifest_p = Path("data/real_torgo_samples/manifest.json")
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    device = "cpu"
    print("[*] Loading faster-whisper-tiny.en on CPU...")
    whisper = WhisperModel("tiny.en", device=device, compute_type="int8")

    print("[*] Loading Neural 1D-CNN from checkpoints/best_1dcnn.pth...")
    ckpt = torch.load("checkpoints/best_1dcnn.pth", map_location=device, weights_only=False)
    acoustic_model = Acoustic1DCNN().to(device)
    acoustic_model.load_state_dict(ckpt["model_state_dict"])
    acoustic_model.eval()

    tokenizer = PhonemeTokenizer()
    mel_extractor = LogMelSpectrogramTransform()
    ctc_fn = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)

    print("[*] Loading Semantic Fluency Scorer (all-MiniLM-L6-v2)...")
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    context_prompts = [
        "A natural fluent spoken English sentence or daily request.",
        "Common everyday conversation, objects, and household activities."
    ]
    ctx_embs = semantic_model.encode(context_prompts, normalize_embeddings=True)

    gate = ConfidenceGate(threshold=0.65, min_margin=0.30)
    reranker = BeamReranker(alpha=0.4, beta=0.3, gamma=0.3)

    tiers = {"Control": [], "Mild": [], "Moderate": [], "Severe": []}

    print("\n[*] Processing real audio files...\n")
    for idx, rec in enumerate(manifest, 1):
        wav_path = rec["file"]
        ref_raw = rec["reference"]
        ref = clean_txt(ref_raw)
        sev = rec["severity"]
        spk = rec["speaker_id"]

        t0 = time.time()

        audio_np, sr = sf.read(wav_path, dtype="float32")
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)

        segments, info = whisper.transcribe(audio_np, beam_size=5, language="en")
        seg_list = list(segments)
        top1_raw = " ".join([s.text for s in seg_list]).strip() if seg_list else ""
        top1 = clean_txt(top1_raw)
        top1_logprob = seg_list[0].avg_logprob if seg_list else -5.0

        cands = [top1] if top1 else ["A"]
        for w in top1.split():
            if len(w) > 3 and len(cands) < 5:
                perturbed = top1.replace(w, w[:-1], 1)
                if perturbed not in cands:
                    cands.append(perturbed)
        while len(cands) < 5:
            cands.append(top1 + f" OPTION{len(cands)}")

        cand_logprobs = [top1_logprob - (0.35 * i) for i in range(len(cands))]

        gate_info = gate.route([{"logprob": lp} for lp in cand_logprobs])
        route = gate_info["route"]

        wav_t = torch.from_numpy(audio_np).unsqueeze(0)
        with torch.no_grad():
            mel = mel_extractor(wav_t)
            logits = acoustic_model(mel)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

        t_frames = log_probs.size(0)
        in_len = torch.tensor([t_frames], dtype=torch.long)
        neg_ctc_losses = []
        for c in cands:
            p_ids = tokenizer.encode(c)
            t_len = max(1, len(p_ids))
            tgt_t = torch.tensor([p_ids], dtype=torch.long)
            loss_val = float(ctc_fn(log_probs, tgt_t, in_len, torch.tensor([t_len])).item())
            neg_ctc_losses.append(-loss_val / t_len)

        cand_embs = semantic_model.encode(cands, normalize_embeddings=True)
        # Using numpy dot product and mean across context prompts
        sem_scores = [float(np.mean(np.dot(ctx_embs, e))) for e in cand_embs]

        if route == "FAST_PATH":
            final_out = top1
        else:
            fusion_vals = reranker.fuse(cand_logprobs, neg_ctc_losses, sem_scores)
            best_cand_idx = int(np.argmax(fusion_vals))
            final_out = cands[best_cand_idx]

        lat_ms = (time.time() - t0) * 1000
        w_base = wer(ref, top1) if top1 else 1.0
        w_tide = wer(ref, final_out) if final_out else 1.0

        tiers[sev].append({
            "spk": spk,
            "ref": ref,
            "top1": top1,
            "tide": final_out,
            "route": route,
            "w_base": w_base,
            "w_tide": w_tide,
            "lat_ms": lat_ms,
        })
        print(f"[{idx:02d}/16] [{sev:<8}] {spk} | Route: {route:<9} | Base WER: {w_base:.2f} | TIDE WER: {w_tide:.2f} | Ref: \"{ref}\"")

    print("\n" + "=" * 80)
    print(f"{'Severity Tier':<14} | {'Samples':<8} | {'Fast Path %':<12} | {'Whisper WER':<13} | {'TIDE WER':<11} | {'Avg Latency':<11}")
    print("-" * 80)
    for s_tier, items in tiers.items():
        avg_base = np.mean([x["w_base"] for x in items]) * 100
        avg_tide = np.mean([x["w_tide"] for x in items]) * 100
        fast_pct = (sum(1 for x in items if x["route"] == "FAST_PATH") / len(items)) * 100
        avg_lat = np.mean([x["lat_ms"] for x in items])
        print(f"{s_tier:<14} | {len(items):<8} | {fast_pct:>10.1f}% | {avg_base:>11.1f}% | {avg_tide:>9.1f}% | {avg_lat:>8.0f} ms")
    print("=" * 80)

if __name__ == "__main__":
    main()
