"""Real 4-Tier Evaluation over extracted TORGO utterances.

Runs every sample in data/real_torgo_samples/ (16 utterances: 4 per tier
Control / Mild / Moderate / Severe) through the updated pipeline:

  1. Real faster-whisper / Whisper ASR beam search (baseline Top-1 + candidates).
  2. Confidence gate (src/routing/confidence_gate): FAST vs HARD path.
  3. Hard-path reranking: AcousticScorer (1D-CNN CTC) + fluency pseudo-likelihood
     via cached distilbert-base-uncased, then beam-level Z-score fusion
     (src/fusion/beam_fusion): 0.4*ASR + 0.3*Acoustic + 0.3*Semantic.
  4. Metrics (unadjusted, real): Baseline Whisper WER vs TIDE Reranked WER,
     Candidate Correction Rate, wall-clock latency per severity tier.

Reads audio directly from the local HuggingFace parquet cache (no internet).
Writes results to data/real_torgo_eval.json and prints a per-tier table.
"""

import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import soundfile as sf
import pyarrow.parquet as pq
import glob
import torch
from jiwer import wer

from src.models.baseline_whisper import WhisperBaseline
from src.models.acoustic_1dcnn import AcousticScorer
from src.fusion.beam_fusion import compute_fusion_scores
from src.routing.confidence_gate import decide_routing

CACHE_DIR = Path(os.path.expanduser("~/.cache/huggingface/hub")) / "datasets--abnerh--TORGO-database"
EVAL_OUT = REPO_ROOT / "data" / "real_torgo_eval.json"

N_PER_TIER = 4
MIN_WORDS = 3
THRESHOLD = 0.65

TORGO_SPEAKER_SEVERITY: Dict[str, str] = {
    "FC01": "Control", "FC02": "Control", "FC03": "Control",
    "MC01": "Control", "MC02": "Control", "MC03": "Control", "MC04": "Control",
    "M05": "Mild", "F03": "Moderate", "F04": "Moderate", "M04": "Moderate",
    "F01": "Severe", "M01": "Severe", "M02": "Severe",
}
TIER_ORDER = ["Control", "Mild", "Moderate", "Severe"]


def _speaker_from_path(path: str) -> str:
    m = re.match(r"^([A-Z0-9]+)_", path.upper())
    return m.group(1) if m else ""


def _mic_from_path(path: str) -> str:
    low = path.lower()
    if "headmic" in low or "head_mic" in low:
        return "headMic"
    if "arraymic" in low or "array_mic" in low:
        return "arrayMic"
    return "unknown"


def select_real_samples() -> List[Dict[str, Any]]:
    """Select 16 real TORGO utterances (4/tier) from the offline parquet cache."""
    files = sorted(glob.glob(str(CACHE_DIR / "**" / "*.parquet"), recursive=True))
    rows: List[Dict[str, Any]] = []
    for f in files:
        table = pq.ParquetFile(f).read()
        d = table.to_pydict()
        n = table.num_rows
        for i in range(n):
            audio = d["audio"][i]
            path = audio["path"]
            if isinstance(path, list):
                path = path[0] if path else ""
            raw = audio["bytes"]
            if isinstance(raw, list):
                raw = raw[0] if raw else b""
            rows.append({
                "path": str(path),
                "mic": _mic_from_path(str(path)),
                "speaker_id": _speaker_from_path(str(path)),
                "transcript": str(d["transcription"][i] or ""),
                "speech_status": str(d["speech_status"][i] or ""),
                "gender": str(d["gender"][i] or ""),
                "duration": float(d["duration"][i] or 0.0),
                "audio_bytes": raw,
            })

    selected: List[Dict[str, Any]] = []
    for tier in TIER_ORDER:
        tier_rows: List[Dict[str, Any]] = []
        for r in rows:
            if TORGO_SPEAKER_SEVERITY.get(r["speaker_id"], "") != tier:
                continue
            if r["mic"] != "headMic":
                continue
            expected_status = "healthy" if tier == "Control" else "dysarthria"
            if r["speech_status"] != expected_status:
                continue
            words = len([w for w in r["transcript"].upper().split() if w.strip()])
            if words < MIN_WORDS:
                continue
            tier_rows.append(r)
        # Deterministic spread: one best (longest) per speaker, then top-up.
        tier_rows = sorted(tier_rows, key=lambda x: (x["speaker_id"], -x["duration"], x["path"]))
        picked: List[Dict[str, Any]] = []
        last_spk = None
        for r in tier_rows:
            if len(picked) >= N_PER_TIER:
                break
            if r["speaker_id"] != last_spk:
                picked.append(r)
                last_spk = r["speaker_id"]
        if len(picked) < N_PER_TIER:
            for r in tier_rows:
                if len(picked) >= N_PER_TIER:
                    break
                if all(x is not r for x in picked):
                    picked.append(r)
        selected.extend(picked)
        print(f"[EVAL] Tier={tier} selected {len(picked)} utterance(s)")
    return selected


def load_audio(row: Dict[str, Any]) -> Tuple[np.ndarray, int]:
    raw = row["audio_bytes"]
    arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return np.asarray(arr, dtype=np.float32), int(sr)


def main() -> None:
    samples = select_real_samples()
    print(f"[EVAL] Total utterances: {len(samples)}\n")

    asr = WhisperBaseline(model_name="openai/whisper-tiny.en", device="cpu")
    scorer = AcousticScorer(checkpoint_path=str(REPO_ROOT / "checkpoints" / "best_1dcnn.pth"), device="cpu")
    from sentence_transformers import SentenceTransformer
    sem_model = SentenceTransformer("distilbert-base-uncased")

    def semantic_score(text: str) -> float:
        contexts = [
            "A natural fluent spoken English sentence or daily request.",
            "Common everyday conversation, objects, and household activities.",
        ]
        cand_emb = sem_model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        ctx_embs = sem_model.encode(contexts, convert_to_tensor=True, normalize_embeddings=True)
        cos_sim = float((cand_emb * ctx_embs).sum(dim=-1).mean().item())
        return float(np.clip(0.5 * (cos_sim + 1.0), 0.0, 1.0))

    results: List[Dict[str, Any]] = []
    tier_buckets: Dict[str, List[Dict[str, Any]]] = {t: [] for t in TIER_ORDER}

    for row in samples:
        tier = TORGO_SPEAKER_SEVERITY[row["speaker_id"]]
        transcript = row["transcript"].upper().strip()
        arr, sr = load_audio(row)

        # ---- Baseline ASR (real Whisper beam) ----
        t0 = time.perf_counter()
        beam = asr.transcribe_beam(arr, beam_width=5, sample_rate=sr)
        asr_latency_ms = (time.perf_counter() - t0) * 1000.0

        top1 = beam["top1_transcript"]
        candidates = beam["candidates"]
        conf = beam["confidence_metrics"]

        baseline_wer = wer([transcript], [top1])

        # ---- Confidence gate ----
        gate = decide_routing(
            confidence_score=conf["p_norm_top1"],
            confidence_margin=conf["delta_margin"],
            threshold=THRESHOLD,
        )

        reranked_top1 = top1
        reranked_wer = baseline_wer
        path = gate["path"]
        fusion_detail = None

        if path == "HARD":
            all_texts = [c["transcript"] for c in candidates]
            logprobs = [c.get("sequence_log_prob", c.get("asr_logprob", 0.0)) for c in candidates]
            exp_lp = np.exp(np.array(logprobs) - np.max(logprobs))
            asr_norm = (exp_lp / np.sum(exp_lp)).tolist()

            wf = torch.from_numpy(arr).unsqueeze(0)
            acoustic_res = scorer.score_candidates(wf, list(all_texts), sample_rate=sr)
            acous_by_text = {r["transcript"]: r for r in acoustic_res}
            acous_scores = [float(acous_by_text.get(t, {}).get("acoustic_score", 0.0)) for t in all_texts]

            sem_scores = [semantic_score(t) for t in all_texts]

            fusion_detail = compute_fusion_scores(all_texts, asr_norm, acous_scores, sem_scores,
                                                  weights={"asr": 0.4, "acoustic": 0.3, "semantic": 0.3})
            reranked_top1 = fusion_detail[0]["transcript"]
            reranked_wer = wer([transcript], [reranked_top1])

        case_rec = {
            "tier": tier,
            "speaker_id": row["speaker_id"],
            "gender": row["gender"],
            "reference": transcript,
            "path": path,
            "gate_decision": gate["gate_decision"],
            "confidence_score": conf["p_norm_top1"],
            "confidence_margin": conf["delta_margin"],
            "baseline_top1": top1,
            "baseline_wer": round(float(baseline_wer), 4),
            "reranked_top1": reranked_top1,
            "reranked_wer": round(float(reranked_wer), 4),
            "wer_improved": float(reranked_wer) < float(baseline_wer),
            "baseline_correct": float(baseline_wer) == 0.0,
            "reranked_correct": float(reranked_wer) == 0.0,
            "asr_latency_ms": round(asr_latency_ms, 1),
            "num_candidates": len(candidates),
            "fusion_detail": fusion_detail,
        }
        results.append(case_rec)
        tier_buckets[tier].append(case_rec)
        print(f"[{tier}] {row['speaker_id']} | gate={path} | baseWER={baseline_wer:.3f} "
              f"| rerankWER={reranked_wer:.3f} | ref={transcript[:50]!r}")

    # ---- Aggregate per tier ----
    print("\n" + "=" * 78)
    print(f"{'Tier':<10} {'N':<4} {'BaseWER':<9} {'RerankWER':<10} {'CorrRate':<10} {'AvgLatency(ms)':<15}")
    print("-" * 78)
    overall_base = 0.0
    overall_rerank = 0.0
    overall_corr = 0
    overall_n = 0
    for tier in TIER_ORDER:
        bucket = tier_buckets[tier]
        n = len(bucket)
        if n == 0:
            continue
        base_wers = [b["baseline_wer"] for b in bucket]
        rerank_wers = [b["reranked_wer"] for b in bucket]
        # Correction rate: baseline wrong -> reranked right (WER 0)
        corrected = sum(1 for b in bucket if not b["baseline_correct"] and b["reranked_correct"])
        corr_rate = corrected / max(1, sum(1 for b in bucket if not b["baseline_correct"]))
        avg_lat = sum(b["asr_latency_ms"] for b in bucket) / n
        print(f"{tier:<10} {n:<4} {sum(base_wers)/n:<9.3f} {sum(rerank_wers)/n:<10.3f} "
              f"{corr_rate:<10.1%} {avg_lat:<15.1f}")
        overall_base += sum(base_wers)
        overall_rerank += sum(rerank_wers)
        overall_corr += corrected
        overall_n += n
    print("-" * 78)
    print(f"{'OVERALL':<10} {overall_n:<4} {overall_base/overall_n:<9.3f} "
          f"{overall_rerank/overall_n:<10.3f} {overall_corr/max(1,overall_n):<10.1%} "
          f"{sum(b['asr_latency_ms'] for b in results)/len(results):<15.1f}")

    # ---- Persist ----
    EVAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_utterances": len(results),
        "per_tier": {
            t: {
                "n": len(tier_buckets[t]),
                "mean_baseline_wer": round(sum(b["baseline_wer"] for b in tier_buckets[t]) / max(1, len(tier_buckets[t])), 4),
                "mean_reranked_wer": round(sum(b["reranked_wer"] for b in tier_buckets[t]) / max(1, len(tier_buckets[t])), 4),
            } for t in TIER_ORDER if tier_buckets[t]
        },
        "overall_mean_baseline_wer": round(overall_base / overall_n, 4),
        "overall_mean_reranked_wer": round(overall_rerank / overall_n, 4),
        "overall_correction_rate": round(overall_corr / max(1, overall_n), 4),
        "mean_asr_latency_ms": round(sum(b["asr_latency_ms"] for b in results) / len(results), 1),
    }
    with open(EVAL_OUT, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\n[EVAL] Detailed results written to: {EVAL_OUT}")


if __name__ == "__main__":
    main()
