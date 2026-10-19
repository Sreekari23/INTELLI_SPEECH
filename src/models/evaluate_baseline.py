"""Baseline Evaluation & Candidate Beam Extraction Pipeline.

Tasks accomplished:
- Day 9: Severity-disaggregated WER/CER benchmarking & RTF latency profiling.
- Day 10: N-best candidate beam caching (K=5) with sequence log-probs and confidence metrics.
"""

import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import soundfile as sf
import jiwer

from src.models.baseline_whisper import WhisperBaseline


def evaluate_records(
    records: List[Dict[str, Any]],
    asr_engine: WhisperBaseline,
    beam_width: int = 1,
    max_samples: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Runs batch inference over manifest records and computes disaggregated metrics.

    Returns:
        benchmark_results: Overall and disaggregated metrics (WER, CER, RTF, latencies).
        candidate_cache: Detailed candidate beam records for Day 10/11 caching.
    """
    if max_samples:
        records = records[:max_samples]

    all_refs: List[str] = []
    all_hyps: List[str] = []
    all_durations: List[float] = []
    all_latencies: List[float] = []

    # Groupings for disaggregated metrics
    severity_buckets: Dict[str, Dict[str, List[str]]] = {
        "Control": {"refs": [], "hyps": []},
        "Mild": {"refs": [], "hyps": []},
        "Moderate": {"refs": [], "hyps": []},
        "Severe": {"refs": [], "hyps": []},
    }
    speaker_buckets: Dict[str, Dict[str, Any]] = {}

    candidate_cache: List[Dict[str, Any]] = []

    for item in records:
        ref_text = item.get("transcript", "").strip().upper()
        audio_path = item.get("raw_audio_path")
        spk = item.get("speaker_id", "UNKNOWN")
        sev = item.get("severity", "Unknown")
        item_dur = item.get("duration", 0.0)

        # Load audio (or handle missing file gracefully in test harnesses)
        if audio_path and os.path.exists(audio_path):
            audio, sr = sf.read(audio_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
        else:
            # Fallback synthetic buffer for testing
            sr = 16000
            audio = np.zeros(int(sr * max(1.0, item_dur)), dtype=np.float32)

        # Perform decoding
        if beam_width > 1:
            res = asr_engine.transcribe_beam(audio, beam_width=beam_width, sample_rate=sr)
            hyp_text = res["top1_transcript"]
            candidates = res["candidates"]
            conf_metrics = res["confidence_metrics"]
            latency = res["latency_sec"]
        else:
            res = asr_engine.transcribe_greedy(audio, sample_rate=sr)
            hyp_text = res["transcript"]
            candidates = [{"rank": 1, "transcript": hyp_text, "sequence_log_prob": 0.0}]
            conf_metrics = {"p_norm_top1": 1.0, "delta_margin": 0.0}
            latency = res["latency_sec"]

        all_refs.append(ref_text)
        all_hyps.append(hyp_text)
        all_durations.append(item_dur)
        all_latencies.append(latency)

        # Severity grouping
        if sev in severity_buckets:
            severity_buckets[sev]["refs"].append(ref_text)
            severity_buckets[sev]["hyps"].append(hyp_text)

        # Speaker grouping
        if spk not in speaker_buckets:
            speaker_buckets[spk] = {"refs": [], "hyps": [], "severity": sev}
        speaker_buckets[spk]["refs"].append(ref_text)
        speaker_buckets[spk]["hyps"].append(hyp_text)

        # Cache candidate entry for Day 10/11
        candidate_cache.append({
            "utterance_id": item.get("utterance_id", f"{spk}_{len(candidate_cache)}"),
            "speaker_id": spk,
            "severity": sev,
            "reference": ref_text,
            "top1_hypothesis": hyp_text,
            "candidates": candidates,
            "confidence_metrics": conf_metrics,
            "duration_sec": item_dur,
            "latency_sec": latency,
        })

    # Overall Metrics
    overall_wer = float(jiwer.wer(all_refs, all_hyps)) if all_refs else 0.0
    overall_cer = float(jiwer.cer(all_refs, all_hyps)) if all_refs else 0.0
    total_audio_time = sum(all_durations)
    total_proc_time = sum(all_latencies)
    rtf = round(total_proc_time / max(1e-6, total_audio_time), 4)

    # Disaggregated Severity Metrics
    severity_results: Dict[str, Any] = {}
    for sev, bucket in severity_buckets.items():
        if bucket["refs"]:
            sev_wer = round(float(jiwer.wer(bucket["refs"], bucket["hyps"])), 4)
            sev_cer = round(float(jiwer.cer(bucket["refs"], bucket["hyps"])), 4)
            severity_results[sev] = {
                "utterance_count": len(bucket["refs"]),
                "wer": sev_wer,
                "cer": sev_cer,
            }

    # Disaggregated Speaker Metrics
    speaker_results: Dict[str, Any] = {}
    for spk, bucket in sorted(speaker_buckets.items()):
        spk_wer = round(float(jiwer.wer(bucket["refs"], bucket["hyps"])), 4)
        speaker_results[spk] = {
            "severity": bucket["severity"],
            "utterance_count": len(bucket["refs"]),
            "wer": spk_wer,
        }

    benchmark_summary = {
        "model_name": asr_engine.model.name_or_path,
        "beam_width": beam_width,
        "total_evaluated": len(all_refs),
        "total_audio_duration_sec": round(total_audio_time, 2),
        "total_processing_time_sec": round(total_proc_time, 2),
        "real_time_factor_rtf": rtf,
        "latency_percentiles_sec": {
            "mean": round(float(np.mean(all_latencies)), 4) if all_latencies else 0.0,
            "p50": round(float(np.median(all_latencies)), 4) if all_latencies else 0.0,
            "p95": round(float(np.percentile(all_latencies, 95)), 4) if all_latencies else 0.0,
        },
        "overall_wer": round(overall_wer, 4),
        "overall_cer": round(overall_cer, 4),
        "severity_breakdown": severity_results,
        "speaker_breakdown": speaker_results,
    }

    return benchmark_summary, candidate_cache


def generate_benchmark_markdown(summary: Dict[str, Any]) -> str:
    """Formats benchmark results into a clean academic report."""
    md = []
    md.append(f"# Project TIDE: Baseline ASR Benchmark Report\n")
    md.append(f"* **ASR Model Backbone:** `{summary['model_name']}`")
    md.append(f"* **Decoding Strategy:** {'Beam Search (K=' + str(summary['beam_width']) + ')' if summary['beam_width'] > 1 else 'Greedy (K=1)'}")
    md.append(f"* **Total Evaluated Utterances:** {summary['total_evaluated']}")
    md.append(f"* **Real-Time Factor (RTF):** {summary['real_time_factor_rtf']} (< 0.35 target)")
    md.append(f"* **Latency (P50 / P95):** {summary['latency_percentiles_sec']['p50']}s / {summary['latency_percentiles_sec']['p95']}s\n")

    md.append("## 1. Overall System Performance\n")
    md.append("| Metric | Result | Benchmark Definition |")
    md.append("| :--- | :---: | :--- |")
    md.append(f"| **Word Error Rate (WER)** | **{summary['overall_wer'] * 100:.2f}%** | Edit distance over target reference tokens |")
    md.append(f"| **Character Error Rate (CER)** | **{summary['overall_cer'] * 100:.2f}%** | Edit distance over character sequences |")
    md.append(f"| **Real-Time Factor (RTF)** | **{summary['real_time_factor_rtf']:.3f}** | Processing Time / Audio Duration |\n")

    md.append("## 2. Clinical Severity Degradation Matrix\n")
    md.append("| Severity Bracket | Utterances | WER (%) | CER (%) | Status |")
    md.append("| :--- | :---: | :---: | :---: | :--- |")
    for sev, d in summary["severity_breakdown"].items():
        status = "Baseline Anchor" if sev == "Control" else "Degraded (Target for Hard Path)"
        md.append(f"| **{sev}** | {d['utterance_count']} | **{d['wer'] * 100:.2f}%** | {d['cer'] * 100:.2f}% | {status} |")

    md.append("\n## 3. Speaker Breakdown\n")
    md.append("| Speaker ID | Clinical Severity | Utterances | WER (%) |")
    md.append("| :--- | :---: | :---: | :---: |")
    for spk, d in summary["speaker_breakdown"].items():
        md.append(f"| `{spk}` | {d['severity']} | {d['utterance_count']} | {d['wer'] * 100:.2f}% |")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline Whisper ASR and extract beam candidates.")
    parser.add_argument("--manifest", type=str, default="configs/splits_manifest.json")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"])
    parser.add_argument("--model_name", type=str, default="openai/whisper-tiny.en")
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_report", type=str, default="reports/baseline_asr_benchmark.json")
    parser.add_argument("--output_md", type=str, default="reports/baseline_asr_benchmark.md")
    parser.add_argument("--export_candidates", type=str, default="data/manifests/test_beam_candidates.json")
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get(args.split, [])
    if not records:
        raise ValueError(f"No records found for partition '{args.split}' in {args.manifest}")

    print(f"Loading Whisper model: {args.model_name}...")
    engine = WhisperBaseline(model_name=args.model_name)

    print(f"Evaluating {len(records)} utterances (Beam Width={args.beam_width})...")
    benchmark_res, candidate_cache = evaluate_records(
        records, engine, beam_width=args.beam_width, max_samples=args.max_samples
    )

    # Save benchmark JSON
    Path(args.output_report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(benchmark_res, f, indent=2)

    # Save benchmark Markdown
    md_content = generate_benchmark_markdown(benchmark_res)
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Save Day 10 Candidate Beam Cache
    if args.export_candidates:
        Path(args.export_candidates).parent.mkdir(parents=True, exist_ok=True)
        with open(args.export_candidates, "w", encoding="utf-8") as f:
            json.dump(candidate_cache, f, indent=2)
        print(f"[DAY 10] Candidate beam cache saved to: {args.export_candidates}")

    print(f"[DAY 9] Benchmark JSON saved to: {args.output_report}")
    print(f"[DAY 9] Benchmark Markdown saved to: {args.output_md}")
    print(f"Overall WER: {benchmark_res['overall_wer'] * 100:.2f}% | RTF: {benchmark_res['real_time_factor_rtf']}")


if __name__ == "__main__":
    main()
