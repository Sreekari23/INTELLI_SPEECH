"""Oracle Top-N Candidate Coverage Analysis for Project TIDE (Day 11).

Evaluates the theoretical upper bound of candidate reranking:
1. Calculates Oracle WER: The minimum achievable WER if an omniscient reranker
   always selects the best candidate transcript from the N-best beam.
2. Quantifies Reranking Headroom: Top-1 WER - Oracle WER.
3. Computes Candidate Coverage Metrics:
   - Exact match rate in candidate beam (R in {Y_1 ... Y_K})
   - Recoverable error proportion (Top-1 wrong, but better candidate exists in beam).
4. Disaggregates headroom across clinical severity brackets.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import jiwer


def compute_oracle_coverage(
    candidate_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Computes Oracle WER, headroom, and candidate availability metrics."""
    if not candidate_records:
        return {"error": "No candidate records provided"}

    all_refs: List[str] = []
    top1_hyps: List[str] = []
    oracle_hyps: List[str] = []
    
    oracle_ranks: List[int] = []
    exact_match_top1_count = 0
    exact_match_in_beam_count = 0
    recoverable_errors_count = 0
    total_utterances = len(candidate_records)

    # Disaggregated severity containers
    severity_data: Dict[str, Dict[str, List[str]]] = {
        "Control": {"refs": [], "top1": [], "oracle": []},
        "Mild": {"refs": [], "top1": [], "oracle": []},
        "Moderate": {"refs": [], "top1": [], "oracle": []},
        "Severe": {"refs": [], "top1": [], "oracle": []},
    }

    for item in candidate_records:
        ref = item.get("reference", "").strip().upper()
        candidates = item.get("candidates", [])
        sev = item.get("severity", "Unknown")

        if not candidates:
            # Fallback if no candidates
            candidates = [{"rank": 1, "transcript": item.get("top1_hypothesis", ""), "sequence_log_prob": 0.0}]

        top1_cand = candidates[0]["transcript"].strip().upper()

        # Find oracle candidate: candidate minimizing sentence-level edit distance to ref
        best_wer = float("inf")
        oracle_cand = top1_cand
        oracle_rank = 1

        cand_exact_found = False
        for c in candidates:
            c_text = c["transcript"].strip().upper()
            c_wer = jiwer.wer(ref, c_text)
            if c_text == ref:
                cand_exact_found = True

            if c_wer < best_wer:
                best_wer = c_wer
                oracle_cand = c_text
                oracle_rank = c.get("rank", 1)

        all_refs.append(ref)
        top1_hyps.append(top1_cand)
        oracle_hyps.append(oracle_cand)
        oracle_ranks.append(oracle_rank)

        # Track exact matches
        if top1_cand == ref:
            exact_match_top1_count += 1
        if cand_exact_found:
            exact_match_in_beam_count += 1

        # Track recoverable errors: Top-1 had an error, but oracle had lower WER
        top1_wer = jiwer.wer(ref, top1_cand)
        if top1_wer > 0.0 and best_wer < top1_wer:
            recoverable_errors_count += 1

        # Severity tracking
        if sev in severity_data:
            severity_data[sev]["refs"].append(ref)
            severity_data[sev]["top1"].append(top1_cand)
            severity_data[sev]["oracle"].append(oracle_cand)

    # Global WERs
    baseline_wer = float(jiwer.wer(all_refs, top1_hyps)) if all_refs else 0.0
    oracle_wer = float(jiwer.wer(all_refs, oracle_hyps)) if all_refs else 0.0
    absolute_headroom = baseline_wer - oracle_wer
    relative_headroom = (absolute_headroom / max(1e-6, baseline_wer)) * 100.0

    # Severity-disaggregated breakdown
    severity_breakdown: Dict[str, Any] = {}
    for sev, d in severity_data.items():
        if d["refs"]:
            b_wer = float(jiwer.wer(d["refs"], d["top1"]))
            o_wer = float(jiwer.wer(d["refs"], d["oracle"]))
            abs_head = b_wer - o_wer
            rel_head = (abs_head / max(1e-6, b_wer)) * 100.0 if b_wer > 0 else 0.0
            severity_breakdown[sev] = {
                "utterance_count": len(d["refs"]),
                "baseline_wer": round(b_wer, 4),
                "oracle_wer": round(o_wer, 4),
                "absolute_headroom": round(abs_head, 4),
                "relative_headroom_pct": round(rel_head, 2),
            }

    # Oracle rank distribution
    rank_counts: Dict[int, int] = {}
    for r in oracle_ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1

    rank_distribution = {
        f"rank_{k}": {
            "count": v,
            "pct": round((v / max(1, total_utterances)) * 100.0, 2)
        }
        for k, v in sorted(rank_counts.items())
    }

    results = {
        "total_utterances": total_utterances,
        "baseline_top1_wer": round(baseline_wer, 4),
        "oracle_top_n_wer": round(oracle_wer, 4),
        "absolute_reranking_headroom": round(absolute_headroom, 4),
        "relative_reranking_headroom_pct": round(relative_headroom, 2),
        "exact_match_top1_pct": round((exact_match_top1_count / max(1, total_utterances)) * 100.0, 2),
        "exact_match_in_beam_pct": round((exact_match_in_beam_count / max(1, total_utterances)) * 100.0, 2),
        "recoverable_error_count": recoverable_errors_count,
        "recoverable_error_pct": round((recoverable_errors_count / max(1, total_utterances)) * 100.0, 2),
        "oracle_rank_distribution": rank_distribution,
        "severity_breakdown": severity_breakdown,
    }

    return results


def generate_oracle_markdown(results: Dict[str, Any]) -> str:
    """Renders formatted Markdown report of the Oracle coverage evaluation."""
    md = []
    md.append("# Project TIDE: Oracle Top-N Candidate Coverage Report (Day 11)\n")
    md.append("> **Scientific Purpose:** Establishes the theoretical upper-bound error reduction")
    md.append("> available to secondary acoustic and semantic rerankers from the candidate beam.\n")

    md.append("## 1. Executive Summary & Reranking Headroom\n")
    md.append("| Metric | Value | Architectural Interpretation |")
    md.append("| :--- | :---: | :--- |")
    md.append(f"| **Baseline Top-1 WER** | **{results['baseline_top1_wer'] * 100:.2f}%** | Standard ASR linguistic benchmark |")
    md.append(f"| **Oracle Top-N WER** | **{results['oracle_top_n_wer'] * 100:.2f}%** | Theoretical limit if ideal candidate is chosen |")
    md.append(f"| **Absolute Reranking Headroom** | **{results['absolute_reranking_headroom'] * 100:.2f}% WER** | Max possible WER reduction from reranking |")
    md.append(f"| **Relative Headroom Potential** | **{results['relative_reranking_headroom_pct']:.2f}%** | Theoretical error elimination potential |")
    md.append(f"| **Exact Match in Beam** | **{results['exact_match_in_beam_pct']:.2f}%** | Utterances with 100% correct transcript in beam |")
    md.append(f"| **Recoverable Error Utterances** | **{results['recoverable_error_pct']:.2f}%** | Utterances where Top-1 erred but beam has better candidate |\n")

    md.append("## 2. Clinical Severity Reranking Headroom Matrix\n")
    md.append("| Severity Bracket | Utterances | Top-1 WER (%) | Oracle WER (%) | Absolute Headroom (%) | Relative Headroom (%) |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
    for sev, d in results.get("severity_breakdown", {}).items():
        md.append(f"| **{sev}** | {d['utterance_count']} | {d['baseline_wer'] * 100:.2f}% | {d['oracle_wer'] * 100:.2f}% | **{d['absolute_headroom'] * 100:.2f}%** | **{d['relative_headroom_pct']:.2f}%** |")

    md.append("\n## 3. Oracle Candidate Rank Distribution\n")
    md.append("| Candidate Rank | Chosen as Oracle (Count) | Percentage (%) |")
    md.append("| :--- | :---: | :---: |\n")
    for rank_key, d in results.get("oracle_rank_distribution", {}).items():
        rank_num = rank_key.replace("rank_", "Rank ")
        md.append(f"| `{rank_num}` | {d['count']} | {d['pct']:.2f}% |")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Analyze Oracle Top-N coverage from cached candidate beams.")
    parser.add_argument("--candidates_path", type=str, default="data/manifests/test_beam_candidates.json")
    parser.add_argument("--output_report", type=str, default="reports/oracle_candidate_coverage.json")
    parser.add_argument("--output_md", type=str, default="reports/oracle_candidate_coverage.md")
    args = parser.parse_args()

    cand_p = Path(args.candidates_path)
    if not cand_p.exists():
        raise FileNotFoundError(f"Candidate cache not found: {args.candidates_path}")

    with open(cand_p, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"Loaded {len(records)} candidate beam records. Computing Oracle coverage...")
    results = compute_oracle_coverage(records)

    # Save JSON
    Path(args.output_report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save Markdown
    md_content = generate_oracle_markdown(results)
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"[DAY 11] Oracle JSON report saved to: {args.output_report}")
    print(f"[DAY 11] Oracle Markdown report saved to: {args.output_md}")
    print(f"Baseline WER: {results['baseline_top1_wer'] * 100:.2f}% -> Oracle WER: {results['oracle_top_n_wer'] * 100:.2f}%")
    print(f"Absolute Headroom: {results['absolute_reranking_headroom'] * 100:.2f}% WER points.")


if __name__ == "__main__":
    main()
