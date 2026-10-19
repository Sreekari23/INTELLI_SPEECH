"""Dysarthric Speech Error Taxonomy & Phonetic Analysis (Day 12).

Classifies and quantifies standard ASR failure modes across clinical dysarthria:
1. Vowel Centralization / Distortion: Vowel substitutions (e.g. DOG -> DAG, PAN -> PEN).
2. Consonant Deletion / Omission: Loss of stops, fricatives, or cluster elements.
3. Word Boundary / Pause Truncation: Utterances truncated due to atypical pause durations.
4. Word Length Degradation: Error distribution on short monosyllabic vs polysyllabic words.
5. Severity Stratification: Correlating neuromotor impairment severity with error types.
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import jiwer


VOWELS = set("AEIOU")


def is_vowel_substitution(ref_word: str, hyp_word: str) -> bool:
    """Detects if two single-syllable or short words differ primarily by a vowel."""
    if len(ref_word) != len(hyp_word) or len(ref_word) < 2:
        return False

    diff_indices = [i for i in range(len(ref_word)) if ref_word[i] != hyp_word[i]]
    if len(diff_indices) == 1:
        idx = diff_indices[0]
        # Check if the single difference is a vowel substitution
        if ref_word[idx] in VOWELS or hyp_word[idx] in VOWELS:
            return True
    return False


def is_consonant_omission(ref_word: str, hyp_word: str) -> bool:
    """Detects consonant omissions/truncations (e.g., COLD -> COL, FAST -> FAS, SPIN -> PIN)."""
    if len(hyp_word) >= len(ref_word) or len(hyp_word) < 2:
        return False

    # Check suffix omission (e.g. COLD -> COL, FAST -> FAS)
    if ref_word.startswith(hyp_word):
        dropped_chars = ref_word[len(hyp_word):]
        if any(c not in VOWELS for c in dropped_chars):
            return True

    # Check prefix omission (e.g. SPIN -> PIN, STOP -> TOP)
    if ref_word.endswith(hyp_word):
        dropped_chars = ref_word[:len(ref_word) - len(hyp_word)]
        if any(c not in VOWELS for c in dropped_chars):
            return True

    # Check single internal consonant drop (e.g. STAMP -> STAP)
    if len(hyp_word) == len(ref_word) - 1:
        for i in range(len(ref_word)):
            if ref_word[i] not in VOWELS:
                if ref_word[:i] + ref_word[i+1:] == hyp_word:
                    return True

    return False


def analyze_utterance_errors(ref: str, hyp: str) -> Dict[str, Any]:
    """Analyzes a single reference-hypothesis pair for linguistic error categories."""
    ref_clean = re.sub(r"[^A-Z\s]", "", ref.upper()).split()
    hyp_clean = re.sub(r"[^A-Z\s]", "", hyp.upper()).split()

    vowel_distortions = 0
    consonant_omissions = 0
    word_substitutions = 0
    word_deletions = 0
    word_insertions = 0

    # Boundary truncation check: hypothesis is substantially shorter than reference
    is_boundary_truncated = False
    if len(ref_clean) >= 4 and len(hyp_clean) <= len(ref_clean) // 2:
        is_boundary_truncated = True

    # Word-level alignment
    len_ref = len(ref_clean)
    len_hyp = len(hyp_clean)

    for i in range(min(len_ref, len_hyp)):
        r_w = ref_clean[i]
        h_w = hyp_clean[i]
        if r_w != h_w:
            word_substitutions += 1
            if is_vowel_substitution(r_w, h_w):
                vowel_distortions += 1
            elif is_consonant_omission(r_w, h_w):
                consonant_omissions += 1

    if len_ref > len_hyp:
        word_deletions += (len_ref - len_hyp)
    elif len_hyp > len_ref:
        word_insertions += (len_hyp - len_ref)

    return {
        "ref_word_count": len_ref,
        "hyp_word_count": len_hyp,
        "word_substitutions": word_substitutions,
        "word_deletions": word_deletions,
        "word_insertions": word_insertions,
        "vowel_distortions": vowel_distortions,
        "consonant_omissions": consonant_omissions,
        "boundary_truncated": is_boundary_truncated,
    }


def compile_corpus_error_taxonomy(
    records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Aggregates error taxonomy statistics across all audited candidate records."""
    total_utts = len(records)
    total_ref_words = 0
    total_subs = 0
    total_dels = 0
    total_ins = 0
    total_vowel_dist = 0
    total_cons_omiss = 0
    total_truncations = 0

    severity_taxonomy: Dict[str, Dict[str, Any]] = {
        "Control": {"utts": 0, "subs": 0, "dels": 0, "ins": 0, "vowel_dist": 0, "cons_omiss": 0, "trunc": 0},
        "Mild": {"utts": 0, "subs": 0, "dels": 0, "ins": 0, "vowel_dist": 0, "cons_omiss": 0, "trunc": 0},
        "Moderate": {"utts": 0, "subs": 0, "dels": 0, "ins": 0, "vowel_dist": 0, "cons_omiss": 0, "trunc": 0},
        "Severe": {"utts": 0, "subs": 0, "dels": 0, "ins": 0, "vowel_dist": 0, "cons_omiss": 0, "trunc": 0},
    }

    for item in records:
        ref = item.get("reference", "")
        hyp = item.get("top1_hypothesis", "")
        sev = item.get("severity", "Unknown")

        analysis = analyze_utterance_errors(ref, hyp)

        total_ref_words += analysis["ref_word_count"]
        total_subs += analysis["word_substitutions"]
        total_dels += analysis["word_deletions"]
        total_ins += analysis["word_insertions"]
        total_vowel_dist += analysis["vowel_distortions"]
        total_cons_omiss += analysis["consonant_omissions"]
        if analysis["boundary_truncated"]:
            total_truncations += 1

        if sev in severity_taxonomy:
            bucket = severity_taxonomy[sev]
            bucket["utts"] += 1
            bucket["subs"] += analysis["word_substitutions"]
            bucket["dels"] += analysis["word_deletions"]
            bucket["ins"] += analysis["word_insertions"]
            bucket["vowel_dist"] += analysis["vowel_distortions"]
            bucket["cons_omiss"] += analysis["consonant_omissions"]
            if analysis["boundary_truncated"]:
                bucket["trunc"] += 1

    total_errors = total_subs + total_dels + total_ins
    results = {
        "total_evaluated_utterances": total_utts,
        "total_reference_words": total_ref_words,
        "error_operation_breakdown": {
            "substitutions": total_subs,
            "deletions": total_dels,
            "insertions": total_ins,
            "total_errors": total_errors,
        },
        "dysarthric_acoustic_categories": {
            "vowel_centralization_distortions": total_vowel_dist,
            "vowel_distortion_pct_of_subs": round((total_vowel_dist / max(1, total_subs)) * 100.0, 2),
            "consonant_cluster_omissions": total_cons_omiss,
            "consonant_omission_pct_of_subs": round((total_cons_omiss / max(1, total_subs)) * 100.0, 2),
            "boundary_pause_truncations": total_truncations,
            "boundary_truncation_pct_of_utts": round((total_truncations / max(1, total_utts)) * 100.0, 2),
        },
        "severity_stratified_taxonomy": severity_taxonomy,
    }

    return results


def generate_taxonomy_markdown(results: Dict[str, Any]) -> str:
    """Renders formatted Markdown report of the Dysarthric Error Taxonomy."""
    md = []
    md.append("# Project TIDE: Dysarthric Error Taxonomy & Linguistic Analysis (Day 12)\n")
    md.append("> **Scientific Motivation:** Isolates specific neuromotor speech failure patterns")
    md.append("> to validate why secondary acoustic CTC and semantic LM heads are necessary.\n")

    md.append("## 1. Classical Speech Error Operations\n")
    ops = results["error_operation_breakdown"]
    md.append("| Edit Operation | Occurrences | Percentage of Total Errors |")
    md.append("| :--- | :---: | :---: |\n")
    tot = max(1, ops["total_errors"])
    md.append(f"| **Substitutions (S)** | {ops['substitutions']} | {ops['substitutions'] / tot * 100:.2f}% |")
    md.append(f"| **Deletions (D)** | {ops['deletions']} | {ops['deletions'] / tot * 100:.2f}% |")
    md.append(f"| **Insertions (I)** | {ops['insertions']} | {ops['insertions'] / tot * 100:.2f}% |")
    md.append(f"| **Total Edit Operations** | **{ops['total_errors']}** | 100.00% |\n")

    md.append("## 2. Dysarthria-Specific Acoustic Error Patterns\n")
    acous = results["dysarthric_acoustic_categories"]
    md.append("| Acoustic Error Phenomenon | Count | Proportion | Remediation in TIDE Architecture |")
    md.append("| :--- | :---: | :---: | :--- |")
    md.append(f"| **Vowel Centralization / Distortion** | {acous['vowel_centralization_distortions']} | {acous['vowel_distortion_pct_of_subs']:.2f}% of substitutions | **Semantic LM:** Evaluates contextual word plausibility (e.g. DOG vs DAG). |")
    md.append(f"| **Consonant Omission / Weak Stops** | {acous['consonant_cluster_omissions']} | {acous['consonant_omission_pct_of_subs']:.2f}% of substitutions | **Acoustic Head:** 1D-CNN CTC phone posterior alignment. |")
    md.append(f"| **Boundary Pause Truncations** | {acous['boundary_pause_truncations']} | {acous['boundary_truncation_pct_of_utts']:.2f}% of utterances | **Dysarthric VAD:** 800–1200ms silence hangover buffering. |\n")

    md.append("## 3. Clinical Severity vs Error Distribution\n")
    md.append("| Severity Bracket | Utterances | Substitutions | Deletions | Vowel Distortions | Consonant Omissions | Truncations |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
    for sev, d in results.get("severity_stratified_taxonomy", {}).items():
        md.append(f"| **{sev}** | {d['utts']} | {d['subs']} | {d['dels']} | {d['vowel_dist']} | {d['cons_omiss']} | {d['trunc']} |")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Compile dysarthric error taxonomy from candidate predictions.")
    parser.add_argument("--candidates_path", type=str, default="data/manifests/test_beam_candidates.json")
    parser.add_argument("--output_report", type=str, default="reports/dysarthric_error_taxonomy.json")
    parser.add_argument("--output_md", type=str, default="reports/dysarthric_error_taxonomy.md")
    args = parser.parse_args()

    cand_p = Path(args.candidates_path)
    if not cand_p.exists():
        raise FileNotFoundError(f"Candidate cache not found: {args.candidates_path}")

    with open(cand_p, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"Analyzing {len(records)} candidate records for dysarthric error patterns...")
    results = compile_corpus_error_taxonomy(records)

    # Save JSON
    Path(args.output_report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save Markdown
    md_content = generate_taxonomy_markdown(results)
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"[DAY 12] Error Taxonomy JSON saved to: {args.output_report}")
    print(f"[DAY 12] Error Taxonomy Markdown saved to: {args.output_md}")


if __name__ == "__main__":
    main()
