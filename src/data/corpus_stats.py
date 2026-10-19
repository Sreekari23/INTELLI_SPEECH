"""Corpus Statistics & Reporting Module for Project TIDE.

Computes comprehensive corpus descriptive statistics from audited manifests:
- Valid hours for Control vs Dysarthric speakers
- Duration distribution across severity brackets (Control, Mild, Moderate, Severe)
- Lexical metrics: vocabulary size, token count, average utterance duration
- Generates publication-grade Markdown and JSON summary reports.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np


def compute_corpus_statistics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyzes audited utterance records to compute descriptive corpus metrics."""
    if not records:
        return {"error": "Empty records list"}

    total_utterances = len(records)
    durations = [r["duration"] for r in records]
    total_duration_sec = sum(durations)
    total_duration_hrs = total_duration_sec / 3600.0

    # Lexical statistics
    all_words = []
    utterance_word_counts = []
    for r in records:
        words = r.get("transcript", "").split()
        all_words.extend(words)
        utterance_word_counts.append(len(words))

    vocab = set(all_words)
    total_tokens = len(all_words)

    # Demographic & Clinical Groupings
    speakers = set(r["speaker_id"] for r in records)
    severity_groups: Dict[str, List[Dict[str, Any]]] = {
        "Control": [],
        "Mild": [],
        "Moderate": [],
        "Severe": [],
    }
    speaker_stats: Dict[str, Dict[str, Any]] = {}

    for r in records:
        spk = r["speaker_id"]
        sev = r.get("severity", "Unknown")
        if sev in severity_groups:
            severity_groups[sev].append(r)

        if spk not in speaker_stats:
            speaker_stats[spk] = {
                "gender": r.get("gender", "Unknown"),
                "severity": sev,
                "dysarthric": r.get("dysarthric", False),
                "utterance_count": 0,
                "total_duration_sec": 0.0,
                "word_count": 0,
            }

        speaker_stats[spk]["utterance_count"] += 1
        speaker_stats[spk]["total_duration_sec"] += r["duration"]
        speaker_stats[spk]["word_count"] += len(r.get("transcript", "").split())

    # Format speaker stats
    for spk, data in speaker_stats.items():
        data["total_duration_min"] = round(data["total_duration_sec"] / 60.0, 2)
        data["avg_utterance_sec"] = round(data["total_duration_sec"] / max(1, data["utterance_count"]), 2)

    # Format severity summary
    severity_summary = {}
    for sev, sev_records in severity_groups.items():
        sev_durations = [x["duration"] for x in sev_records]
        sev_words = [len(x.get("transcript", "").split()) for x in sev_records]
        severity_summary[sev] = {
            "speaker_count": len(set(x["speaker_id"] for x in sev_records)),
            "utterance_count": len(sev_records),
            "duration_hours": round(sum(sev_durations) / 3600.0, 3),
            "avg_duration_sec": round(float(np.mean(sev_durations)) if sev_durations else 0.0, 2),
            "total_words": sum(sev_words),
        }

    stats = {
        "total_speakers": len(speakers),
        "total_utterances": total_utterances,
        "total_duration_hours": round(total_duration_hrs, 3),
        "total_duration_seconds": round(total_duration_sec, 2),
        "duration_distribution": {
            "mean_sec": round(float(np.mean(durations)), 2),
            "median_sec": round(float(np.median(durations)), 2),
            "std_sec": round(float(np.std(durations)), 2),
            "min_sec": round(float(np.min(durations)), 2),
            "max_sec": round(float(np.max(durations)), 2),
        },
        "lexical_metrics": {
            "total_tokens": total_tokens,
            "unique_vocabulary": len(vocab),
            "avg_words_per_utterance": round(float(np.mean(utterance_word_counts)), 2) if utterance_word_counts else 0,
        },
        "severity_summary": severity_summary,
        "speaker_breakdown": speaker_stats,
    }

    return stats


def generate_markdown_report(stats: Dict[str, Any]) -> str:
    """Renders a formatted Markdown report of the corpus statistics."""
    md = []
    md.append("# Project TIDE: TORGO Dataset Ingestion & Audit Report\n")
    md.append("## 1. Corpus High-Level Summary\n")
    md.append("| Metric | Value |")
    md.append("| :--- | :--- |")
    md.append(f"| **Total Audited Speakers** | {stats['total_speakers']} |")
    md.append(f"| **Verified Utterances (headMic)** | {stats['total_utterances']} |")
    md.append(f"| **Total Audio Duration** | {stats['total_duration_hours']} hours ({stats['total_duration_seconds']} sec) |")
    md.append(f"| **Mean Utterance Length** | {stats['duration_distribution']['mean_sec']} ± {stats['duration_distribution']['std_sec']} sec |")
    md.append(f"| **Unique Vocabulary** | {stats['lexical_metrics']['unique_vocabulary']} words ({stats['lexical_metrics']['total_tokens']} tokens) |")
    md.append(f"| **Average Words / Utterance** | {stats['lexical_metrics']['avg_words_per_utterance']} words |\n")

    md.append("## 2. Clinical Severity Breakdown\n")
    md.append("| Severity Bracket | Speakers | Utterances | Total Duration (hrs) | Avg Utterance (sec) | Total Words |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
    for sev, d in stats["severity_summary"].items():
        md.append(f"| **{sev}** | {d['speaker_count']} | {d['utterance_count']} | {d['duration_hours']} | {d['avg_duration_sec']} | {d['total_words']} |")

    md.append("\n## 3. Individual Speaker Breakdown\n")
    md.append("| Speaker ID | Gender | Diagnosis | Severity | Utterances | Duration (min) | Avg Length (sec) | Word Count |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
    for spk, d in sorted(stats["speaker_breakdown"].items()):
        diag = "Dysarthric" if d["dysarthric"] else "Control"
        md.append(f"| `{spk}` | {d['gender']} | {diag} | {d['severity']} | {d['utterance_count']} | {d['total_duration_min']} | {d['avg_utterance_sec']} | {d['word_count']} |")

    return "\n".join(md)


def generate_and_save_reports(
    manifest_path: str,
    output_json_path: str = "data/manifests/torgo_audit_report.json",
    output_md_path: str = "reports/torgo_corpus_statistics.md",
) -> Dict[str, Any]:
    """Loads manifest, computes statistics, and writes JSON & Markdown reports."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    stats = compute_corpus_statistics(records)
    md_content = generate_markdown_report(stats)

    # Save JSON
    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Save Markdown
    Path(output_md_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"[REPORT] Statistics JSON saved to: {output_json_path}")
    print(f"[REPORT] Markdown summary saved to: {output_md_path}")
    return stats
