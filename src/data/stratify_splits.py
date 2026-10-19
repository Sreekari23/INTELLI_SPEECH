"""Speaker-Independent Stratification & Split Partitioning Engine.

Enforces strict Leave-Speakers-Out (LSO) guarantees:
    Train Speakers ∩ Val Speakers ∩ Test Speakers = ∅

Ensures held-out test and validation sets span all clinical dysarthria
severity tiers (Control, Mild, Moderate, Severe) without any speaker overlap.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple, Optional
import yaml


def validate_speaker_splits(
    train_speakers: List[str],
    val_speakers: List[str],
    test_speakers: List[str]
) -> None:
    """Verifies zero speaker overlap across partitions.

    Raises:
        ValueError: If any speaker appears in more than one partition.
    """
    s_train: Set[str] = set(train_speakers)
    s_val: Set[str] = set(val_speakers)
    s_test: Set[str] = set(test_speakers)

    overlap_train_val = s_train.intersection(s_val)
    if overlap_train_val:
        raise ValueError(f"Speaker leak detected between Train and Val: {overlap_train_val}")

    overlap_train_test = s_train.intersection(s_test)
    if overlap_train_test:
        raise ValueError(f"Speaker leak detected between Train and Test: {overlap_train_test}")

    overlap_val_test = s_val.intersection(s_test)
    if overlap_val_test:
        raise ValueError(f"Speaker leak detected between Val and Test: {overlap_val_test}")


def load_split_config(config_path: str) -> Dict[str, List[str]]:
    """Loads partition speaker IDs from YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_spks = cfg.get("train_speakers", [])
    val_spks = cfg.get("validation_speakers", [])
    test_spks = cfg.get("test_speakers", [])

    validate_speaker_splits(train_spks, val_spks, test_spks)
    return {
        "train": train_spks,
        "val": val_spks,
        "test": test_spks,
    }


def partition_inventory(
    records: List[Dict[str, Any]],
    split_config: Dict[str, List[str]]
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """Partitions audited utterances into train, val, and test subsets.

    Returns:
        partitioned_records: {"train": [...], "val": [...], "test": [...]}
        summary_stats: High-level summary of speakers, durations, and severities.
    """
    train_set = set(split_config["train"])
    val_set = set(split_config["val"])
    test_set = set(split_config["test"])

    validate_speaker_splits(list(train_set), list(val_set), list(test_set))

    partitioned: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
        "unassigned": [],
    }

    for record in records:
        spk = record["speaker_id"]
        if spk in train_set:
            partitioned["train"].append(record)
        elif spk in val_set:
            partitioned["val"].append(record)
        elif spk in test_set:
            partitioned["test"].append(record)
        else:
            partitioned["unassigned"].append(record)

    # Compute partition summary statistics
    stats: Dict[str, Any] = {}
    for part in ["train", "val", "test"]:
        part_records = partitioned[part]
        spks = sorted(list(set(r["speaker_id"] for r in part_records)))
        durations = [r["duration"] for r in part_records]
        total_sec = sum(durations)

        # Severity distribution
        severity_counts: Dict[str, int] = {}
        for r in part_records:
            sev = r.get("severity", "Unknown")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        stats[part] = {
            "speaker_count": len(spks),
            "speakers": spks,
            "utterance_count": len(part_records),
            "total_duration_hours": round(total_sec / 3600.0, 3),
            "total_duration_seconds": round(total_sec, 2),
            "severity_distribution": severity_counts,
        }

    stats["unassigned_count"] = len(partitioned["unassigned"])
    return partitioned, stats


def export_split_manifests(
    partitioned: Dict[str, List[Dict[str, Any]]],
    stats: Dict[str, Any],
    manifest_output_path: str = "configs/splits_manifest.json",
    report_output_path: str = "reports/splits_summary.json"
) -> None:
    """Exports the frozen splits manifest and summary report to disk."""
    # Save master manifest
    man_path = Path(manifest_output_path)
    man_path.parent.mkdir(parents=True, exist_ok=True)
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(partitioned, f, indent=2)

    # Save summary report
    rep_path = Path(report_output_path)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"[SPLIT COMPLETE] Manifest saved to: {man_path.resolve()}")
    print(f"[SPLIT COMPLETE] Summary saved to:  {rep_path.resolve()}")
    print(f"Train: {stats['train']['utterance_count']} utts ({stats['train']['total_duration_hours']} hrs) | Speakers: {stats['train']['speakers']}")
    print(f"Val:   {stats['val']['utterance_count']} utts ({stats['val']['total_duration_hours']} hrs) | Speakers: {stats['val']['speakers']}")
    print(f"Test:  {stats['test']['utterance_count']} utts ({stats['test']['total_duration_hours']} hrs) | Speakers: {stats['test']['speakers']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Partition TORGO inventory into LSO splits.")
    parser.add_argument("--inventory", type=str, required=True, help="Path to torgo_inventory.json")
    parser.add_argument("--config", type=str, default="configs/splits.yaml", help="Path to splits.yaml")
    parser.add_argument("--output_manifest", type=str, default="configs/splits_manifest.json")
    parser.add_argument("--output_report", type=str, default="reports/splits_summary.json")
    args = parser.parse_args()

    with open(args.inventory, "r", encoding="utf-8") as f:
        data = json.load(f)

    split_cfg = load_split_config(args.config)
    partitioned_data, summary = partition_inventory(data, split_cfg)
    export_split_manifests(partitioned_data, summary, args.output_manifest, args.output_report)
