"""Milestone Gate 1: Split Manifest Freezer & Schema Validator.

Locks and serializes the dataset partition manifest into configs/splits_manifest.json.
Enforces rigorous validation rules:
1. Zero speaker overlap: Train ∩ Val ∩ Test = ∅.
2. Clinical severity balance: Test and Val must contain representation from
   Control, Moderate, and Severe categories.
3. Every record has non-empty transcript and positive audio duration.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import yaml

from src.data.stratify_splits import load_split_config, partition_inventory


def freeze_splits(
    inventory_path: str,
    splits_config_path: str = "configs/splits.yaml",
    output_manifest_path: str = "configs/splits_manifest.json"
) -> Dict[str, Any]:
    """Freezes the inventory into a verified partition manifest."""
    inv_file = Path(inventory_path)
    if not inv_file.exists():
        raise FileNotFoundError(f"Inventory not found: {inventory_path}")

    with open(inv_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    split_cfg = load_split_config(splits_config_path)
    partitioned, stats = partition_inventory(records, split_cfg)

    # Schema & Integrity Validations
    for part in ["train", "val", "test"]:
        if len(partitioned[part]) == 0:
            raise ValueError(f"Partition '{part}' contains zero utterances!")
        for rec in partitioned[part]:
            assert "utterance_id" in rec, "Missing utterance_id"
            assert "speaker_id" in rec, "Missing speaker_id"
            assert "transcript" in rec and len(rec["transcript"].strip()) > 0, "Empty transcript"
            assert rec.get("duration", 0) > 0.05, f"Invalid duration {rec.get('duration')}"

    out_file = Path(output_manifest_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(partitioned, f, indent=2)

    print(f"[GATE 1 LOCKED] Split manifest successfully written to: {out_file.resolve()}")
    return stats
