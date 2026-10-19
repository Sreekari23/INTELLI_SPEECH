"""Unit tests for speaker-independent stratification and corpus statistics."""

import json
import pytest
from src.data.stratify_splits import (
    validate_speaker_splits,
    load_split_config,
    partition_inventory,
)
from src.data.corpus_stats import (
    compute_corpus_statistics,
    generate_markdown_report,
)


@pytest.fixture
def mock_inventory():
    """Generates synthetic audited utterance records across various speakers."""
    records = []
    # Test speakers
    records.append({"utterance_id": "u1", "speaker_id": "M01", "severity": "Severe", "dysarthric": True, "duration": 2.5, "transcript": "THE DOG BARKED"})
    records.append({"utterance_id": "u2", "speaker_id": "F04", "severity": "Moderate", "dysarthric": True, "duration": 3.0, "transcript": "GOING TO THE STORE"})
    records.append({"utterance_id": "u3", "speaker_id": "MC01", "severity": "Control", "dysarthric": False, "duration": 1.5, "transcript": "GOOD MORNING"})
    
    # Val speakers
    records.append({"utterance_id": "u4", "speaker_id": "F01", "severity": "Severe", "dysarthric": True, "duration": 4.0, "transcript": "OPEN THE WINDOW"})
    records.append({"utterance_id": "u5", "speaker_id": "MC02", "severity": "Control", "dysarthric": False, "duration": 2.0, "transcript": "HELLO THERE"})

    # Train speakers
    records.append({"utterance_id": "u6", "speaker_id": "M02", "severity": "Severe", "dysarthric": True, "duration": 3.5, "transcript": "READ A BOOK"})
    records.append({"utterance_id": "u7", "speaker_id": "F03", "severity": "Moderate", "dysarthric": True, "duration": 2.8, "transcript": "BLUE SKY"})
    records.append({"utterance_id": "u8", "speaker_id": "FC03", "severity": "Control", "dysarthric": False, "duration": 1.8, "transcript": "NICE WEATHER TODAY"})

    return records


def test_validate_speaker_splits_disjoint():
    """Verifies valid disjoint sets pass validation."""
    train = ["M02", "F03", "FC03"]
    val = ["F01", "MC02"]
    test = ["M01", "F04", "MC01"]
    # Should not raise any exception
    validate_speaker_splits(train, val, test)


def test_validate_speaker_splits_leak_detection():
    """Verifies that an intentional speaker overlap raises ValueError."""
    # Leaking 'M01' in both train and test
    train_leak = ["M01", "M02"]
    val = ["F01"]
    test_leak = ["M01", "FC01"]
    with pytest.raises(ValueError, match="Speaker leak detected between Train and Test"):
        validate_speaker_splits(train_leak, val, test_leak)

    # Leaking 'F01' in both val and test
    train = ["M02"]
    val_leak = ["F01"]
    test_val_leak = ["F01", "MC01"]
    with pytest.raises(ValueError, match="Speaker leak detected between Val and Test"):
        validate_speaker_splits(train, val_leak, test_val_leak)


def test_partition_inventory(mock_inventory):
    """Verifies partitioning correctly assigns utterances without data leakage."""
    split_cfg = {
        "train": ["M02", "F03", "FC03"],
        "val": ["F01", "MC02"],
        "test": ["M01", "F04", "MC01"],
    }

    partitioned, stats = partition_inventory(mock_inventory, split_cfg)

    # Check utterance counts
    assert len(partitioned["train"]) == 3
    assert len(partitioned["val"]) == 2
    assert len(partitioned["test"]) == 3
    assert len(partitioned["unassigned"]) == 0

    # Strict isolation check: No test speaker in train or val records
    train_speakers = set(r["speaker_id"] for r in partitioned["train"])
    val_speakers = set(r["speaker_id"] for r in partitioned["val"])
    test_speakers = set(r["speaker_id"] for r in partitioned["test"])

    assert train_speakers.isdisjoint(test_speakers)
    assert train_speakers.isdisjoint(val_speakers)
    assert val_speakers.isdisjoint(test_speakers)

    # Verify severity coverage in test set
    test_severities = set(r["severity"] for r in partitioned["test"])
    assert "Severe" in test_severities
    assert "Moderate" in test_severities
    assert "Control" in test_severities


def test_compute_corpus_statistics(mock_inventory):
    """Verifies calculation of corpus metrics and markdown generation."""
    stats = compute_corpus_statistics(mock_inventory)

    assert stats["total_speakers"] == 8
    assert stats["total_utterances"] == 8
    assert stats["total_duration_seconds"] > 20.0
    assert stats["lexical_metrics"]["total_tokens"] > 0
    assert "Severe" in stats["severity_summary"]
    assert "Control" in stats["severity_summary"]

    # Markdown rendering check
    md_report = generate_markdown_report(stats)
    assert "# Project TIDE: TORGO Dataset Ingestion & Audit Report" in md_report
    assert "Clinical Severity Breakdown" in md_report
    assert "Individual Speaker Breakdown" in md_report
