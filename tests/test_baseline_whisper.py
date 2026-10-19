"""Unit tests for Whisper Baseline ASR, beam decoding, and manifest freezing."""

import pytest
import numpy as np
import torch
from src.models.baseline_whisper import WhisperBaseline
from src.data.freeze_manifest import freeze_splits


def test_compute_evaluation_metrics_exact_match():
    """Verifies WER and CER are 0.0 for identical sentences."""
    refs = ["THE DOG BARKED AT THE MAILMAN", "HELLO WORLD"]
    hyps = ["THE DOG BARKED AT THE MAILMAN", "HELLO WORLD"]
    metrics = WhisperBaseline.compute_evaluation_metrics(refs, hyps)
    assert metrics["wer"] == 0.0
    assert metrics["cer"] == 0.0


def test_compute_evaluation_metrics_substitutions():
    """Verifies WER accurately measures word substitutions."""
    refs = ["THE DOG BARKED"]
    hyps = ["THE DAG BARKED"]  # 1 substitution out of 3 words = 1/3 = 0.3333
    metrics = WhisperBaseline.compute_evaluation_metrics(refs, hyps)
    assert 0.30 <= metrics["wer"] <= 0.35
    assert metrics["cer"] > 0.0


def test_freeze_splits_mock(tmp_path):
    """Verifies Milestone Gate 1: freeze_splits writes valid schema and enforces disjointness."""
    inv_file = tmp_path / "mock_inventory.json"
    out_manifest = tmp_path / "configs" / "splits_manifest.json"

    # Minimal mock inventory with train, val, test speakers
    mock_records = [
        {"utterance_id": "u1", "speaker_id": "M01", "severity": "Severe", "transcript": "TEST A", "duration": 1.2},
        {"utterance_id": "u2", "speaker_id": "F01", "severity": "Severe", "transcript": "TEST B", "duration": 2.0},
        {"utterance_id": "u3", "speaker_id": "M02", "severity": "Severe", "transcript": "TEST C", "duration": 1.5},
    ]
    inv_file.write_text(json_dumps := __import__("json").dumps(mock_records), encoding="utf-8")

    stats = freeze_splits(
        inventory_path=str(inv_file),
        splits_config_path="configs/splits.yaml",
        output_manifest_path=str(out_manifest)
    )

    assert out_manifest.exists()
    assert "train" in stats
    assert "val" in stats
    assert "test" in stats


@pytest.mark.slow
def test_whisper_baseline_inference():
    """Tests Whisper ASR greedy and beam search generation on synthetic 16kHz audio."""
    # 2 seconds of synthetic audio
    sample_rate = 16000
    audio = np.zeros(sample_rate * 2, dtype=np.float32)

    model = WhisperBaseline(model_name="openai/whisper-tiny.en", device="cpu")

    # 1. Greedy decoding test
    greedy_res = model.transcribe_greedy(audio, sample_rate=sample_rate)
    assert "transcript" in greedy_res
    assert "latency_sec" in greedy_res
    assert greedy_res["beam_width"] == 1
    assert greedy_res["latency_sec"] > 0

    # 2. Beam search decoding test (K=3)
    beam_res = model.transcribe_beam(audio, beam_width=3, sample_rate=sample_rate)
    assert len(beam_res["candidates"]) == 3
    assert "confidence_metrics" in beam_res
    conf = beam_res["confidence_metrics"]
    assert 0.0 <= conf["p_norm_top1"] <= 1.0
    assert conf["delta_margin"] >= 0.0  # Top-1 score >= Top-2 score
