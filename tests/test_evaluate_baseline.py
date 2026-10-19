"""Unit tests for Baseline Evaluation & Candidate Beam Extraction (Days 9 & 10)."""

import pytest
import numpy as np
from src.models.baseline_whisper import WhisperBaseline
from src.models.evaluate_baseline import evaluate_records, generate_benchmark_markdown


@pytest.fixture
def mock_evaluation_records():
    """Generates synthetic records spanning Control, Mild, Moderate, and Severe."""
    return [
        {
            "utterance_id": "utt_ctrl_01",
            "speaker_id": "MC01",
            "severity": "Control",
            "transcript": "GOOD MORNING",
            "duration": 1.0,
            "raw_audio_path": None,  # Will use synthetic audio fallback
        },
        {
            "utterance_id": "utt_mild_01",
            "speaker_id": "M05",
            "severity": "Mild",
            "transcript": "GOING HOME",
            "duration": 1.5,
            "raw_audio_path": None,
        },
        {
            "utterance_id": "utt_mod_01",
            "speaker_id": "F04",
            "severity": "Moderate",
            "transcript": "THE DOG BARKED",
            "duration": 2.0,
            "raw_audio_path": None,
        },
        {
            "utterance_id": "utt_sev_01",
            "speaker_id": "M01",
            "severity": "Severe",
            "transcript": "PLEASE HELP ME",
            "duration": 2.5,
            "raw_audio_path": None,
        },
    ]


def test_evaluate_records_and_candidate_caching(mock_evaluation_records):
    """Verifies batch evaluation computes disaggregated metrics and extracts candidates."""
    engine = WhisperBaseline(model_name="openai/whisper-tiny.en", device="cpu")
    beam_width = 3

    summary, candidate_cache = evaluate_records(
        records=mock_evaluation_records,
        asr_engine=engine,
        beam_width=beam_width,
    )

    # 1. Day 9 Metrics Verification
    assert summary["total_evaluated"] == 4
    assert "overall_wer" in summary
    assert "overall_cer" in summary
    assert summary["real_time_factor_rtf"] > 0.0

    # Severity disaggregation check
    sev_breakdown = summary["severity_breakdown"]
    assert "Control" in sev_breakdown
    assert "Mild" in sev_breakdown
    assert "Moderate" in sev_breakdown
    assert "Severe" in sev_breakdown

    # Speaker disaggregation check
    spk_breakdown = summary["speaker_breakdown"]
    assert "MC01" in spk_breakdown
    assert "M01" in spk_breakdown

    # 2. Day 10 Candidate Cache Verification
    assert len(candidate_cache) == 4
    for entry in candidate_cache:
        assert "utterance_id" in entry
        assert "reference" in entry
        assert "candidates" in entry
        assert len(entry["candidates"]) == beam_width
        
        # Verify sequence log-probs and ranking
        cand_list = entry["candidates"]
        assert cand_list[0]["rank"] == 1
        assert cand_list[1]["rank"] == 2
        assert cand_list[0]["sequence_log_prob"] >= cand_list[1]["sequence_log_prob"]

        # Verify confidence metrics
        conf = entry["confidence_metrics"]
        assert 0.0 <= conf["p_norm_top1"] <= 1.0
        assert conf["delta_margin"] >= 0.0

    # 3. Markdown Report Generation Check
    md_report = generate_benchmark_markdown(summary)
    assert "# Project TIDE: Baseline ASR Benchmark Report" in md_report
    assert "Clinical Severity Degradation Matrix" in md_report
    assert "Speaker Breakdown" in md_report
