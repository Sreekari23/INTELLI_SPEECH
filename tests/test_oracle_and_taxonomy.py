"""Unit tests for Oracle Candidate Coverage and Dysarthric Error Taxonomy (Days 11 & 12)."""

import pytest
from src.models.oracle_coverage import compute_oracle_coverage, generate_oracle_markdown
from src.models.error_taxonomy import (
    is_vowel_substitution,
    is_consonant_omission,
    analyze_utterance_errors,
    compile_corpus_error_taxonomy,
    generate_taxonomy_markdown,
)


@pytest.fixture
def mock_candidate_records():
    """Generates synthetic candidate records testing Oracle selection and error taxonomy."""
    return [
        {
            "utterance_id": "utt_01",
            "speaker_id": "M01",
            "severity": "Severe",
            "reference": "THE DOG BARKED",
            "top1_hypothesis": "THE DAG BARKED",  # Vowel distortion in Top-1
            "candidates": [
                {"rank": 1, "transcript": "THE DAG BARKED", "sequence_log_prob": -0.8},
                {"rank": 2, "transcript": "THE DOG BARKED", "sequence_log_prob": -1.2},  # Exact match in beam!
                {"rank": 3, "transcript": "THE DUG BARKED", "sequence_log_prob": -1.5},
            ],
            "confidence_metrics": {"p_norm_top1": 0.52, "delta_margin": 0.4},
        },
        {
            "utterance_id": "utt_02",
            "speaker_id": "F04",
            "severity": "Moderate",
            "reference": "COLD WATER",
            "top1_hypothesis": "COL WATER",  # Consonant omission in Top-1
            "candidates": [
                {"rank": 1, "transcript": "COL WATER", "sequence_log_prob": -0.9},
                {"rank": 2, "transcript": "COLD WATER", "sequence_log_prob": -1.1},  # Exact match in beam!
            ],
            "confidence_metrics": {"p_norm_top1": 0.60, "delta_margin": 0.2},
        },
        {
            "utterance_id": "utt_03",
            "speaker_id": "MC01",
            "severity": "Control",
            "reference": "GOOD MORNING",
            "top1_hypothesis": "GOOD MORNING",  # Already exact match
            "candidates": [
                {"rank": 1, "transcript": "GOOD MORNING", "sequence_log_prob": -0.1},
                {"rank": 2, "transcript": "GOOD MOANING", "sequence_log_prob": -2.0},
            ],
            "confidence_metrics": {"p_norm_top1": 0.95, "delta_margin": 1.9},
        },
    ]


def test_oracle_candidate_coverage(mock_candidate_records):
    """Verifies Oracle selection calculates positive headroom and recovers Top-1 errors."""
    results = compute_oracle_coverage(mock_candidate_records)

    assert results["total_utterances"] == 3
    # Top-1 had errors on utt_01 and utt_02
    assert results["baseline_top1_wer"] > 0.0
    # Oracle found exact matches in beam for all 3!
    assert results["oracle_top_n_wer"] == 0.0
    assert results["absolute_reranking_headroom"] > 0.0
    assert results["relative_reranking_headroom_pct"] == 100.0

    # Recoverable errors: utt_01 and utt_02 were recovered
    assert results["recoverable_error_count"] == 2
    assert results["exact_match_in_beam_pct"] == 100.0

    # Verify Markdown generation
    md_report = generate_oracle_markdown(results)
    assert "# Project TIDE: Oracle Top-N Candidate Coverage Report" in md_report
    assert "Executive Summary & Reranking Headroom" in md_report


def test_error_taxonomy_linguistic_detectors():
    """Verifies phonetic detectors for vowel distortions and consonant omissions."""
    # Vowel substitutions
    assert is_vowel_substitution("DOG", "DAG") is True
    assert is_vowel_substitution("PEN", "PAN") is True
    assert is_vowel_substitution("DOG", "CAT") is False

    # Consonant omissions
    assert is_consonant_omission("COLD", "COL") is True
    assert is_consonant_omission("FAST", "FAS") is True
    assert is_consonant_omission("BLUE", "RED") is False


def test_error_taxonomy_corpus_compilation(mock_candidate_records):
    """Verifies aggregation of error taxonomy across multiple utterances."""
    results = compile_corpus_error_taxonomy(mock_candidate_records)

    assert results["total_evaluated_utterances"] == 3
    acous = results["dysarthric_acoustic_categories"]
    assert acous["vowel_centralization_distortions"] >= 1  # DAG -> DOG
    assert acous["consonant_cluster_omissions"] >= 1       # COAL -> COLD

    # Markdown generation check
    md_report = generate_taxonomy_markdown(results)
    assert "# Project TIDE: Dysarthric Error Taxonomy & Linguistic Analysis" in md_report
    assert "Vowel Centralization / Distortion" in md_report
