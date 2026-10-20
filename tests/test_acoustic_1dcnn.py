"""Unit tests for Acoustic1DCNN architecture, checkpoint loading, and CTC scoring."""

import pytest
import torch
from pathlib import Path
from src.models.acoustic_1dcnn import (
    Acoustic1DCNN,
    AcousticScorer,
    PhonemeTokenizer,
    LogMelSpectrogramTransform,
    load_acoustic_model,
    NUM_PHONEMES,
)

CHECKPOINT_PATH = Path("checkpoints/best_1dcnn.pth")


def test_acoustic_1dcnn_architecture():
    """Verifies Acoustic1DCNN dimensions, forward pass output shape, and parameter budget."""
    model = Acoustic1DCNN(in_features=80, num_classes=NUM_PHONEMES, hidden_dim=256)
    param_count = model.count_parameters()

    # Parameter budget constraint: < 1.5M parameters
    assert param_count < 1_500_000
    assert param_count > 500_000

    # Test forward pass with synthetic spectrogram: (Batch=2, Mels=80, Time=150)
    x = torch.randn(2, 80, 150)
    out = model(x)
    assert out.shape == (2, 150, NUM_PHONEMES)


def test_load_trained_checkpoint():
    """Verifies loading checkpoint from checkpoints/best_1dcnn.pth."""
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"Checkpoint not found at {CHECKPOINT_PATH}")

    model, meta = load_acoustic_model(CHECKPOINT_PATH, device="cpu")
    assert isinstance(model, Acoustic1DCNN)
    assert "model_state_dict" in meta
    assert model.count_parameters() < 1_500_000


def test_phoneme_tokenizer():
    """Verifies CMUDict phoneme encoding and decoding."""
    tokenizer = PhonemeTokenizer()
    text = "COLD WATER"
    tokens = tokenizer.encode(text)
    assert len(tokens) > 0
    assert all(isinstance(t, int) and 0 <= t < NUM_PHONEMES for t in tokens)

    decoded = tokenizer.decode(tokens)
    assert isinstance(decoded, str)
    assert len(decoded) > 0


def test_acoustic_scorer_ctc_alignment():
    """Verifies CTC alignment loss and acoustic score normalization on synthetic audio."""
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"Checkpoint not found at {CHECKPOINT_PATH}")

    scorer = AcousticScorer(checkpoint_path=CHECKPOINT_PATH, device="cpu")
    # Synthetic 1-second 16kHz audio
    waveform = torch.randn(1, 16000)
    candidates = ["HELLO WORLD", "COLD WATER", "SOME TEXT"]

    scores = scorer.score_candidates(waveform, candidates, sample_rate=16000)
    assert len(scores) == len(candidates)
    for res in scores:
        assert "transcript" in res
        assert "ctc_loss" in res
        assert "acoustic_score" in res
        assert 0.0 <= res["acoustic_score"] <= 1.0

    # Ensure at least one candidate achieves normalized max score 1.0
    max_score = max(r["acoustic_score"] for r in scores)
    assert max_score == 1.0
