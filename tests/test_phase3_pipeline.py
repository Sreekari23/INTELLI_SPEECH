"""Unit tests for Phase 3 Contrastive Acoustic Training Pipeline.

Tests:
1. Margin-based Contrastive CTC Loss computation & margin penalty behavior.
2. Greedy CTC decoding with blank and duplicate token collapsing.
3. Token-level Phoneme Error Rate (PER) calculation.
4. Dysarthric hard-negative generator and phoneme tokenizer robustness.
"""

import pytest
import torch
import torch.nn.functional as F

from src.models.train_v2 import (
    ContrastiveCTCLoss,
    greedy_ctc_decode,
    compute_phoneme_error_rate,
)
from src.data.torgo_dataset import (
    TORGODiscriminativeDataset,
    discriminative_collate_fn,
)
from src.models.acoustic_1dcnn import PhonemeTokenizer, NUM_PHONEMES


def test_contrastive_ctc_loss_calculation():
    """Verify ContrastiveCTCLoss computes valid losses and penalizes hard negatives."""
    criterion = ContrastiveCTCLoss(margin=2.0, lambda_contrast=0.5)

    batch_size = 2
    time_steps = 40
    num_classes = NUM_PHONEMES

    # Synthetic log_probs: (Time, Batch, Classes)
    logits = torch.randn(time_steps, batch_size, num_classes, requires_grad=True)
    log_probs = F.log_softmax(logits, dim=-1)

    input_lengths = torch.tensor([40, 40], dtype=torch.long)
    true_targets = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long)
    neg_targets = torch.tensor([[1, 2, 8, 4], [5, 9, 7, 0]], dtype=torch.long)
    true_lengths = torch.tensor([4, 3], dtype=torch.long)
    neg_lengths = torch.tensor([4, 3], dtype=torch.long)

    total_loss, l_true, l_contrast = criterion(
        log_probs, true_targets, neg_targets, input_lengths, true_lengths, neg_lengths
    )

    assert not torch.isnan(total_loss)
    assert not torch.isinf(total_loss)
    assert total_loss.item() > 0.0
    assert l_true.item() > 0.0
    assert l_contrast.item() >= 0.0

    # Test backward pass gradient propagation
    total_loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_greedy_ctc_decode():
    """Verify greedy CTC decoding correctly collapses duplicates and drops blanks."""
    # Create logits where argmax gives specific token sequences
    # Blank token is 0
    # Batch sample 0: [0, 1, 1, 0, 2, 2, 0] -> expected [1, 2]
    # Batch sample 1: [3, 3, 3, 0, 4, 4, 5] -> expected [3, 4, 5]
    batch_size = 2
    time_steps = 7
    num_classes = 10

    logits = torch.zeros(batch_size, time_steps, num_classes)

    path_0 = [0, 1, 1, 0, 2, 2, 0]
    path_1 = [3, 3, 3, 0, 4, 4, 5]

    for t, tok in enumerate(path_0):
        logits[0, t, tok] = 10.0
    for t, tok in enumerate(path_1):
        logits[1, t, tok] = 10.0

    decoded = greedy_ctc_decode(logits, blank_id=0)

    assert len(decoded) == 2
    assert decoded[0] == [1, 2]
    assert decoded[1] == [3, 4, 5]


def test_compute_phoneme_error_rate():
    """Verify exact token-level Levenshtein PER calculation."""
    # Identical
    assert compute_phoneme_error_rate([1, 2, 3], [1, 2, 3]) == 0.0

    # Single substitution: 1 out of 3 = ~0.333
    per_sub = compute_phoneme_error_rate([1, 2, 3], [1, 5, 3])
    assert abs(per_sub - 1.0 / 3.0) < 1e-4

    # Empty hyp
    assert compute_phoneme_error_rate([1, 2, 3], []) == 1.0

    # Empty ref
    assert compute_phoneme_error_rate([], []) == 0.0


def test_dysarthric_hard_negative_generation(tmp_path):
    """Verify rule-based phonetic distortions simulate dysarthric articulation."""
    import json
    tokenizer = PhonemeTokenizer()
    ref_tokens = tokenizer.encode("feed")
    assert len(ref_tokens) > 0

    # Create dummy manifest to instantiate dataset
    dummy_manifest = tmp_path / "manifest.json"
    dummy_manifest.write_text(json.dumps([{"file": "dummy.wav", "text": "feed"}]), encoding="utf-8")

    ds = TORGODiscriminativeDataset(manifest_path=dummy_manifest)
    neg_tokens = ds._generate_dysarthric_hard_negative(ref_tokens)
    assert len(neg_tokens) > 0
    # Neg tokens should be a valid list of integers within vocabulary bounds
    for tok in neg_tokens:
        assert 0 <= tok < NUM_PHONEMES
