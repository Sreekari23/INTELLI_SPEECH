"""Speech models: Baseline ASR, 1D-CNN phonetic head, and semantic LM."""

from src.models.acoustic_1dcnn import (
    Acoustic1DCNN,
    DepthwiseSeparableConv1d,
    AcousticScorer,
    PhonemeTokenizer,
    LogMelSpectrogramTransform,
    load_acoustic_model,
    CMUDICT_PHONEMES,
    NUM_PHONEMES,
)

__all__ = [
    "Acoustic1DCNN",
    "DepthwiseSeparableConv1d",
    "AcousticScorer",
    "PhonemeTokenizer",
    "LogMelSpectrogramTransform",
    "load_acoustic_model",
    "CMUDICT_PHONEMES",
    "NUM_PHONEMES",
]
