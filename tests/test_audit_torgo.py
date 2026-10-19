"""Unit tests for TORGO channel audit, text hygiene, and resampling pipeline."""

import os
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import torch

from src.data.audit_torgo import (
    normalize_transcript,
    is_head_mic_path,
    get_speaker_id_from_path,
    audit_torgo_corpus,
    standardize_and_resample_audio,
)


def test_normalize_transcript():
    """Verifies transcript normalization strips annotations and standardizes casing."""
    raw_1 = "[cough] The dog, barked at the mailman! (laughter)"
    assert normalize_transcript(raw_1) == "THE DOG BARKED AT THE MAILMAN"

    raw_2 = "don't give up...   keep going."
    assert normalize_transcript(raw_2) == "DON'T GIVE UP KEEP GOING"

    raw_3 = "   [groan]  \n  "
    assert normalize_transcript(raw_3) == ""

    assert normalize_transcript("") == ""


def test_is_head_mic_path():
    """Verifies channel isolation strictly separates headMic from arrayMic."""
    # Positive headMic cases
    assert is_head_mic_path("F01/Session1/wav_headMic/001.wav") is True
    assert is_head_mic_path("dataset/torgo/MC01/head_mic/sample.wav") is True
    assert is_head_mic_path("C:\\Users\\data\\F03\\wav_headMic\\test_01.wav") is True

    # Negative arrayMic cases (must be rejected)
    assert is_head_mic_path("F01/Session1/wav_arrayMic/001.wav") is False
    assert is_head_mic_path("torgo/M02/Session2/array_mic/005.wav") is False
    assert is_head_mic_path("C:\\data\\arrayMic\\sample_01.wav") is False


def test_get_speaker_id_from_path():
    """Verifies speaker extraction from directory paths."""
    assert get_speaker_id_from_path("data/TORGO/F01/Session1/wav_headMic/001.wav") == "F01"
    assert get_speaker_id_from_path("C:\\torgo\\MC02\\wav_headMic\\test.wav") == "MC02"
    assert get_speaker_id_from_path("F04/Session2/001.wav") == "F04"
    assert get_speaker_id_from_path("random_folder/test.wav") is None


def create_synthetic_wav(path: Path, sample_rate: int = 16000, duration_sec: float = 0.5, channels: int = 1):
    """Creates a synthetic PCM WAV file for test fixtures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(sample_rate * duration_sec)
    time_axis = np.linspace(0, duration_sec, num_samples, endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 440 * time_axis)
    if channels > 1:
        audio = np.stack([audio] * channels, axis=-1)
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")


def test_audit_torgo_corpus_mock(tmp_path):
    """Tests end-to-end audit on a mock TORGO directory structure."""
    torgo_root = tmp_path / "mock_torgo"

    # 1. Valid Dysarthric Utterance (Speaker F01, headMic)
    f01_head = torgo_root / "F01" / "Session1" / "wav_headMic" / "0001.wav"
    f01_prompt = torgo_root / "F01" / "Session1" / "prompts" / "0001.txt"
    create_synthetic_wav(f01_head, sample_rate=16000, duration_sec=1.0)
    f01_prompt.parent.mkdir(parents=True, exist_ok=True)
    f01_prompt.write_text("The quick brown fox", encoding="utf-8")

    # 2. Array Mic recording (Must be discarded!)
    f01_array = torgo_root / "F01" / "Session1" / "wav_arrayMic" / "0001.wav"
    create_synthetic_wav(f01_array, sample_rate=16000, duration_sec=1.0)

    # 3. Valid Control Utterance (Speaker FC01, headMic)
    fc01_head = torgo_root / "FC01" / "Session1" / "wav_headMic" / "0002.wav"
    fc01_prompt = torgo_root / "FC01" / "Session1" / "prompts" / "0002.txt"
    create_synthetic_wav(fc01_head, sample_rate=44100, duration_sec=0.8)
    fc01_prompt.parent.mkdir(parents=True, exist_ok=True)
    fc01_prompt.write_text("[cough] Jumped over the lazy dog.", encoding="utf-8")

    # 4. Utterance with missing prompt (Must be discarded!)
    m01_head_noprompt = torgo_root / "M01" / "Session1" / "wav_headMic" / "0003.wav"
    create_synthetic_wav(m01_head_noprompt, sample_rate=16000, duration_sec=0.5)

    # 5. Empty 0-byte audio file (Must be discarded!)
    m02_empty = torgo_root / "M02" / "Session1" / "wav_headMic" / "0004.wav"
    m02_empty.parent.mkdir(parents=True, exist_ok=True)
    m02_empty.touch()
    m02_prompt = torgo_root / "M02" / "Session1" / "prompts" / "0004.txt"
    m02_prompt.parent.mkdir(parents=True, exist_ok=True)
    m02_prompt.write_text("Hello world", encoding="utf-8")

    # Run Corpus Audit
    records, stats = audit_torgo_corpus(torgo_root)

    # Assertions
    assert stats["valid_utterances"] == 2
    assert stats["array_mic_files_dropped"] == 1
    assert stats["missing_prompt_dropped"] == 1
    assert stats["corrupted_audio_dropped"] == 1
    assert len(records) == 2

    # Check F01 record
    rec_f01 = next(r for r in records if r["speaker_id"] == "F01")
    assert rec_f01["transcript"] == "THE QUICK BROWN FOX"
    assert rec_f01["severity"] == "Severe"
    assert rec_f01["dysarthric"] is True

    # Check FC01 record
    rec_fc01 = next(r for r in records if r["speaker_id"] == "FC01")
    assert rec_fc01["transcript"] == "JUMPED OVER THE LAZY DOG"
    assert rec_fc01["severity"] == "Control"
    assert rec_fc01["dysarthric"] is False


def test_standardize_and_resample_audio(tmp_path):
    """Tests audio resampling to 16 kHz mono float32."""
    source_wav = tmp_path / "raw_stereo_44k.wav"
    target_wav = tmp_path / "standardized_16k.wav"

    # Create 44.1kHz stereo file
    create_synthetic_wav(source_wav, sample_rate=44100, duration_sec=0.5, channels=2)

    # Standardize
    success = standardize_and_resample_audio(str(source_wav), str(target_wav), target_sample_rate=16000)
    assert success is True
    assert target_wav.exists()

    # Verify properties of resampled audio
    info = sf.info(str(target_wav))
    assert info.samplerate == 16000
    assert info.channels == 1
    assert info.subtype == "PCM_16"
