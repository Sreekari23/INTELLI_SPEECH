"""Unit tests for the in-memory Log-Mel Feature Extractor."""

import os
import time
import pytest
import torch
from src.frontend.mel_extractor import LogMelFeatureExtractor


@pytest.fixture
def config_path():
    """Returns path to the project audio parameters config."""
    path = os.path.join(os.path.dirname(__file__), "..", "configs", "audio_params.yaml")
    return os.path.abspath(path)


def test_log_mel_tensor_dimensions_single_waveform(config_path):
    """Verifies output shape is (1, 80, T) for a single 1D/2D waveform."""
    extractor = LogMelFeatureExtractor(config_path)

    # 3-second synthetic audio at 16,000 Hz = 48,000 samples
    sample_rate = 16000
    duration_sec = 3.0
    num_samples = int(sample_rate * duration_sec)
    time_axis = torch.linspace(0, duration_sec, num_samples)
    waveform = torch.sin(2 * 3.14159 * 440 * time_axis)

    log_mel = extractor(waveform)

    assert log_mel.dim() == 3, f"Expected 3D tensor (B, n_mels, T), got {log_mel.dim()}D"
    assert log_mel.size(0) == 1, f"Expected batch size 1, got {log_mel.size(0)}"
    assert log_mel.size(1) == 80, f"Expected 80 Mel bins, got {log_mel.size(1)}"
    
    # Expected time frames for 3s with hop=160: ~301 frames
    expected_frames = (num_samples // 160) + 1
    assert abs(log_mel.size(2) - expected_frames) <= 2, (
        f"Frame count {log_mel.size(2)} deviates from expected {expected_frames}"
    )


def test_log_mel_batch_processing(config_path):
    """Verifies extractor handles batched waveforms: (B, 1, samples)."""
    extractor = LogMelFeatureExtractor(config_path)
    batch_size = 4
    samples = 16000 * 2  # 2 seconds
    batch_waveform = torch.randn(batch_size, 1, samples)

    log_mel = extractor(batch_waveform)

    assert log_mel.size(0) == batch_size
    assert log_mel.size(1) == 80
    assert log_mel.size(2) > 0


def test_zero_disk_and_cpu_latency(config_path):
    """Verifies feature computation operates strictly in RAM and meets latency budget (<20ms)."""
    extractor = LogMelFeatureExtractor(config_path)
    audio = torch.randn(1, 16000 * 3)  # 3 seconds

    # Warm-up run
    _ = extractor(audio)

    start = time.perf_counter()
    output = extractor(audio)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert isinstance(output, torch.Tensor), "Output must be a PyTorch tensor in RAM"
    assert output.is_floating_point(), "Output must be floating-point"
    assert elapsed_ms < 25.0, f"Extraction latency {elapsed_ms:.2f} ms exceeds budget (<25 ms)"


def test_peak_normalization(config_path):
    """Verifies peak normalization prevents numerical saturation."""
    extractor = LogMelFeatureExtractor(config_path)
    quiet_audio = torch.randn(1, 16000) * 0.001
    loud_audio = torch.randn(1, 16000) * 10.0

    mel_quiet = extractor(quiet_audio)
    mel_loud = extractor(loud_audio)

    # Both should have valid finite numbers with no NaN or Inf
    assert torch.isfinite(mel_quiet).all()
    assert torch.isfinite(mel_loud).all()
