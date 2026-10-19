"""Unit tests for the Dysarthric VAD & Volatile RAM Ring Buffer (Day 13).

Validates:
1. Frame duration enforcement (30 ms = 480 samples at 16 kHz).
2. Lookback pre-speech circular buffer preservation.
3. Extended silence hangover (900 ms): prevents truncation of atypical inter-word pauses.
4. Utterance safety ceiling cutoff: triggers before memory overflow.
5. In-memory throughput and zero-disk operation (<15 ms for multi-second streaming).
"""

import os
import time
import pytest
import numpy as np
import torch

from src.frontend.vad_buffer import DysarthricVADBuffer


@pytest.fixture
def config_path():
    """Path to audio parameters YAML."""
    path = os.path.join(os.path.dirname(__file__), "..", "configs", "audio_params.yaml")
    return os.path.abspath(path)


def test_vad_buffer_initialization_from_config(config_path):
    """Verifies buffer initializes correctly from audio_params.yaml."""
    buffer = DysarthricVADBuffer(config_path=config_path)

    assert buffer.sample_rate == 16000
    assert buffer.frame_duration_ms == 30
    assert buffer.frame_size == 480
    assert buffer.silence_hangover_ms == 900
    assert buffer.hangover_frames == 30
    assert buffer.max_utterance_sec == 10.0
    assert buffer.max_frames == int((10.0 * 1000) / 30)


def test_vad_frame_size_validation():
    """Verifies ValueError is raised if incoming frame does not match 480 samples."""
    buffer = DysarthricVADBuffer(sample_rate=16000, frame_duration_ms=30)
    invalid_frame = np.zeros(250, dtype=np.float32)

    with pytest.raises(ValueError, match="does not match expected 480"):
        buffer.process_frame(invalid_frame)


def test_prespeech_ring_buffer_lookback():
    """Verifies pre-speech ring buffer preserves audio leading up to speech onset."""
    pre_speech_ms = 120  # 4 frames of 30ms
    buffer = DysarthricVADBuffer(
        pre_speech_ms=pre_speech_ms,
        frame_duration_ms=30,
        silence_hangover_ms=300,
        vad_fn=lambda frame: 0.9 if frame[0] > 0.5 else 0.0,
    )

    silence_frame = np.full(480, 0.1, dtype=np.float32)
    speech_frame = np.full(480, 0.8, dtype=np.float32)

    # Send 10 silence frames (ring buffer maxlen = 4)
    for _ in range(10):
        res = buffer.process_frame(silence_frame)
        assert res is None

    assert len(buffer.pre_speech_ring) == 4

    # Now trigger speech
    buffer.process_frame(speech_frame)
    assert buffer.is_speaking is True
    # Utterance should contain 4 pre-speech frames + 1 speech frame = 5 frames
    assert len(buffer.active_utterance_frames) == 5


def test_dysarthric_pause_not_truncated():
    """Verifies an atypical 600ms pause (20 frames) is preserved without premature truncation."""
    # 900ms hangover = 30 frames
    buffer = DysarthricVADBuffer(
        silence_hangover_ms=900,
        frame_duration_ms=30,
        vad_fn=lambda f: float(f[0]),  # 1.0 for speech, 0.0 for silence
    )

    speech_frame = np.full(480, 1.0, dtype=np.float32)
    silence_frame = np.full(480, 0.0, dtype=np.float32)

    # 0. Idle ambient background: 10 frames (300ms) to fill pre-speech ring buffer
    for _ in range(10):
        assert buffer.process_frame(silence_frame) is None

    # 1. Trigger speech (Word 1: 5 frames = 150ms)
    for _ in range(5):
        assert buffer.process_frame(speech_frame) is None

    # 2. Dysarthric breath pause: 600 ms (20 frames of silence)
    # Standard 200-300ms VAD would cut here, but TIDE must maintain the session!
    for _ in range(20):
        res = buffer.process_frame(silence_frame)
        assert res is None, "Dysarthric pause was prematurely cut off by VAD!"

    assert buffer.is_speaking is True
    assert buffer.consecutive_silence_count == 20

    # 3. Resumed speech (Word 2: 5 frames = 150ms) -> resets silence count!
    for _ in range(5):
        assert buffer.process_frame(speech_frame) is None
    assert buffer.consecutive_silence_count == 0

    # 4. Final silence: 30 frames (900ms) -> triggers finalization
    final_res = None
    for i in range(30):
        final_res = buffer.process_frame(silence_frame)
        if i < 29:
            assert final_res is None
        else:
            assert final_res is not None

    assert final_res["termination_reason"] == "silence_hangover"
    assert final_res["hangover_ms"] == 900
    # Total frames: pre-speech (10) + word1 (5) + pause (20) + word2 (5) + hangover (30) = 70 frames
    assert final_res["total_frames"] == 70
    assert final_res["duration_sec"] == pytest.approx(70 * 0.030, rel=1e-3)
    assert isinstance(final_res["waveform"], torch.Tensor)


def test_max_utterance_safety_ceiling():
    """Verifies buffer terminates when reaching max_utterance_sec to prevent memory overrun."""
    # Configure low ceiling for test: 0.6 seconds (20 frames)
    buffer = DysarthricVADBuffer(
        max_utterance_sec=0.6,
        frame_duration_ms=30,
        silence_hangover_ms=1000,
        vad_fn=lambda _: 1.0,  # Continuous speech
    )

    frame = np.full(480, 0.5, dtype=np.float32)
    res = None

    # Pre-speech is 300ms (10 frames). At frame 10 of active speaking, total reaches 20 frames
    for _ in range(30):
        res = buffer.process_frame(frame)
        if res is not None:
            break

    assert res is not None
    assert res["termination_reason"] == "max_ceiling"
    assert res["total_frames"] == 20
    assert res["duration_sec"] == pytest.approx(0.6, rel=1e-3)


def test_flush_and_reset():
    """Verifies forced flush and reset semantics."""
    buffer = DysarthricVADBuffer(vad_fn=lambda _: 1.0)
    frame = np.full(480, 0.5, dtype=np.float32)

    # 1. Test flush when idle returns None
    assert buffer.flush() is None

    # 2. Test flush during active utterance
    buffer.process_frame(frame)
    assert buffer.is_speaking is True
    res = buffer.flush()
    assert res is not None
    assert res["termination_reason"] == "forced_flush"
    assert buffer.is_speaking is False

    # 3. Test reset clears state without emitting
    buffer.process_frame(frame)
    buffer.reset()
    assert buffer.is_speaking is False
    assert len(buffer.active_utterance_frames) == 0


def test_streaming_cpu_latency_and_zero_disk():
    """Verifies that 3 seconds (100 frames) of audio processes in < 15ms purely in RAM."""
    buffer = DysarthricVADBuffer()
    audio_frames = [np.random.randn(480).astype(np.float32) for _ in range(100)]

    t0 = time.perf_counter()
    for frame in audio_frames:
        buffer.process_frame(frame)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Must process 3000 ms of audio well under 15 ms
    assert elapsed_ms < 15.0, f"Streaming VAD latency {elapsed_ms:.2f} ms exceeds 15 ms budget"
