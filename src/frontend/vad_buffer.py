"""Dysarthric Voice Activity Detection (VAD) & Volatile RAM Ring Buffer.

Project TIDE: Edge-first Dysarthric Speech Interpretation.

Architectural Design & Objectives:
1. Volatile RAM Ring Buffer:
   - Zero disk I/O; incoming streaming audio is maintained purely in RAM.
   - Fixed lookback pre-speech circular ring buffer (e.g., 200-300 ms) to capture
     acoustic onsets and soft articulatory precursors.
2. Extended Silence Hangover (800–1200 ms):
   - Standard commercial VADs (200–300 ms hangover) prematurely truncate dysarthric
     speech during prolonged inter-word pauses and respiratory resets.
   - TIDE enforces a calibrated 800–1200 ms (default 900 ms) hangover window.
3. Max Utterance Safety Ceiling:
   - Hard upper bound (default 10.0 s) preventing runaway memory allocation on edge
     devices in noisy or high-strain acoustic environments.
4. Modular VAD Engine:
   - Supports pluggable neural VAD (e.g., Silero VAD) with robust, zero-dependency
     energy-based fallback for deterministic edge runtime and unit testing.
"""

import os
from collections import deque
from typing import Dict, List, Optional, Union, Callable, Any
import numpy as np
import torch
import yaml


class DysarthricVADBuffer:
    """Real-time streaming circular ring buffer with dysarthric-tailored VAD hangover."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        silence_hangover_ms: int = 900,
        max_utterance_sec: float = 10.0,
        pre_speech_ms: int = 300,
        vad_threshold: float = 0.5,
        vad_fn: Optional[Callable[[np.ndarray], float]] = None,
    ):
        """Initializes the VAD circular ring buffer.

        Args:
            config_path: Path to YAML configuration file (e.g., audio_params.yaml).
            sample_rate: Target sampling rate in Hz (enforced at 16000).
            frame_duration_ms: Frame length in ms (30 ms = 480 samples at 16 kHz).
            silence_hangover_ms: Prolonged silence timeout before segment cut (800-1200 ms).
            max_utterance_sec: Hard safety ceiling in seconds to prevent buffer overflow.
            pre_speech_ms: Audio history captured before speech trigger in ms (e.g. 300 ms).
            vad_threshold: Probability threshold for classification as active speech.
            vad_fn: Custom callable returning speech probability in [0.0, 1.0].
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.silence_hangover_ms = silence_hangover_ms
        self.max_utterance_sec = max_utterance_sec
        self.pre_speech_ms = pre_speech_ms
        self.vad_threshold = vad_threshold
        self.vad_fn = vad_fn

        # Override defaults if config YAML is supplied
        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if "audio" in cfg:
                self.sample_rate = cfg["audio"].get("sample_rate", self.sample_rate)
            if "vad" in cfg:
                vad_cfg = cfg["vad"]
                self.frame_duration_ms = vad_cfg.get("frame_duration_ms", self.frame_duration_ms)
                self.silence_hangover_ms = vad_cfg.get("silence_hangover_ms", self.silence_hangover_ms)
                self.max_utterance_sec = vad_cfg.get("max_utterance_sec", self.max_utterance_sec)
                self.vad_threshold = vad_cfg.get("threshold", self.vad_threshold)

        # Derived sample and frame counts
        self.frame_size = int(self.sample_rate * (self.frame_duration_ms / 1000.0))
        self.pre_speech_frames = max(1, int(self.pre_speech_ms / self.frame_duration_ms))
        self.hangover_frames = max(1, int(self.silence_hangover_ms / self.frame_duration_ms))
        self.max_frames = int((self.max_utterance_sec * 1000.0) / self.frame_duration_ms)

        # State tracking
        self.pre_speech_ring = deque(maxlen=self.pre_speech_frames)
        self.active_utterance_frames: List[np.ndarray] = []
        self.is_speaking = False
        self.consecutive_silence_count = 0
        self.total_speech_frames_in_utterance = 0

    def _default_energy_vad(self, frame: np.ndarray) -> float:
        """Lightweight root-mean-square energy detector for speech activity.

        Calibrated to return high confidence (>0.5) on articulated phonemes and
        low confidence (<0.5) on ambient background / room hiss.
        """
        # Ensure float32 normalized in [-1.0, 1.0]
        if frame.dtype == np.int16:
            frame_float = frame.astype(np.float32) / 32768.0
        else:
            frame_float = frame.astype(np.float32)

        rms = float(np.sqrt(np.mean(frame_float**2) + 1e-12))
        # Standard ambient noise floor is typically < 0.01 (-40 dBFS)
        # Moderate speech typically reaches 0.03 - 0.2
        norm_val = np.clip((rms - 0.01) / 0.04, 0.0, 1.0)
        return float(norm_val)

    def classify_frame(self, frame: np.ndarray) -> float:
        """Computes speech activity probability for a single audio frame."""
        if self.vad_fn is not None:
            return float(self.vad_fn(frame))
        return self._default_energy_vad(frame)

    def process_frame(
        self,
        frame: Union[np.ndarray, torch.Tensor, List[float]],
    ) -> Optional[Dict[str, Any]]:
        """Ingests a single 30ms streaming audio frame and updates VAD state machine.

        Args:
            frame: 1D array or tensor containing exactly `frame_size` audio samples.

        Returns:
            Dict containing the completed utterance waveform and telemetry metadata
            if an utterance boundaries was finalized (either by silence hangover or
            safety ceiling), otherwise None.
        """
        if isinstance(frame, torch.Tensor):
            frame_np = frame.detach().cpu().numpy().squeeze()
        elif isinstance(frame, list):
            frame_np = np.array(frame, dtype=np.float32)
        else:
            frame_np = np.asarray(frame)

        # Enforce 1D frame
        if frame_np.ndim != 1:
            frame_np = frame_np.reshape(-1)

        # Check frame size
        if len(frame_np) != self.frame_size:
            raise ValueError(
                f"Frame length {len(frame_np)} does not match expected {self.frame_size} "
                f"({self.frame_duration_ms}ms at {self.sample_rate}Hz)"
            )

        speech_prob = self.classify_frame(frame_np)
        frame_is_speech = speech_prob >= self.vad_threshold

        # State 1: Currently Idle / Listening
        if not self.is_speaking:
            if frame_is_speech:
                # Transition: Speech Triggered
                self.is_speaking = True
                self.consecutive_silence_count = 0
                self.total_speech_frames_in_utterance = 1
                self.active_utterance_frames = list(self.pre_speech_ring)
                self.active_utterance_frames.append(frame_np.copy())
            else:
                # Remain Idle: Update circular lookback pre-speech buffer
                self.pre_speech_ring.append(frame_np.copy())
            return None

        # State 2: Currently Active Speaking
        self.active_utterance_frames.append(frame_np.copy())

        if frame_is_speech:
            self.total_speech_frames_in_utterance += 1
            # Reset silence counter because speech has resumed
            self.consecutive_silence_count = 0
        else:
            # Silence detected during utterance
            self.consecutive_silence_count += 1

        # Check Termination Condition A: Safety Ceiling Exceeded
        if len(self.active_utterance_frames) >= self.max_frames:
            return self._finalize_utterance(reason="max_ceiling")

        # Check Termination Condition B: Dysarthric Silence Hangover Elapsed
        if self.consecutive_silence_count >= self.hangover_frames:
            return self._finalize_utterance(reason="silence_hangover")

        return None

    def _finalize_utterance(self, reason: str) -> Dict[str, Any]:
        """Finalizes the accumulated utterance and resets state machine."""
        if not self.active_utterance_frames:
            self.reset()
            return {}

        waveform_np = np.concatenate(self.active_utterance_frames, axis=0)
        num_samples = len(waveform_np)
        duration_sec = float(num_samples / self.sample_rate)

        result = {
            "waveform": torch.from_numpy(waveform_np).float(),
            "waveform_np": waveform_np,
            "duration_sec": duration_sec,
            "num_samples": num_samples,
            "total_frames": len(self.active_utterance_frames),
            "speech_frames": self.total_speech_frames_in_utterance,
            "termination_reason": reason,
            "hangover_ms": self.silence_hangover_ms,
        }

        # Clear active state and reset ring buffer
        self.active_utterance_frames = []
        self.pre_speech_ring.clear()
        self.is_speaking = False
        self.consecutive_silence_count = 0
        self.total_speech_frames_in_utterance = 0

        return result

    def flush(self) -> Optional[Dict[str, Any]]:
        """Flushes remaining speech in the buffer if currently active."""
        if self.is_speaking and self.active_utterance_frames:
            return self._finalize_utterance(reason="forced_flush")
        self.reset()
        return None

    def reset(self) -> None:
        """Resets the buffer and internal state machine to clean initial state."""
        self.pre_speech_ring.clear()
        self.active_utterance_frames.clear()
        self.is_speaking = False
        self.consecutive_silence_count = 0
        self.total_speech_frames_in_utterance = 0
