"""In-Memory Log-Mel Spectrogram Feature Extractor for Project TIDE.

This module computes 80-channel log-Mel filterbank spectrograms directly
in RAM as PyTorch floating-point tensors.

Architectural Guarantees:
1. Zero Disk Footprint: Spectrograms are never converted to image files
   (PNG, JPEG) or written to disk.
2. Standardized Format: Enforces 16 kHz sampling, 25 ms window (n_fft=400),
   10 ms hop (160 samples), and 80 Mel frequency bins.
3. Numerical Stability: Applies calibrated log-compression (log(mel + eps)).
"""

import os
from typing import Dict, Any, Optional, Union
import torch
import torchaudio.transforms as T
import yaml


class LogMelFeatureExtractor:
    """Extracts 80-channel log-Mel filterbank tensors in memory."""

    def __init__(self, config_path: Optional[str] = None):
        """Initializes the extractor with project configuration.

        Args:
            config_path: Path to YAML config file. If None or not found, uses defaults.
        """
        # Default fallback parameters
        self.sample_rate = 16000
        self.n_mels = 80
        self.n_fft = 400
        self.hop_length = 160
        self.f_min = 0.0
        self.f_max = 8000.0
        self.power = 2.0
        self.log_eps = 1.0e-5
        self.normalize_peak = True

        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            
            if "audio" in cfg:
                self.sample_rate = cfg["audio"].get("sample_rate", self.sample_rate)
                self.normalize_peak = cfg["audio"].get("normalize_peak", self.normalize_peak)
            
            if "features" in cfg:
                feat = cfg["features"]
                self.n_mels = feat.get("n_mels", self.n_mels)
                self.n_fft = feat.get("n_fft", self.n_fft)
                self.hop_length = feat.get("hop_length", self.hop_length)
                self.f_min = float(feat.get("f_min", self.f_min))
                self.f_max = float(feat.get("f_max", self.f_max))
                self.power = float(feat.get("power", self.power))
                self.log_eps = float(feat.get("log_eps", self.log_eps))

        # Instantiate torchaudio MelSpectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            f_min=self.f_min,
            f_max=self.f_max,
            n_mels=self.n_mels,
            power=self.power,
            center=True,
            pad_mode="reflect",
            norm="slaney",
            mel_scale="slaney",
        )

    def __call__(self, waveform: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Transforms a 16 kHz raw waveform tensor into a log-Mel tensor.

        Args:
            waveform: Float tensor of shape (samples,), (channels, samples),
                      or (batch, channels, samples).

        Returns:
            torch.Tensor: Log-Mel spectrogram of shape (batch, 80, time_frames).
        """
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform, dtype=torch.float32)
        else:
            waveform = waveform.to(dtype=torch.float32)

        # Standardize dimensionality to (batch, channels, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            # (channels, samples) -> (1, channels, samples)
            waveform = waveform.unsqueeze(0)

        # Downmix multi-channel audio to mono if necessary
        if waveform.size(1) > 1:
            waveform = waveform.mean(dim=1, keepdim=True)

        # Optional peak normalization per utterance
        if self.normalize_peak:
            peak = torch.max(torch.abs(waveform), dim=-1, keepdim=True)[0]
            peak = torch.clamp(peak, min=1e-7)
            waveform = waveform / peak

        # Compute power Mel spectrogram: (batch, 1, n_mels, time_frames)
        mel_power = self.mel_transform(waveform)

        # Squeeze channel dimension: (batch, n_mels, time_frames)
        mel_power = mel_power.squeeze(1)

        # Log compression: log(mel + eps)
        log_mel = torch.log(torch.clamp(mel_power, min=self.log_eps))

        return log_mel
