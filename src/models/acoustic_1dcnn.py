"""Acoustic 1D-CNN Phonetic Alignment Model for Project TIDE.

Provides:
1. DepthwiseSeparableConv1d and Acoustic1DCNN PyTorch architectures (<1.5M parameters).
2. CMUDict 43-phoneme vocabulary and PhonemeTokenizer using g2p_en.
3. In-memory 80-channel Log-Mel spectrogram transform with per-channel normalization.
4. Checkpoint loader and AcousticScorer for CTC alignment rescoring.
"""

import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import soundfile as sf
import numpy as np

# CMUDict 39 phonemes + blank (CTC 0) + special tokens = 43 total classes
CMUDICT_PHONEMES = [
    "<blank>",  # Index 0 is strictly reserved for CTC blank token
    "<pad>",
    "<unk>",
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T",
    "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH", " "
]

PHONEME_TO_ID: Dict[str, int] = {p: idx for idx, p in enumerate(CMUDICT_PHONEMES)}
ID_TO_PHONEME: Dict[int, str] = {idx: p for idx, p in enumerate(CMUDICT_PHONEMES)}
NUM_PHONEMES: int = len(CMUDICT_PHONEMES)  # 43


class PhonemeTokenizer:
    """Converts orthographic text into integer phoneme sequences via G2P."""

    def __init__(self):
        try:
            from g2p_en import G2p
            self.g2p = G2p()
        except ImportError:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "g2p_en"])
            from g2p_en import G2p
            self.g2p = G2p()

    def encode(self, text: str) -> List[int]:
        """Converts raw text transcript into list of integer phoneme IDs."""
        clean_text = text.strip()
        if not clean_text:
            return []

        raw_phonemes = self.g2p(clean_text)
        token_ids: List[int] = []

        for p in raw_phonemes:
            # Strip numeric stress digits (e.g., 'AH0' -> 'AH')
            p_clean = re.sub(r"\d+", "", p).upper()
            if p_clean in PHONEME_TO_ID:
                token_ids.append(PHONEME_TO_ID[p_clean])
            elif p == " ":
                token_ids.append(PHONEME_TO_ID[" "])
            else:
                token_ids.append(PHONEME_TO_ID["<unk>"])

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decodes integer phoneme sequence into human-readable string."""
        return " ".join([ID_TO_PHONEME.get(idx, "<unk>") for idx in token_ids if idx > 0])


class LogMelSpectrogramTransform(nn.Module):
    """Computes standardized 80-channel log-Mel spectrograms purely in RAM."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            f_min=0.0,
            f_max=sample_rate // 2,
            n_mels=n_mels,
            power=2.0,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Forward pass converting raw waveform (1, Time) or (Batch, 1, Time) to Log-Mel (Batch, 80, Frames)."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (1, Time)
        elif waveform.dim() == 3:
            # (Batch, Channels, Time) -> downmix channels to mono
            if waveform.size(1) > 1:
                waveform = waveform.mean(dim=1)
            else:
                waveform = waveform.squeeze(1)
        # At this point, waveform is (Batch, Time)

        # Compute mel spectrogram: (Batch, n_mels, time_frames)
        mel = self.mel_transform(waveform)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))

        # Standardize normalization across frequency channels
        mean = log_mel.mean(dim=-1, keepdim=True)
        std = log_mel.std(dim=-1, keepdim=True) + 1e-6
        return (log_mel - mean) / std


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise-Separable 1D Convolution block with BatchNorm, GELU, and Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out = self.act(out)
        return self.dropout(out)


class Acoustic1DCNN(nn.Module):
    """Lightweight 1D-CNN Phonetic Alignment Model for Edge Deployment.

    Input: Log-Mel spectrogram (Batch, 80, Time)
    Output: Phoneme Logits (Batch, Time, Num_Phonemes)
    Total Parameters: ~1.2M (< 1.5M budget)
    """

    def __init__(
        self,
        in_features: int = 80,
        num_classes: int = NUM_PHONEMES,
        hidden_dim: int = 256
    ):
        super().__init__()

        # Initial stem projection: (B, 80, T) -> (B, 256, T)
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Depthwise-Separable Stack (4 Blocks with Residual Stages)
        self.block1 = DepthwiseSeparableConv1d(hidden_dim, hidden_dim, kernel_size=5)
        self.block2 = DepthwiseSeparableConv1d(hidden_dim, 384, kernel_size=5)
        self.block3 = DepthwiseSeparableConv1d(384, 384, kernel_size=5)
        self.block4 = DepthwiseSeparableConv1d(384, hidden_dim, kernel_size=3)

        # Skip connection adapters
        self.skip_adapter2 = nn.Conv1d(hidden_dim, 384, kernel_size=1)
        self.skip_adapter4 = nn.Conv1d(384, hidden_dim, kernel_size=1)

        # Final Classifier Head: (B, 256, T) -> (B, Num_Phonemes, T)
        self.classifier = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Log-Mel spectrogram tensor of shape (Batch, 80, Time).

        Returns:
            Logits of shape (Batch, Time, Num_Phonemes).
        """
        # Stem
        h0 = self.stem(x)  # (B, 256, T)

        # Block 1 with residual
        h1 = self.block1(h0) + h0  # (B, 256, T)

        # Block 2 with projected residual
        h2 = self.block2(h1) + self.skip_adapter2(h1)  # (B, 384, T)

        # Block 3 with residual
        h3 = self.block3(h2) + h2  # (B, 384, T)

        # Block 4 with projected residual
        h4 = self.block4(h3) + self.skip_adapter4(h3)  # (B, 256, T)

        # Classifier: (B, Num_Phonemes, T)
        logits = self.classifier(h4)

        # Transpose to (Batch, Time, Num_Phonemes)
        return logits.transpose(1, 2)

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_acoustic_model(
    checkpoint_path: Union[str, Path] = "checkpoints/best_1dcnn.pth",
    device: str = "cpu"
) -> Tuple[Acoustic1DCNN, Dict[str, Any]]:
    """Loads trained Acoustic1DCNN from checkpoint on specified device.

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        device: Device to load model onto ('cpu' or 'cuda').

    Returns:
        Tuple of (model, checkpoint_dict).
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    model = Acoustic1DCNN(
        in_features=80,
        num_classes=NUM_PHONEMES,
        hidden_dim=256
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint


class AcousticScorer:
    """Computes neural CTC alignment scores for candidate transcripts against audio in memory."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "checkpoints/best_1dcnn.pth",
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model, self.checkpoint = load_acoustic_model(checkpoint_path, device=device)
        self.tokenizer = PhonemeTokenizer()
        self.mel_extractor = LogMelSpectrogramTransform()
        self.ctc_loss_fn = nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)

    @torch.no_grad()
    def score_candidates(
        self,
        waveform: torch.Tensor,
        candidates: List[str],
        sample_rate: int = 16000
    ) -> List[Dict[str, Any]]:
        """Computes CTC alignment loss and normalized acoustic probability score for candidates.

        Args:
            waveform: Float audio tensor (Time,) or (1, Time).
            candidates: List of candidate transcript strings.
            sample_rate: Audio sampling rate (resampled to 16kHz if different).

        Returns:
            List of dicts containing:
                - transcript: Candidate transcript
                - ctc_loss: Raw CTC loss
                - target_len: Phoneme sequence length
                - norm_loss: Length-normalized CTC loss (ctc_loss / target_len)
                - acoustic_score: Normalized [0, 1] acoustic score
        """
        if sample_rate != 16000:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        waveform = waveform.to(self.device)
        mel = self.mel_extractor(waveform)  # (1, 80, Time_frames)

        # Forward pass through Acoustic1DCNN
        logits = self.model(mel)  # (1, Time, Num_Phonemes)
        # CTCLoss expects log_probs of shape (Time, Batch, Num_Phonemes)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

        t_frames = log_probs.size(0)
        input_lengths = torch.tensor([t_frames], dtype=torch.long, device=self.device)

        results: List[Dict[str, Any]] = []
        norm_losses: List[float] = []

        for cand_text in candidates:
            target_ids = self.tokenizer.encode(cand_text)
            if not target_ids:
                loss_val = 999.0
                target_len = 1
            else:
                target_len = len(target_ids)
                target_tensor = torch.tensor(target_ids, dtype=torch.long, device=self.device).unsqueeze(0)
                target_lengths = torch.tensor([target_len], dtype=torch.long, device=self.device)
                loss_val = float(self.ctc_loss_fn(log_probs, target_tensor, input_lengths, target_lengths).item())

            norm_loss = loss_val / max(1, target_len)
            norm_losses.append(norm_loss)
            results.append({
                "transcript": cand_text,
                "ctc_loss": round(loss_val, 3),
                "target_len": target_len,
                "norm_loss": round(norm_loss, 4),
            })

        # Length-normalized negative CTC loss normalized into [0, 1] relative to best candidate:
        # S_Acoustic = exp(-(norm_loss - min_norm_loss)) = exp(-norm_loss) / max(exp(-norm_loss))
        min_norm = min(norm_losses) if norm_losses else 0.0
        for item, n_loss in zip(results, norm_losses):
            s_acous = float(np.exp(-(n_loss - min_norm)))
            item["acoustic_score"] = round(float(np.clip(s_acous, 0.0, 1.0)), 4)

        return results
