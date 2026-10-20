# Kaggle Environment Bootstrap (Run in Notebook Cell if needed):
# !pip install g2p_en torchaudio soundfile

"""Acoustic 1D-CNN Training Script for Project TIDE (Phase 2).

Designed for direct execution inside a Kaggle Notebook GPU Environment (NVIDIA T4 / P100).

Architectural Highlights:
1. Custom PyTorch TORGO Dataset:
   - Recursively traverses dysarthric and control speaker subdirectories (F_Dys, M_Con, M_Dys, etc.).
   - Extracts 80-channel log-Mel spectrograms directly in RAM (Zero Disk Footprint).
   - Maps transcripts to CMUDict phoneme sequences using g2p_en.
2. Lightweight Depthwise-Separable 1D-CNN:
   - Input: Log-Mel spectrograms (Batch, 80, Time).
   - Architecture: Depthwise-Separable Convolutions + BatchNorm1d + GELU + Residual connections.
   - Total Parameters: < 1.5M (Edge-deployable budget for Jetson/RK3588).
   - Output: Frame-level phoneme posteriors (Batch, Time, Num_Phonemes).
3. CTC Loss Alignment:
   - torch.nn.CTCLoss aligns unsegmented acoustic frames to phoneme targets without manual timestamps.
4. GPU-Accelerated Training Loop:
   - Mixed Precision (torch.cuda.amp.autocast + GradScaler).
   - AdamW optimizer with Cosine Annealing learning rate schedule.
   - Saves checkpoint to /kaggle/working/best_1dcnn.pth.
"""

import os
import sys
import re
import glob
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Enforce UTF-8 stdout on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import numpy as np

# Ensure g2p_en is accessible
try:
    from g2p_en import G2p
except ImportError:
    print("[!] Installing g2p_en dependency...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "g2p_en"])
    from g2p_en import G2p


# =====================================================================
# 1. Phoneme Vocabulary & Tokenizer
# =====================================================================

# Standard CMUDict 39 phonemes + blank (CTC 0) + special tokens
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
NUM_PHONEMES: int = len(CMUDICT_PHONEMES)


class PhonemeTokenizer:
    """Converts orthographic text into integer phoneme sequences via G2P."""

    def __init__(self):
        self.g2p = G2p()

    def encode(self, text: str) -> List[int]:
        """Converts raw text transcript into list of integer phoneme IDs."""
        clean_text = text.strip()
        if not clean_text:
            return []

        # Run G2P
        raw_phonemes = self.g2p(clean_text)
        token_ids: List[int] = []

        for p in raw_phonemes:
            # Strip numeric stress digits (e.g., 'AH0' -> 'AH', 'IY1' -> 'IY')
            p_clean = re.sub(r"\d+", "", p).upper()
            if p_clean in PHONEME_TO_ID:
                token_ids.append(PHONEME_TO_ID[p_clean])
            elif p == " ":
                token_ids.append(PHONEME_TO_ID[" "])
            else:
                # Character fallback
                token_ids.append(PHONEME_TO_ID["<unk>"])

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decodes integer phoneme sequence into human-readable string."""
        return " ".join([ID_TO_PHONEME.get(idx, "<unk>") for idx in token_ids if idx > 0])


# =====================================================================
# 2. In-Memory Audio Feature Extractor
# =====================================================================

class LogMelSpectrogramTransform(nn.Module):
    """Computes standardized 80-channel log-Mel spectrograms purely in RAM."""

    def __init__(self, sample_rate: int = 16000, n_mels: int = 80, n_fft: int = 400, hop_length: int = 160):
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
        """Forward pass converting raw waveform (1, Time) to Log-Mel (80, Frames)."""
        mel = self.mel_transform(waveform)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        # Standardize normalization across frequency channels
        mean = log_mel.mean(dim=-1, keepdim=True)
        std = log_mel.std(dim=-1, keepdim=True) + 1e-6
        return (log_mel - mean) / std


# =====================================================================
# 3. Custom TORGO PyTorch Dataset & Collate Function
# =====================================================================

class TORGOAcousticDataset(Dataset):
    """PyTorch Dataset traversing Kaggle TORGO dataset directory hierarchy."""

    def __init__(self, root_dir: str, sample_rate: int = 16000, max_duration_sec: float = 12.0):
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.max_duration_sec = max_duration_sec
        self.tokenizer = PhonemeTokenizer()
        self.mel_extractor = LogMelSpectrogramTransform(sample_rate=sample_rate)

        self.samples: List[Tuple[str, str]] = self._discover_samples()

    def _discover_samples(self) -> List[Tuple[str, str]]:
        """Discovers (audio_path, transcript) pairs across TORGO directory structure."""
        discovered: List[Tuple[str, str]] = []

        if not self.root_dir.exists():
            print(f"[!] Warning: Root path {self.root_dir} does not exist.")
            return discovered

        # Find all .wav files recursively
        all_wavs = glob.glob(str(self.root_dir / "**" / "*.wav"), recursive=True)
        print(f"[*] Discovered {len(all_wavs)} total WAV files in {self.root_dir}...")

        for wav_path in all_wavs:
            p = Path(wav_path)
            # Enforce head-mic channel priority (reject arrayMic if present)
            if "arrayMic" in p.parts or "array_mic" in p.parts:
                continue

            # Check candidate transcript text file paths
            # Pattern A: Same basename with .txt (e.g., 001.wav -> 001.txt)
            txt_candidate = p.with_suffix(".txt")
            # Pattern B: Sibling prompts directory (e.g., .../wav_headMic/001.wav -> .../prompts/001.txt)
            prompt_candidate = p.parent.parent / "prompts" / f"{p.stem}.txt"

            transcript_text = ""
            if txt_candidate.exists():
                try:
                    transcript_text = txt_candidate.read_text(encoding="utf-8", errors="ignore").strip()
                except Exception:
                    pass
            elif prompt_candidate.exists():
                try:
                    transcript_text = prompt_candidate.read_text(encoding="utf-8", errors="ignore").strip()
                except Exception:
                    pass

            # Clean transcript: strip annotations like [cough], [groan]
            if transcript_text:
                clean_text = re.sub(r"\[.*?\]|\(.*?\)", "", transcript_text)
                clean_text = re.sub(r"[^A-Za-z\s']", "", clean_text).strip().upper()
                if clean_text:
                    discovered.append((str(p), clean_text))

        print(f"[OK] Successfully indexed {len(discovered)} valid head-mic audio-text pairs.")
        return discovered

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        wav_path, transcript = self.samples[idx]

        try:
            # Load audio via soundfile
            audio_np, sr = sf.read(wav_path, dtype="float32")
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)  # Convert stereo to mono

            # Resample if necessary
            waveform = torch.from_numpy(audio_np).unsqueeze(0)  # (1, T)
            if sr != self.sample_rate:
                resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)

            # Cap duration to prevent out-of-memory
            max_samples = int(self.sample_rate * self.max_duration_sec)
            if waveform.size(1) > max_samples:
                waveform = waveform[:, :max_samples]

            # In-memory Log-Mel Spectrogram extraction: shape (80, Time)
            with torch.no_grad():
                mel_spec = self.mel_extractor(waveform).squeeze(0)  # (80, Time)

            # Convert text transcript to phoneme targets
            phoneme_ids = self.tokenizer.encode(transcript)
            if not phoneme_ids:
                return None

            targets = torch.tensor(phoneme_ids, dtype=torch.long)
            return mel_spec, targets

        except Exception as e:
            return None


def ctc_collate_fn(
    batch: List[Optional[Tuple[torch.Tensor, torch.Tensor]]]
) -> Dict[str, torch.Tensor]:
    """Custom collator padding variable-length Mel spectrograms and phoneme targets."""
    # Filter out any corrupted samples
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        return {}

    # Sort batch descending by Mel time frames (improves efficiency)
    valid_batch.sort(key=lambda x: x[0].size(1), reverse=True)

    mel_specs, targets = zip(*valid_batch)

    batch_size = len(valid_batch)
    n_mels = mel_specs[0].size(0)
    input_lengths = torch.tensor([m.size(1) for m in mel_specs], dtype=torch.long)
    target_lengths = torch.tensor([t.size(0) for t in targets], dtype=torch.long)

    # Pad Mel spectrograms: (Batch, 80, Max_Time)
    max_time = int(input_lengths.max().item())
    padded_mels = torch.zeros(batch_size, n_mels, max_time, dtype=torch.float32)
    for i, m in enumerate(mel_specs):
        padded_mels[i, :, :m.size(1)] = m

    # Pad Targets: (Batch, Max_Target_Length)
    max_target_len = int(target_lengths.max().item())
    padded_targets = torch.zeros(batch_size, max_target_len, dtype=torch.long)
    for i, t in enumerate(targets):
        padded_targets[i, :t.size(0)] = t

    return {
        "mel_specs": padded_mels,        # (Batch, 80, Max_Time)
        "targets": padded_targets,        # (Batch, Max_Target_Len)
        "input_lengths": input_lengths,  # (Batch,)
        "target_lengths": target_lengths # (Batch,)
    }


# =====================================================================
# 4. Depthwise-Separable 1D-CNN Model (< 1.5M Parameters)
# =====================================================================

class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise-Separable 1D Convolution block with BatchNorm, GELU, and Dropout."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, dropout: float = 0.1):
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
    """

    def __init__(self, in_features: int = 80, num_classes: int = NUM_PHONEMES, hidden_dim: int = 256):
        super().__init__()

        # Initial stem projection: (B, 80, T) -> (B, 256, T)
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Depthwise-Separable Stack (3 Residual Stages)
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
            Log-probabilities of shape (Batch, Time, Num_Phonemes).
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

        # Transpose to (Batch, Time, Num_Phonemes) for batch_first CTCLoss
        return logits.transpose(1, 2)

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =====================================================================
# 5. Training & Validation Engine
# =====================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CTCLoss,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> float:
    """Runs a single training epoch using Mixed Precision."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        if not batch or "mel_specs" not in batch:
            continue

        mels = batch["mel_specs"].to(device)
        targets = batch["targets"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
            logits = model(mels)  # (Batch, Time, Num_Phonemes)
            # CTCLoss expects log_probs of shape (Time, Batch, Num_Phonemes)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

        if torch.isnan(loss) or torch.isinf(loss):
            print("[!] Warning: Loss exploded to NaN/Inf. Skipping batch.")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.CTCLoss,
    device: torch.device,
) -> float:
    """Evaluates validation loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        if not batch or "mel_specs" not in batch:
            continue

        mels = batch["mel_specs"].to(device)
        targets = batch["targets"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
            logits = model(mels)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(1, num_batches)


# =====================================================================
# 6. Main Execution Pipeline
# =====================================================================

def main() -> None:
    print("=" * 75)
    print(" PROJECT TIDE: ACOUSTIC 1D-CNN PHONETIC ALIGNMENT TRAINING")
    print(" Dual T4 GPU Mixed-Precision Pipeline (Kaggle Environment)")
    print("=" * 75)

    # 1. Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        print(f"[OK] Active Compute: {gpu_count}x {gpu_name} (CUDA Enabled)")
    else:
        print("[!] CUDA not detected. Operating on CPU.")

    # 2. Dataset Path Discovery (Kaggle vs Local Fallback)
    kaggle_paths = [
        "/kaggle/input/torgo-dataset-for-dysarthric-speech",
        "/kaggle/input/torgo-dataset",
        "/kaggle/input/torgo",
        "/kaggle/input",
    ]
    data_dir = None
    for p in kaggle_paths:
        if os.path.exists(p):
            data_dir = p
            break

    # Local fallback for development testing
    if data_dir is None:
        local_candidate = Path(__file__).resolve().parent.parent.parent / "data"
        data_dir = str(local_candidate)
        print(f"[*] Running outside Kaggle. Using local data path: {data_dir}")
    else:
        print(f"[OK] Target Kaggle Dataset Path: {data_dir}")

    # Output checkpoint path
    output_dir = Path("/kaggle/working") if os.path.exists("/kaggle") else Path("checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best_1dcnn.pth"

    # 3. Initialize Dataset
    dataset = TORGOAcousticDataset(root_dir=data_dir, sample_rate=16000)

    # If dataset has no files (e.g. initial dry run or unmounted dir), provide synthetic demo set
    if len(dataset) == 0:
        print("\n[!] Notice: No audio files found in directory. Creating mock acoustic samples for trial run...")
        mock_samples = []
        for i in range(16):
            # Synthetic 2-second spectrograms and target labels
            m_time = np.random.randint(120, 250)
            mock_mel = torch.randn(80, m_time)
            mock_targets = torch.randint(1, NUM_PHONEMES, (np.random.randint(10, 25),), dtype=torch.long)
            mock_samples.append((mock_mel, mock_targets))

        train_data = mock_samples[:12]
        val_data = mock_samples[12:]
    else:
        # 80/20 Train/Validation Split
        total_len = len(dataset)
        train_len = int(0.8 * total_len)
        val_len = total_len - train_len
        train_data, val_data = random_split(
            dataset,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(42)
        )
        print(f"[OK] Split: {train_len} Training samples, {val_len} Validation samples.")

    # 4. DataLoaders
    batch_size = 16 if device.type == "cuda" else 4
    num_workers = 2 if device.type == "cuda" else 0

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ctc_collate_fn,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ctc_collate_fn,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    # 5. Model Initialization
    model = Acoustic1DCNN(in_features=80, num_classes=NUM_PHONEMES, hidden_dim=256).to(device)
    param_count = model.count_parameters()
    print(f"[OK] Acoustic1DCNN initialized with {param_count:,} parameters (Budget: <1.5M).")
    assert param_count < 1_500_000, f"Parameter count {param_count} exceeds 1.5M ceiling!"

    # Multi-GPU DataParallel if dual GPUs present
    if torch.cuda.device_count() > 1:
        print(f"[OK] Enabling torch.nn.DataParallel across {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    # 6. Loss, Optimizer & AMP Scaler
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # 7. Training Loop (5 Trial Epochs)
    num_epochs = 5
    best_val_loss = float("inf")
    print(f"\n[*] Commencing {num_epochs}-epoch trial training loop...\n")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "phonemes": CMUDICT_PHONEMES,
                },
                best_model_path
            )

        marker = "* (Saved Best)" if is_best else ""
        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {elapsed:.1f}s {marker}"
        )

    print("\n" + "=" * 75)
    print(f"[OK] Phase 2 Trial Training Complete.")
    print(f"[OK] Best Model Saved to: {best_model_path} (Val Loss: {best_val_loss:.4f})")
    print("=" * 75)


if __name__ == "__main__":
    main()
