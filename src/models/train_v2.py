# Kaggle Environment Bootstrap (Run in Notebook Cell if needed):
# !pip install g2p_en jiwer torchaudio soundfile

"""Phase 3 Acoustic 1D-CNN Training Pipeline: Clinical-Grade Discriminative Ranker.

Project TIDE: Transcribing Impaired Dysarthric Edge-speech.

Architectural Advances in Phase 3:
1. Strict Clinical Speaker Splits:
   - Zero speaker leakage: Train (M02, F03, MC03, MC04, FC03),
     Val (F01, M04, MC02, FC02), Test (M01, F04, M05, MC01, FC01).
2. Discriminative PyTorch Dataset:
   - Yields 80-channel Log-Mel spectrograms alongside ground-truth phonemes
     and hard-negative phoneme sequences (mined Whisper errors or calibrated dysarthric distortions).
3. Margin-Based Contrastive CTC Objective:
   - L_total = L_CTC(y_true) + lambda * max(0, gamma - (L_CTC(y_whisper_err) - L_CTC(y_true)))
   - Actively penalizes the model when dysarthric substitutions (e.g. DAG vs DOG)
     yield comparable acoustic likelihood to the ground truth.
4. Validation PER Tracking & Greedy CTC Decoding:
   - Evaluates frame logits via argmax and collapsing duplicates/blanks.
   - Calculates true Phoneme Error Rate (PER) on clinical validation speakers (F01 & M04).
   - Saves 'best_1dcnn_v2.pth' strictly on lowest validation PER.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from torch.utils.data import DataLoader
import numpy as np
from jiwer import wer

from src.models.acoustic_1dcnn import (
    Acoustic1DCNN,
    PhonemeTokenizer,
    LogMelSpectrogramTransform,
    NUM_PHONEMES,
    ID_TO_PHONEME,
    CMUDICT_PHONEMES,
)
from src.data.torgo_dataset import (
    TORGODiscriminativeDataset,
    discriminative_collate_fn,
)


# =====================================================================
# 1. Margin-Based Contrastive CTC Loss
# =====================================================================

class ContrastiveCTCLoss(nn.Module):
    """Margin-based Contrastive CTC Loss for discriminative acoustic reranking.

    Formula:
        L_total = L_CTC(y_true) + lambda * max(0, gamma - (L_CTC(y_neg) - L_CTC(y_true)))
    """

    def __init__(
        self,
        blank: int = 0,
        margin: float = 2.0,
        lambda_contrast: float = 0.3,
        zero_infinity: bool = True,
    ):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, reduction="none", zero_infinity=zero_infinity)
        self.margin = margin
        self.lambda_contrast = lambda_contrast

    def forward(
        self,
        log_probs: torch.Tensor,       # (Time, Batch, Num_Phonemes)
        true_targets: torch.Tensor,    # (Batch, Max_True_Len)
        neg_targets: torch.Tensor,     # (Batch, Max_Neg_Len)
        input_lengths: torch.Tensor,   # (Batch,)
        true_lengths: torch.Tensor,    # (Batch,)
        neg_lengths: torch.Tensor,     # (Batch,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes total contrastive loss, true CTC loss, and margin penalty."""
        # Compute unreduced CTC loss for ground truth and negative candidate
        loss_true = self.ctc(log_probs, true_targets, input_lengths, true_lengths)
        loss_neg = self.ctc(log_probs, neg_targets, input_lengths, neg_lengths)

        # Margin contrastive penalty: penalize if (loss_neg - loss_true) < gamma
        margin_diff = loss_neg - loss_true
        contrastive_penalty = torch.clamp(self.margin - margin_diff, min=0.0)

        total_loss = loss_true + (self.lambda_contrast * contrastive_penalty)

        return total_loss.mean(), loss_true.mean(), contrastive_penalty.mean()


# =====================================================================
# 2. Greedy CTC Decoder & PER Calculation
# =====================================================================

def greedy_ctc_decode(logits: torch.Tensor, blank_id: int = 0) -> List[List[int]]:
    """Decodes frame-level logits (Batch, Time, Classes) via argmax, collapsing blanks and duplicates.

    Args:
        logits: Tensor of shape (Batch, Time, Classes)
        blank_id: Token ID for CTC blank (0)

    Returns:
        List of decoded phoneme integer lists for each sequence in the batch.
    """
    best_paths = torch.argmax(logits, dim=-1)  # (Batch, Time)
    batch_decoded: List[List[int]] = []

    for path in best_paths:
        tokens: List[int] = []
        prev: Optional[int] = None
        for tok in path.tolist():
            if tok != blank_id and tok != prev:
                tokens.append(tok)
            prev = tok
        batch_decoded.append(tokens)

    return batch_decoded


def compute_phoneme_error_rate(ref_tokens: List[int], hyp_tokens: List[int]) -> float:
    """Computes exact token-level Levenshtein Phoneme Error Rate (PER)."""
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    ref_str = " ".join(map(str, ref_tokens))
    hyp_str = " ".join(map(str, hyp_tokens)) if hyp_tokens else ""
    return float(wer(ref_str, hyp_s := hyp_str if hyp_str else "<empty>"))


# =====================================================================
# 3. Training & Validation Engines
# =====================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: ContrastiveCTCLoss,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> Dict[str, float]:
    """Runs a single training epoch with contrastive margin loss."""
    model.train()
    total_loss, total_true, total_contrast = 0.0, 0.0, 0.0
    num_batches = 0

    for batch in dataloader:
        if not batch or "mel_specs" not in batch:
            continue

        mels = batch["mel_specs"].to(device)
        in_lens = batch["input_lengths"].to(device)
        true_phns = batch["true_phonemes"].to(device)
        true_lens = batch["true_lengths"].to(device)
        neg_phns = batch["neg_phonemes"].to(device)
        neg_lens = batch["neg_lengths"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
            logits = model(mels)  # (Batch, Time, Num_Phonemes)
            # CTCLoss expects log_probs of shape (Time, Batch, Num_Phonemes)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

            loss, l_true, l_contrast = criterion(
                log_probs, true_phns, neg_phns, in_lens, true_lens, neg_lens
            )

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_true += l_true.item()
        total_contrast += l_contrast.item()
        num_batches += 1

    denom = max(1, num_batches)
    return {
        "loss_total": total_loss / denom,
        "loss_true": total_true / denom,
        "loss_contrast": total_contrast / denom,
    }


@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: ContrastiveCTCLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluates validation loss and computes greedy CTC Phoneme Error Rate (PER)."""
    model.eval()
    total_loss, num_batches = 0.0, 0

    all_pers: List[float] = []
    clinical_pers: List[float] = []  # F01 (Severe) and M04 (Moderate)

    for batch in dataloader:
        if not batch or "mel_specs" not in batch:
            continue

        mels = batch["mel_specs"].to(device)
        in_lens = batch["input_lengths"].to(device)
        true_phns = batch["true_phonemes"].to(device)
        true_lens = batch["true_lengths"].to(device)
        neg_phns = batch["neg_phonemes"].to(device)
        neg_lens = batch["neg_lengths"].to(device)
        metadata = batch["metadata"]

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
            logits = model(mels)  # (Batch, Time, Num_Phonemes)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

            loss, _, _ = criterion(log_probs, true_phns, neg_phns, in_lens, true_lens, neg_lens)

        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()
            num_batches += 1

        # Greedy CTC decoding on frame logits
        decoded_batch = greedy_ctc_decode(logits, blank_id=0)

        for i, hyp_tokens in enumerate(decoded_batch):
            ref_len = int(true_lens[i].item())
            ref_tokens = true_phns[i, :ref_len].tolist()
            per_val = compute_phoneme_error_rate(ref_tokens, hyp_tokens)
            all_pers.append(per_val)

            spk = metadata[i].get("speaker_id", "")
            if spk in {"F01", "M04"}:
                clinical_pers.append(per_val)

    mean_loss = total_loss / max(1, num_batches)
    mean_overall_per = float(np.mean(all_pers)) if all_pers else 1.0
    mean_clinical_per = float(np.mean(clinical_pers)) if clinical_pers else mean_overall_per

    return {
        "val_loss": mean_loss,
        "val_per_overall": mean_overall_per,
        "val_per_clinical": mean_clinical_per,
    }


# =====================================================================
# 4. Main Training Pipeline
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Project TIDE Phase 3: Contrastive Acoustic 1D-CNN Ranker")
    parser.add_argument("--train_manifest", type=str, default="data/processed/train/manifest.json", help="Path to train manifest")
    parser.add_argument("--val_manifest", type=str, default="data/processed/val/manifest.json", help="Path to val manifest")
    parser.add_argument("--whisper_negatives", type=str, default="", help="Optional path to cached Whisper hallucinations")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (adjusted on CPU)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--margin", type=float, default=2.0, help="Contrastive CTC margin gamma")
    parser.add_argument("--lambda_contrast", type=float, default=0.3, help="Contrastive loss weight lambda")
    parser.add_argument("--checkpoint_out", type=str, default="checkpoints/best_1dcnn_v2.pth", help="Output path for best checkpoint")
    parser.add_argument("--limit_samples", type=int, default=0, help="Optional limit on samples (for local dry runs)")
    args = parser.parse_args()

    print("=" * 80)
    print(" PROJECT TIDE PHASE 3: DISCRIMINATIVE 1D-CNN ACOUSTIC RANKER")
    print(" Contrastive Margin CTC + Validation PER Tracking")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        print(f"[OK] Active Compute: {gpu_count}x {gpu_name} (CUDA Accelerated)")
    else:
        print("[!] Operating on CPU (Training & Validation Engine).")

    # Resolve paths
    train_mf = Path(args.train_manifest)
    val_mf = Path(args.val_manifest)

    if not train_mf.exists() or not val_mf.exists():
        print(f"[!] Error: Manifests not found. Run 'python src/data/build_clinical_splits.py' first.")
        sys.exit(1)

    # Output directory
    out_ckpt = Path(args.checkpoint_out)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Datasets
    print(f"\n[*] Loading Train Dataset from {train_mf}...")
    train_dataset = TORGODiscriminativeDataset(
        train_mf,
        whisper_negatives_path=args.whisper_negatives if args.whisper_negatives else None
    )
    if args.limit_samples > 0:
        train_dataset.records = train_dataset.records[:args.limit_samples]
    print(f"[OK] Training utterances: {len(train_dataset):,}")

    print(f"[*] Loading Validation Dataset from {val_mf}...")
    val_dataset = TORGODiscriminativeDataset(val_mf)
    if args.limit_samples > 0:
        val_dataset.records = val_dataset.records[:max(4, args.limit_samples // 3)]
    print(f"[OK] Validation utterances: {len(val_dataset):,}")

    # 2. DataLoaders
    batch_size = args.batch_size if device.type == "cuda" else min(args.batch_size, 8)
    num_workers = 2 if device.type == "cuda" else 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=discriminative_collate_fn,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=discriminative_collate_fn,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 3. Model Initialization
    model = Acoustic1DCNN(in_features=80, num_classes=NUM_PHONEMES, hidden_dim=256).to(device)
    param_count = model.count_parameters()
    print(f"[OK] Acoustic1DCNN initialized with {param_count:,} parameters (Budget: <1.5M).")

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"[OK] Enabling DataParallel across {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    # 4. Objective, Optimizer & Scaler
    criterion = ContrastiveCTCLoss(
        blank=0,
        margin=args.margin,
        lambda_contrast=args.lambda_contrast,
        zero_infinity=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # 5. Training Loop with PER Tracking
    best_val_per = float("inf")
    print(f"\n[*] Commencing Phase 3 Contrastive Training ({args.epochs} epochs)...")
    print(f"    Margin gamma={args.margin}, Lambda={args.lambda_contrast}")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_res = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_res = evaluate_validation(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        # Save checkpoint strictly on lowest clinical validation PER (F01 & M04)
        target_per = val_res["val_per_clinical"]
        is_best = target_per < best_val_per

        if is_best:
            best_val_per = target_per
            raw_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": raw_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_res["val_loss"],
                    "val_per_clinical": val_res["val_per_clinical"],
                    "val_per_overall": val_res["val_per_overall"],
                    "phonemes": CMUDICT_PHONEMES,
                    "margin": args.margin,
                    "lambda_contrast": args.lambda_contrast,
                },
                out_ckpt,
            )

        marker = f"* (Saved Best PER: {best_val_per*100:.1f}%)" if is_best else ""
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_res['loss_total']:.3f} (True: {train_res['loss_true']:.3f}, Pen: {train_res['loss_contrast']:.3f}) | "
            f"Val Loss: {val_res['val_loss']:.3f} | "
            f"Clin PER: {val_res['val_per_clinical']*100:.1f}% | "
            f"Over PER: {val_res['val_per_overall']*100:.1f}% | "
            f"Time: {elapsed:.1f}s {marker}"
        )

    print("\n" + "=" * 80)
    print(f"[OK] Phase 3 Training Complete.")
    print(f"[OK] Best Checkpoint Saved to: {out_ckpt} (Lowest Clinical PER: {best_val_per*100:.2f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
