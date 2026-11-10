import os
import sys
import argparse
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import jiwer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.squeezeformer_ctc import SqueezeformerCTC, VOCAB
from src.data.torgo_dataset import TorgoCTCDataset, collate_ctc_batch


def evaluate(
    model: SqueezeformerCTC,
    dataloader: DataLoader,
    criterion: nn.CTCLoss,
    device: torch.device,
    max_batches: int = 0,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_refs = []
    batches_run = 0

    with torch.no_grad():
        for step, (features, feat_lens, targets, targ_lens, refs) in enumerate(dataloader, start=1):
            features = features.to(device)
            feat_lens = feat_lens.to(device)
            targets = targets.to(device)
            targ_lens = targ_lens.to(device)

            logits, out_lens = model(features, feat_lens)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)

            loss = criterion(log_probs, targets, out_lens, targ_lens)
            total_loss += loss.item()
            batches_run += 1

            preds = model.decode_greedy(logits, out_lens)
            all_preds.extend(preds)
            all_refs.extend(refs)

            if max_batches > 0 and step >= max_batches:
                break

    avg_loss = total_loss / max(1, batches_run)
    valid_pairs = [(r, p) for r, p in zip(all_refs, all_preds) if len(r.strip()) > 0]
    if valid_pairs:
        clean_refs, clean_preds = zip(*valid_pairs)
        cer = float(jiwer.cer(list(clean_refs), list(clean_preds)))
        wer = float(jiwer.wer(list(clean_refs), list(clean_preds)))
    else:
        cer, wer = 1.0, 1.0

    return avg_loss, cer, wer


def run_smoke_test(device: torch.device) -> None:
    print("[*] Running synthetic smoke test for SqueezeformerCTC + CTCLoss...")
    model = SqueezeformerCTC(num_classes=len(VOCAB)).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    features = torch.randn(2, 240, 80, device=device)
    feat_lens = torch.LongTensor([240, 180]).to(device)
    targets = torch.LongTensor([[2, 3, 4, 1, 5, 6], [7, 8, 9, 0, 0, 0]]).to(device)
    targ_lens = torch.LongTensor([6, 3]).to(device)

    model.train()
    for step in range(1, 3):
        optimizer.zero_grad()
        logits, out_lens = model(features, feat_lens)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)
        loss = criterion(log_probs, targets, out_lens, targ_lens)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        preds = model.decode_greedy(logits, out_lens)
        print(f"  [Step {step}] Loss: {loss.item():.4f} | Out Lens: {out_lens.tolist()} | Decoded: {preds}")

    beam_out = model.decode_beam(logits[:1], out_lens[:1], beam_width=5)
    print(f"  [Beam Decode] Top-1: '{beam_out['top1_transcript']}' | Confidence: {beam_out['confidence_metrics']}")
    print("[+] Smoke test passed! Forward, CTCLoss, backward, and beam decode are verified.")


def main():
    parser = argparse.ArgumentParser(description="Train Squeezeformer CTC on TORGO")
    parser.add_argument("--train-manifest", type=str, default="data/processed/train/manifest.json")
    parser.add_argument("--val-manifest", type=str, default="data/processed/val/manifest.json")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--encoder-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--max-batches", type=int, default=0, help="Limit batches per epoch for local dry-run")
    parser.add_argument("--smoke-test", action="store_true", help="Run 2 synthetic steps and exit")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing SqueezeformerCTC training on device: {device}")

    if args.smoke_test:
        run_smoke_test(device)
        return

    train_ds = TorgoCTCDataset(args.train_manifest, augment=True)
    val_ds = TorgoCTCDataset(args.val_manifest, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_ctc_batch,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_ctc_batch,
        num_workers=2 if device.type == "cuda" else 0,
    )

    model = SqueezeformerCTC(
        input_dim=80,
        num_classes=len(VOCAB),
        encoder_dim=args.encoder_dim,
        num_layers=args.num_layers,
    ).to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps_per_epoch = args.max_batches if args.max_batches > 0 else max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        pct_start=0.1,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_cer = float("inf")

    print(f"[*] Starting training: {len(train_ds)} train samples | {len(val_ds)} val samples")

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        train_loss = 0.0
        batches_run = 0

        for step, (features, feat_lens, targets, targ_lens, _) in enumerate(train_loader, start=1):
            features = features.to(device)
            feat_lens = feat_lens.to(device)
            targets = targets.to(device)
            targ_lens = targ_lens.to(device)

            optimizer.zero_grad()
            logits, out_lens = model(features, feat_lens)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)

            loss = criterion(log_probs, targets, out_lens, targ_lens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            batches_run += 1

            if args.max_batches > 0 and step >= args.max_batches:
                break

        avg_train_loss = train_loss / max(1, batches_run)
        val_loss, val_cer, val_wer = evaluate(model, val_loader, criterion, device, max_batches=args.max_batches)
        elapsed = time.perf_counter() - t0

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} [{elapsed:.1f}s] | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val CER: {val_cer:.4f} | Val WER: {val_wer:.4f}"
        )

        if val_cer <= best_cer:
            best_cer = val_cer
            ckpt_path = save_dir / "best_squeezeformer_ctc.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_cer": val_cer,
                    "val_wer": val_wer,
                    "encoder_dim": args.encoder_dim,
                    "num_layers": args.num_layers,
                },
                ckpt_path,
            )
            print(f"  [+] Saved best checkpoint to {ckpt_path} (CER: {best_cer:.4f})")


if __name__ == "__main__":
    main()
