import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from jiwer import wer

sys.path.insert(0, os.path.abspath("."))
from src.models.acoustic_1dcnn import Acoustic1DCNN, NUM_PHONEMES
from src.data.torgo_dataset import DysarthricContrastiveDataset, collate_contrastive_batch

def greedy_decode(log_probs_2d: torch.Tensor, blank_id: int = 0):
    preds = torch.argmax(log_probs_2d, dim=-1).cpu().numpy()
    decoded, prev = [], None
    for p in preds:
        if p != blank_id and p != prev:
            decoded.append(int(p))
        prev = p
    return decoded

def main():
    print("=" * 80)
    print(" PHASE 3: CONTRASTIVE 1D-CNN TRAINING (SPEAKER-STRATIFIED CLINICAL SPLIT)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Hardware: {device.type.upper()}")

    train_ds = DysarthricContrastiveDataset("data/processed/train/manifest.json")
    val_ds = DysarthricContrastiveDataset("data/processed/val/manifest.json")

    batch_size = 8 if device.type == "cuda" else 4
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_contrastive_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_contrastive_batch)

    model = Acoustic1DCNN(in_features=80, num_classes=NUM_PHONEMES, hidden_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True, reduction="none")

    EPOCHS, MARGIN, LAMBDA = 20, 2.0, 0.3
    best_val_per = float("inf")
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, t0 = 0.0, time.time()

        for mels, mel_lens, true_p, true_len, neg_p, neg_len in train_loader:
            mels, mel_lens = mels.to(device), mel_lens.to(device)
            true_p, true_len = true_p.to(device), true_len.to(device)
            neg_p, neg_len = neg_p.to(device), neg_len.to(device)

            optimizer.zero_grad()
            logits = model(mels)  # (Batch, Time, Vocab)
            log_probs = F.log_softmax(logits.transpose(0, 1).contiguous(), dim=-1)  # (Time, Batch, Vocab)

            loss_true = ctc_loss_fn(log_probs, true_p, mel_lens, true_len)
            loss_neg = ctc_loss_fn(log_probs, neg_p, mel_lens, neg_len)

            margin_penalty = torch.clamp(MARGIN - (loss_neg - loss_true), min=0.0).mean()
            loss = loss_true.mean() + (LAMBDA * margin_penalty)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_pers = []
        with torch.no_grad():
            for mels, mel_lens, true_p, true_len, _, _ in val_loader:
                mels = mels.to(device)
                logits = model(mels)
                log_probs = F.log_softmax(logits.transpose(0, 1).contiguous(), dim=-1)

                for i in range(mels.size(0)):
                    valid_T = mel_lens[i].item()
                    pred = greedy_decode(log_probs[:valid_T, i, :])
                    ref = true_p[i][:true_len[i]].cpu().tolist()
                    ref_str = " ".join(map(str, ref))
                    pred_str = " ".join(map(str, pred)) if pred else "<empty>"
                    if ref_str.strip():
                        val_pers.append(float(wer(ref_str, pred_str)))

        avg_per = (sum(val_pers) / max(1, len(val_pers))) * 100
        save_tag = ""
        if avg_per < best_val_per:
            best_val_per = avg_per
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_per": best_val_per,
                "margin": MARGIN,
                "lambda": LAMBDA,
            }, "checkpoints/best_1dcnn_v2.pth")
            save_tag = " --> [SAVED checkpoints/best_1dcnn_v2.pth]"

        print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {total_loss/max(1, len(train_loader)):.3f} | Val PER: {avg_per:.2f}% | {time.time()-t0:.0f}s{save_tag}")

if __name__ == "__main__":
    main()
