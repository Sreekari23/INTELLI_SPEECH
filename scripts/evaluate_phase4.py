import sys, os, hashlib
sys.path.insert(0, os.path.abspath("."))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from jiwer import wer

from src.models.acoustic_1dcnn import Acoustic1DCNN, NUM_PHONEMES
from src.data.torgo_dataset import DysarthricContrastiveDataset, collate_contrastive_batch

def greedy_decode(log_probs_2d, blank_id=0):
    preds = torch.argmax(log_probs_2d, dim=-1).cpu().numpy()
    decoded, prev = [], None
    for p in preds:
        if p != blank_id and p != prev:
            decoded.append(int(p))
        prev = p
    return decoded

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()[:16]

def main():
    print("=" * 80)
    print(" PHASE 4: HELD-OUT TEST EVALUATION (UNPADDED PER & CONTRASTIVE RANKING)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "checkpoints/best_1dcnn_v2.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/best_1dcnn.pth"

    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"[*] Loaded Checkpoint: {ckpt_path} (SHA256: {sha256_file(ckpt_path)})")
    if isinstance(ckpt, dict):
        print(f"[*] Checkpoint Metadata: Epoch={ckpt.get('epoch')} | Val_PER_Clinical={ckpt.get('val_per_clinical', ckpt.get('val_per', 'N/A'))}")

    model = Acoustic1DCNN(in_features=80, num_classes=NUM_PHONEMES, hidden_dim=256).to(device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True, reduction="none")

    test_ds = DysarthricContrastiveDataset("data/processed/test/manifest.json")
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_contrastive_batch)

    pers_unpadded = []
    pers_padded = []
    ranker_wins = 0
    total_rank_trials = 0

    with torch.no_grad():
        for mels, mel_lens, true_p, true_len, neg_p, neg_len in test_loader:
            mels, mel_lens = mels.to(device), mel_lens.to(device)
            true_p, true_len = true_p.to(device), true_len.to(device)
            neg_p, neg_len = neg_p.to(device), neg_len.to(device)

            # mels is already normalized (Batch, 80, T_max) from DysarthricContrastiveDataset
            logits = model(mels)  # (Batch, T_max, Vocab)
            log_probs = F.log_softmax(logits.transpose(0, 1).contiguous(), dim=-1)  # (T_max, Batch, Vocab)

            # 1. Evaluate Contrastive Ranker Discrimination (True Phonemes vs. Hallucinated Phonemes)
            loss_true = ctc_loss_fn(log_probs, true_p, mel_lens, true_len) / true_len.float().clamp(min=1.0)
            loss_neg = ctc_loss_fn(log_probs, neg_p, mel_lens, neg_len) / neg_len.float().clamp(min=1.0)

            for i in range(mels.size(0)):
                if not (torch.isnan(loss_true[i]) or torch.isnan(loss_neg[i])):
                    total_rank_trials += 1
                    if loss_true[i].item() < loss_neg[i].item():
                        ranker_wins += 1

                # 2. Evaluate Unpadded vs Padded Free Decoding PER
                valid_T = mel_lens[i].item()
                pred_unpadded = greedy_decode(log_probs[:valid_T, i, :])
                pred_padded = greedy_decode(log_probs[:, i, :])

                ref = true_p[i][:true_len[i]].cpu().tolist()
                ref_str = " ".join(map(str, ref))
                if ref_str.strip():
                    p_unpad = " ".join(map(str, pred_unpadded)) if pred_unpadded else "<empty>"
                    p_pad = " ".join(map(str, pred_padded)) if pred_padded else "<empty>"
                    pers_unpadded.append(float(wer(ref_str, p_unpad)))
                    pers_padded.append(float(wer(ref_str, p_pad)))

    avg_unpadded_per = (sum(pers_unpadded) / max(1, len(pers_unpadded))) * 100
    avg_padded_per = (sum(pers_padded) / max(1, len(pers_padded))) * 100
    rank_acc = (ranker_wins / max(1, total_rank_trials)) * 100

    print("\n" + "=" * 80)
    print(" HELD-OUT TEST SPLIT RESULTS (2,311 UTTERANCES | SPEAKERS F03, F04, M01, M05)")
    print("=" * 80)
    print(f"  1. Legacy Padded Free-Decode PER (with double-norm/pad bug): ~79.59%")
    print(f"  2. Single-Norm Padded Free-Decode PER:                       {avg_padded_per:.2f}%")
    print(f"  3. True Unpadded Free-Decode PER (Masked to mel_lens):       {avg_unpadded_per:.2f}%")
    print(f"  4. Contrastive Acoustic Ranker Accuracy (True < Halluc.):    {rank_acc:.2f}% ({ranker_wins}/{total_rank_trials})")
    print("=" * 80)

if __name__ == "__main__":
    main()
