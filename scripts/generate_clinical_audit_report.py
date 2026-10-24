import sys, os, json, hashlib, random
sys.path.insert(0, os.path.abspath("."))
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from jiwer import wer

from src.models.acoustic_1dcnn import Acoustic1DCNN, NUM_PHONEMES
from src.data.torgo_dataset import DysarthricContrastiveDataset
from src.data.audit_torgo import TORGO_SPEAKER_METADATA

def greedy_decode(log_probs_2d, blank_id=0):
    preds = torch.argmax(log_probs_2d, dim=-1).cpu().numpy()
    decoded, prev = [], None
    for p in preds:
        if p != blank_id and p != prev:
            decoded.append(int(p))
        prev = p
    return decoded

def sha256_file(path):
    if not os.path.exists(path):
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def evaluate_model(model, test_ds, device):
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True, reduction="none")
    spk_stats = defaultdict(lambda: {"pers": [], "wins": 0, "total": 0})
    random.seed(42)

    model.eval()
    with torch.no_grad():
        for idx in range(len(test_ds)):
            rec = test_ds.records[idx]
            spk = rec.get("speaker_id", "UNKNOWN")
            mel, true_p, neg_p = test_ds[idx]

            mel_b = mel.unsqueeze(0).to(device)
            T_len = torch.LongTensor([mel.size(-1)]).to(device)
            t_p = true_p.unsqueeze(0).to(device)
            n_p = neg_p.unsqueeze(0).to(device)
            t_len = torch.LongTensor([len(true_p)]).to(device)
            n_len = torch.LongTensor([len(neg_p)]).to(device)

            logits = model(mel_b)
            log_probs = F.log_softmax(logits.transpose(0, 1).contiguous(), dim=-1)

            loss_t = (ctc_loss_fn(log_probs, t_p, T_len, t_len) / t_len.float().clamp(min=1.0)).item()
            loss_n = (ctc_loss_fn(log_probs, n_p, T_len, n_len) / n_len.float().clamp(min=1.0)).item()

            spk_stats[spk]["total"] += 1
            if loss_t < loss_n:
                spk_stats[spk]["wins"] += 1

            pred = greedy_decode(log_probs[:, 0, :])
            ref_str = " ".join(map(str, true_p.tolist()))
            pred_str = " ".join(map(str, pred)) if pred else "<empty>"
            if ref_str.strip():
                spk_stats[spk]["pers"].append(float(wer(ref_str, pred_str)))

    return spk_stats

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = DysarthricContrastiveDataset("data/processed/test/manifest.json")

    # 1. Baseline: Untrained Blank-Dominant / Phase-2 Trial Baseline
    # Find Phase-2 trial checkpoint if available in Downloads or repo backup, else use blank-prior init
    p2_path = os.path.expanduser("~/Downloads/best_1dcnn.pth")
    if not os.path.exists(p2_path):
        p2_path = "checkpoints/best_1dcnn.pth"
    
    p2_model = Acoustic1DCNN(in_features=80, num_classes=NUM_PHONEMES, hidden_dim=256).to(device)
    p2_hash = sha256_file(p2_path)
    p2_ckpt = torch.load(p2_path, map_location=device)
    p2_model.load_state_dict(p2_ckpt["model_state_dict"] if "model_state_dict" in p2_ckpt else p2_ckpt)
    p2_stats = evaluate_model(p2_model, test_ds, device)

    # 2. Target Checkpoint: checkpoints/best_1dcnn_v2.pth
    v2_path = "checkpoints/best_1dcnn_v2.pth"
    v2_hash = sha256_file(v2_path)
    v2_ckpt = torch.load(v2_path, map_location=device)
    v2_model = Acoustic1DCNN(in_features=80, num_classes=NUM_PHONEMES, hidden_dim=256).to(device)
    v2_model.load_state_dict(v2_ckpt["model_state_dict"] if "model_state_dict" in v2_ckpt else v2_ckpt)
    v2_stats = evaluate_model(v2_model, test_ds, device)

    print("=" * 102)
    print(" PHASE 4: SPEAKER-STRATIFIED AUDIT REPORT (SINGLE-SOURCE METADATA: src/data/audit_torgo.py)")
    print("=" * 102)
    print(f" Baseline Checkpoint (Phase-2) : {p2_path} (SHA-256: {p2_hash[:16]}...)")
    print(f" Target Checkpoint   (Phase-3) : {v2_path} (SHA-256: {v2_hash[:16]}...)")
    print("-" * 102)
    print(f" {'Speaker':<8} | {'Cohort':<10} | {'TORGO Severity':<15} | {'N':<5} | {'Phase2 PER':<10} | {'v2 PER':<9} | {'Delta PER':<10} | {'v2 Win %'}")
    print("-" * 102)

    report = {
        "metadata_source": "src.data.audit_torgo.TORGO_SPEAKER_METADATA",
        "baseline_checkpoint": {"path": p2_path, "sha256": p2_hash},
        "target_checkpoint": {"path": v2_path, "sha256": v2_hash},
        "metric_definitions": {
            "free_decode_per": "Unconstrained greedy CTC decoding edit distance: (S+D+I)/N_ref * 100.",
            "pairwise_contrastive_win_rate": "Per-utterance % where length-normalized CTC loss L_CTC(x, y_true)/|y_true| < L_CTC(x, y_neg)/|y_neg|."
        },
        "speakers": {},
        "cohorts": {}
    }

    cohorts = {
        "Control": {"b_per": [], "v_per": [], "wins": 0, "tot": 0},
        "Dysarthric": {"b_per": [], "v_per": [], "wins": 0, "tot": 0},
        "Overall": {"b_per": [], "v_per": [], "wins": 0, "tot": 0},
    }

    for spk in sorted(v2_stats.keys()):
        meta = TORGO_SPEAKER_METADATA.get(spk, {})
        severity = meta.get("severity", "Unknown")
        cohort = "Control" if severity == "Control" or "C" in spk else "Dysarthric"

        st_v = v2_stats[spk]
        st_b = p2_stats[spk]

        b_per = (sum(st_b["pers"]) / max(1, len(st_b["pers"]))) * 100
        v_per = (sum(st_v["pers"]) / max(1, len(st_v["pers"]))) * 100
        delta = v_per - b_per
        win_pct = (st_v["wins"] / max(1, st_v["total"])) * 100

        print(f" {spk:<8} | {cohort:<10} | {severity:<15} | {st_v['total']:<5} | {b_per:>9.2f}% | {v_per:>7.2f}% | {delta:>+8.2f}% | {win_pct:>7.2f}%")

        report["speakers"][spk] = {
            "cohort": cohort,
            "severity": severity,
            "utterances": st_v["total"],
            "phase2_baseline_per": round(b_per, 2),
            "v2_free_decode_per": round(v_per, 2),
            "delta_per": round(delta, 2),
            "pairwise_contrastive_win_rate": round(win_pct, 2)
        }

        for key in ["Overall", cohort]:
            cohorts[key]["b_per"].extend(st_b["pers"])
            cohorts[key]["v_per"].extend(st_v["pers"])
            cohorts[key]["wins"] += st_v["wins"]
            cohorts[key]["tot"] += st_v["total"]

    print("-" * 102)
    for c_name in ["Control", "Dysarthric", "Overall"]:
        c = cohorts[c_name]
        if c["tot"] > 0:
            bp = (sum(c["b_per"]) / len(c["b_per"])) * 100
            vp = (sum(c["v_per"]) / len(c["v_per"])) * 100
            dp = vp - bp
            wp = (c["wins"] / c["tot"]) * 100
            print(f" {c_name.upper():<8} | {'Summary':<10} | {'Aggregated':<15} | {c['tot']:<5} | {bp:>9.2f}% | {vp:>7.2f}% | {dp:>+8.2f}% | {wp:>7.2f}%")
            report["cohorts"][c_name.lower()] = {
                "utterances": c["tot"],
                "phase2_baseline_per": round(bp, 2),
                "v2_free_decode_per": round(vp, 2),
                "delta_per": round(dp, 2),
                "pairwise_contrastive_win_rate": round(wp, 2)
            }
    print("=" * 102)

    os.makedirs("results", exist_ok=True)
    with open("results/phase4_test_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("[OK] Updated canonical audit report in results/phase4_test_evaluation.json")

if __name__ == "__main__":
    main()
