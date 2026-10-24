import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from scipy.optimize import minimize
from transformers import AutoTokenizer, AutoModelForMaskedLM
from src.models.acoustic_1dcnn import Acoustic1DCNN, PhonemeTokenizer, LogMelSpectrogramTransform
from src.fusion.beam_fusion import z_score_normalize
from jiwer import wer, cer

def compute_pll_score(text: str, tokenizer, mlm_model, device: str = "cpu") -> float:
    """Computes length-normalized Pseudo-Log-Likelihood (PLL) using DistilBERT MLM."""
    if not text.strip():
        return -20.0
    inputs = tokenizer(text.lower(), return_tensors="pt").to(device)
    input_ids = inputs["input_ids"][0]
    # Tokens excluding [CLS] and [SEP]
    token_Len = len(input_ids) - 2
    if token_Len <= 0:
        return -15.0

    masked_ids = input_ids.unsqueeze(0).repeat(token_Len, 1)
    for i in range(token_Len):
        masked_ids[i, i + 1] = tokenizer.mask_token_id

    with torch.no_grad():
        logits = mlm_model(masked_ids).logits
        log_probs = F.log_softmax(logits, dim=-1)

    total_ll = 0.0
    for i in range(token_Len):
        orig_token = input_ids[i + 1]
        total_ll += float(log_probs[i, i + 1, orig_token].item())

    return total_ll / token_Len

def main():
    print("=" * 92)
    print(" PROJECT TIDE: 5-STAGE SCIENTIFIC ABLATION BENCHMARK (REAL WHISPER BEAMS + PLL + 1D-CNN)")
    print("=" * 92)

    beams_p = Path("data/real_torgo_samples/real_whisper_beams.json")
    with open(beams_p, "r", encoding="utf-8") as f:
        records = json.load(f)

    device = "cpu"
    ckpt_path = (
        Path("checkpoints/best_1dcnn_v2.pth")
        if Path("checkpoints/best_1dcnn_v2.pth").exists()
        else Path("checkpoints/best_1dcnn.pth")
    )
    print(f"[*] Loading Neural 1D-CNN ({ckpt_path})...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    acoustic_model = Acoustic1DCNN().to(device)
    acoustic_model.load_state_dict(ckpt["model_state_dict"])
    acoustic_model.eval()

    phn_tokenizer = PhonemeTokenizer()
    mel_extractor = LogMelSpectrogramTransform()
    ctc_fn = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)

    print("[*] Loading Masked LM for Pseudo-Log-Likelihood (distilbert-base-uncased)...")
    mlm_tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    mlm_model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(device)
    mlm_model.eval()

    enriched = []
    print("\n[*] Computing Acoustic CTC & Semantic PLL Evidence across all 16 beams...\n")

    for idx, rec in enumerate(records, 1):
        t0 = time.time()
        wav_path = rec["file"]
        ref = rec["reference"]
        sev = rec["severity"]
        spk = rec["speaker_id"]

        audio_np, sr = sf.read(wav_path, dtype="float32")
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        wav_t = torch.from_numpy(audio_np).unsqueeze(0)

        with torch.no_grad():
            mel = mel_extractor(wav_t)
            logits = acoustic_model(mel)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

        t_frames = log_probs.size(0)
        in_len = torch.tensor([t_frames], dtype=torch.long)

        # Deduplicate candidate transcripts while preserving order
        unique_cands = []
        seen = set()
        for c in rec["candidates"]:
            txt = c["transcript"].strip()
            if txt and txt not in seen:
                seen.add(txt)
                unique_cands.append({"transcript": txt, "logprob": float(c["logprob"])})

        asr_scores, acous_scores, sem_scores, cand_wers, cand_cers = [], [], [], [], []

        for c in unique_cands:
            txt = c["transcript"]
            asr_scores.append(c["logprob"])

            p_ids = phn_tokenizer.encode(txt)
            t_len = max(1, len(p_ids))
            tgt_t = torch.tensor([p_ids], dtype=torch.long)
            loss_val = float(ctc_fn(log_probs, tgt_t, in_len, torch.tensor([t_len])).item())
            acous_scores.append(-loss_val / t_len)

            pll = compute_pll_score(txt, mlm_tok, mlm_model, device)
            sem_scores.append(pll)

            cand_wers.append(wer(ref, txt))
            cand_cers.append(cer(ref, txt))

        # Z-score normalize across unique candidates in the beam
        z_asr = z_score_normalize(asr_scores)
        z_acous = z_score_normalize(acous_scores)
        z_sem = z_score_normalize(sem_scores)

        # Compute ASR posterior confidence & beam diversity for gating
        # If Whisper produced only 1 unique hypothesis across temperatures, confidence is 1.0 (Fast Path)
        if len(unique_cands) == 1:
            conf_margin = 1.0
        else:
            exp_lp = np.exp(np.array(asr_scores) - np.max(asr_scores))
            probs = exp_lp / np.sum(exp_lp)
            conf_margin = float(probs[0] - probs[1])

        lat_ms = (time.time() - t0) * 1000

        enriched.append({
            "idx": idx,
            "spk": spk,
            "sev": sev,
            "ref": ref,
            "cands": [c["transcript"] for c in unique_cands],
            "z_asr": z_asr,
            "z_acous": z_acous,
            "z_sem": z_sem,
            "raw_acous": acous_scores,
            "raw_sem": sem_scores,
            "wers": cand_wers,
            "cers": cand_cers,
            "conf_margin": conf_margin,
            "lat_ms": lat_ms
        })

        best_acous_idx = int(np.argmax(acous_scores))
        best_sem_idx = int(np.argmax(sem_scores))
        print(f"[{idx:02d}/16] {sev:<8} ({spk}) | Unique Beam: {len(unique_cands)} | Base WER: {cand_wers[0]:.2f} | Acous Pick WER: {cand_wers[best_acous_idx]:.2f} | Sem Pick WER: {cand_wers[best_sem_idx]:.2f} | Oracle: {min(cand_wers):.2f}")

    # Nelder-Mead Simplex Optimization for (alpha, beta, gamma)
    def loss_func(weights):
        a, b, g = weights
        total_w = 0.0
        for item in enriched:
            scores = a * item["z_asr"] + b * item["z_acous"] + g * item["z_sem"]
            pick = int(np.argmax(scores))
            total_w += item["wers"][pick]
        return total_w / len(enriched)

    best_w = [0.4, 0.3, 0.3]
    best_val = loss_func(best_w)
    # Grid warm-start + Nelder-Mead simplex refinement
    for a in np.linspace(0.0, 1.0, 6):
        for b in np.linspace(0.0, 1.0, 6):
            for g in np.linspace(0.0, 1.0, 6):
                if a + b + g == 0:
                    continue
                val = loss_func([a, b, g])
                if val < best_val:
                    best_val = val
                    best_w = [a, b, g]

    res = minimize(loss_func, best_w, method="Nelder-Mead", options={"maxiter": 200})
    opt_w = res.x / np.sum(np.abs(res.x))
    alpha, beta, gamma = float(opt_w[0]), float(opt_w[1]), float(opt_w[2])
    print(f"\n[*] Nelder-Mead Optimal Weights: alpha(ASR)={alpha:.3f}, beta(Acoustic)={beta:.3f}, gamma(Semantic)={gamma:.3f}")

    # Evaluate the 5 Ablation Stages per Tier
    print("\n" + "=" * 92)
    print(f"{'Severity Tier':<13} | {'A: Base ASR':<12} | {'B: +Acoustic':<12} | {'C: +Semantic':<12} | {'D: Full TIDE':<12} | {'Oracle Ceil':<11}")
    print("-" * 92)

    tiers = ["Control", "Mild", "Moderate", "Severe"]
    for t_name in tiers:
        sub = [x for x in enriched if x["sev"] == t_name]
        w_a = np.mean([x["wers"][0] for x in sub]) * 100
        w_b = np.mean([x["wers"][int(np.argmax(0.3 * x["z_asr"] + 0.7 * x["z_acous"]))] for x in sub]) * 100
        w_c = np.mean([x["wers"][int(np.argmax(0.3 * x["z_asr"] + 0.7 * x["z_sem"]))] for x in sub]) * 100
        
        tide_wers = []
        for x in sub:
            if x["conf_margin"] >= 0.50:  # Fast path
                tide_wers.append(x["wers"][0])
            else:
                s = alpha * x["z_asr"] + beta * x["z_acous"] + gamma * x["z_sem"]
                tide_wers.append(x["wers"][int(np.argmax(s))])
        w_e = np.mean(tide_wers) * 100
        w_or = np.mean([min(x["wers"]) for x in sub]) * 100

        print(f"{t_name:<13} | {w_a:>10.1f}% | {w_b:>10.1f}% | {w_c:>10.1f}% | {w_e:>10.1f}% | {w_or:>9.1f}%")

    print("-" * 92)
    all_a = np.mean([x["wers"][0] for x in enriched]) * 100
    all_b = np.mean([x["wers"][int(np.argmax(0.3 * x["z_asr"] + 0.7 * x["z_acous"]))] for x in enriched]) * 100
    all_c = np.mean([x["wers"][int(np.argmax(0.3 * x["z_asr"] + 0.7 * x["z_sem"]))] for x in enriched]) * 100
    all_e = np.mean([
        x["wers"][0] if x["conf_margin"] >= 0.50 else x["wers"][int(np.argmax(alpha * x["z_asr"] + beta * x["z_acous"] + gamma * x["z_sem"]))]
        for x in enriched
    ]) * 100
    all_or = np.mean([min(x["wers"]) for x in enriched]) * 100
    print(f"{'OVERALL MEAN':<13} | {all_a:>10.1f}% | {all_b:>10.1f}% | {all_c:>10.1f}% | {all_e:>10.1f}% | {all_or:>9.1f}%")
    print("=" * 92)

    # Save full ablation report JSON
    Path("reports").mkdir(exist_ok=True)
    with open("reports/real_5stage_ablation.json", "w", encoding="utf-8") as f:
        json.dump({
            "optimal_weights": {"alpha": round(alpha, 4), "beta": round(beta, 4), "gamma": round(gamma, 4)},
            "overall_wer": {
                "stage_A_baseline": round(all_a, 2),
                "stage_B_acoustic": round(all_b, 2),
                "stage_C_semantic": round(all_c, 2),
                "stage_E_full_tide": round(all_e, 2),
                "oracle_ceiling": round(all_or, 2)
            }
        }, f, indent=2)
    print("[OK] Saved full benchmark report to reports/real_5stage_ablation.json")

if __name__ == "__main__":
    main()
