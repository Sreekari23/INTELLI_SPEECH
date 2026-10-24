# IntelliSpeech: TIDE (Dysarthric Speech Recognition Enhancement Pipeline)

An end-to-end, speaker-disjoint speech recognition and reranking pipeline tailored for severe dysarthric speech (TORGO dataset). TIDE combines a custom **1D-CNN Acoustic Contrastive Ranker**, **DistilBERT Semantic Pseudo-Log-Likelihood (PLL)** scoring, and **Confidence-Gated Beam Fusion** to rescue distorted ASR hypotheses from unconstrained failures.

---

## 🚀 Key Achievements & Architecture

1. **Speaker-Disjoint Clinical Splits (`train`, `val`, `test`):** 
   - Engineered strict clinical data splits ensuring zero speaker overlap between training and evaluation. Head-microphone TORGO audio filtered and structured into rigorous speaker-stratified sets.
2. **Phase 3: Contrastive 1D-CNN Acoustic Ranker (`best_1dcnn_v2.pth`):**
   - Trained a lightweight (2.65 MB) `Acoustic1DCNN` on Kaggle's T4 GPU using a hybrid **CTC Loss + Contrastive Margin Penalty** against synthetically hallucinated dysarthric phoneme errors.
   - Achieved an overall **74.64% Pairwise Contrastive Ranker Accuracy** on completely unseen test speakers.
3. **Phase 4 & 5: Rigorous Auditing & 5-Stage Scientific Ablation:**
   - Evaluated unpadded phoneme decoding and contrastive discrimination across severity tiers (Control, Mild, Moderate, Severe).
   - Proven end-to-end Word Error Rate (WER) reduction from **90.9% (Base ASR)** down to **74.8% (Full TIDE)**.

---

## 📊 Scientific Ablation Benchmark Results

Evaluated across real Whisper beams, semantic language modeling, and acoustic ranker scoring:

| Severity Tier | Stage A: Base ASR | Stage B: +Acoustic | Stage C: +Semantic | Stage D: Full TIDE | Oracle Ceiling |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Control** | 5.7% | 7.7% | **3.6%** | 7.7% | 3.6% |
| **Mild** | 78.8% | **60.0%** | 69.4% | **60.0%** | 44.4% |
| **Moderate** | 123.7% | 111.2% | 136.2% | **106.6%** | 87.5% |
| **Severe** | 155.5% | 155.5% | 150.3% | **124.7%** | 92.6% |
| **OVERALL MEAN** | **90.9%** | **83.6%** | **89.8%** | **74.8%** | **57.0%** |

---

## 🛠️ Repository Structure

```text
intelliSpeech/
├── checkpoints/              # Verified neural weights (best_1dcnn_v2.pth)
├── data/
│   └── processed/            # Speaker-stratified train/val/test manifests
├── reports/                  # JSON ablation reports (real_5stage_ablation.json)
├── results/                  # Canonical audit ledgers & test evaluations
├── src/
│   ├── data/                 # TORGO dataloaders & clinical split builders
│   ├── fusion/               # Confidence gating & beam fusion logic
│   └── models/               # Acoustic1DCNN, PhonemeTokenizer, LogMelTransform
└── scripts/
    ├── train_contrastive_1dcnn.py  # GPU training script
    ├── evaluate_phase4.py          # Held-out test evaluation
    ├── generate_clinical_audit_report.py # Speaker-stratified auditor
    └── run_5stage_ablation.py      # End-to-end TIDE benchmark

⚙️ Quickstart & Reproduction

    Install Dependencies:
    PowerShell

    pip install torch torchaudio soundfile jiwer transformers g2p_en distance

    Run Held-Out Test Evaluation:
    PowerShell

    python scripts\evaluate_phase4.py

    Run Clinical Audit Report:
    PowerShell

    python scripts\generate_clinical_audit_report.py

    Run Full 5-Stage Ablation Benchmark:
    PowerShell

    python scripts\run_5stage_ablation.py

