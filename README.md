# Project TIDE (intelliSpeech)

**Offline Edge-Based Assistive Device for Dysarthric Speech Interpretation**

Project TIDE investigates whether combining a general automatic speech recognition (ASR) baseline with complementary acoustic-phonetic alignment and contextual candidate reranking can significantly improve interpretation accuracy—while operating entirely offline on an embedded edge processor.

---

## Architecture Overview

* **Dynamic Dual-Path Routing:** A confidence/uncertainty gate routes clear utterances through a **Fast Path** ($< 800\text{ ms}$) and ambiguous speech through a **Hard Path** ($< 1.3\text{ s}$).
* **Acoustic-Phonetic Head:** Lightweight 1D-CNN evaluating target phoneme likelihoods against speech evidence using G2P and CTC alignment.
* **Semantic Context Head:** Text-only compact language model (DistilBERT/MiniLM) evaluating hypothesis plausibility without accessing raw audio.
* **Multi-Signal Linear Fusion:** Z-score normalized score weighting ($\alpha \text{ASR} + \beta \text{Acoustic} + \gamma \text{Semantic}$) tuned on validation data via Nelder-Mead simplex optimization.

---

## Repository Structure

```text
intelliSpeech/
├── configs/               # YAML configuration files (audio parameters, split manifests)
├── data/
│   ├── raw/               # Raw audio datasets (e.g. TORGO) [git-ignored]
│   ├── processed/         # Standardized 16kHz audio [git-ignored]
│   └── manifests/         # Split manifests and audit reports
├── notebooks/             # Exploratory analysis & sanity checks
├── reports/               # Milestone memos & ablation tables
├── src/
│   ├── data/              # Ingestion, channel audit (headMic), LSO stratification
│   ├── frontend/          # In-memory Log-Mel feature extractor, VAD ring buffer
│   ├── models/            # Baseline Whisper, 1D-CNN acoustic head, semantic LM
│   ├── routing/           # Confidence gate and dual-path execution logic
│   ├── fusion/            # Z-score normalization and simplex weight optimization
│   └── hardware/          # I2S microphone ingestion, SPI display drivers
└── tests/                 # Unit and regression test suites
```
