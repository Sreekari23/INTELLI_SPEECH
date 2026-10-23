"""Multi-signal score normalization and candidate beam fusion.

Implements Project TIDE fusion as specified in the research design:

  - Each candidate signal S_ASR, S_Acoustic, S_Semantic is Z-score normalized
    across the candidate beam {Y_1 ... Y_K} before weighting.
  - Final_Score(Y_k) = w_asr * Z(S_ASR)(Y_k) + w_acoustic * Z(S_Acoustic)(Y_k)
                       + w_semantic * Z(S_Semantic)(Y_k)
  - Zero-variance beams (all scores identical) fall back to raw scores to avoid
    NaNs from division by zero.
"""

from typing import Any, Dict, List

import numpy as np


def z_score_normalize(scores: List[float], eps: float = 1e-6) -> np.ndarray:
    """Z-score normalize candidate scores across the beam (mu=0, sigma=1).

    Falls back to raw scores when beam variance is near zero.
    """
    arr = np.array(scores, dtype=np.float64)
    std = float(np.std(arr))
    if std < eps:
        return arr.copy()
    return (arr - float(np.mean(arr))) / (std + eps)


def compute_fusion_scores(
    transcripts: List[str],
    asr_norm_scores: List[float],
    acoustic_scores: List[float],
    semantic_scores: List[float],
    weights: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Beam-level Z-score normalization then weighted linear fusion.

    Args:
        transcripts:       Candidate transcript strings (aligned with scores).
        asr_norm_scores:   Softmax-normalized ASR posteriors per candidate [0,1].
        acoustic_scores:   Best-anchored acoustic scores [0,1] per candidate.
        semantic_scores:   Fluency pseudo-likelihood [0,1] per candidate.
        weights:           {"asr": w_a, "acoustic": w_c, "semantic": w_s}.

    Returns:
        List of dicts sorted by fusion_score descending, each containing:
            transcript, raw_index, asr_norm, acoustic_norm, semantic_norm,
            asr_z, acoustic_z, semantic_z, fusion_score, rank.
    """
    n = max(len(transcripts), len(asr_norm_scores), len(acoustic_scores), len(semantic_scores))
    if n == 0:
        return []

    w_asr = float(weights.get("asr", 0.4))
    w_acoustic = float(weights.get("acoustic", 0.3))
    w_semantic = float(weights.get("semantic", 0.3))

    # Pad missing entries with 0.0 to keep beam-aligned lists consistent.
    def pad(lst: List[float], target: int) -> List[float]:
        lst = list(lst)
        return lst + [0.0] * (target - len(lst))

    transcripts = list(transcripts) + [f"<missing_{i}>" for i in range(len(transcripts), n)]
    asr_pad = pad(asr_norm_scores, n)
    acous_pad = pad(acoustic_scores, n)
    sem_pad = pad(semantic_scores, n)

    asr_z = z_score_normalize(asr_pad).tolist()
    acous_z = z_score_normalize(acous_pad).tolist()
    sem_z = z_score_normalize(sem_pad).tolist()

    fused: List[Dict[str, Any]] = []
    for i in range(n):
        fs = (w_asr * asr_z[i]) + (w_acoustic * acous_z[i]) + (w_semantic * sem_z[i])
        fused.append({
            "transcript": transcripts[i],
            "raw_index": i,
            "asr_norm": round(float(asr_pad[i]), 4),
            "acoustic_norm": round(float(acous_pad[i]), 4),
            "semantic_norm": round(float(sem_pad[i]), 4),
            "asr_z": round(float(asr_z[i]), 4),
            "acoustic_z": round(float(acous_z[i]), 4),
            "semantic_z": round(float(sem_z[i]), 4),
            "fusion_score": round(float(fs), 6),
        })

    fused.sort(key=lambda x: x["fusion_score"], reverse=True)
    for rank, item in enumerate(fused, start=1):
        item["rank"] = rank
    return fused
