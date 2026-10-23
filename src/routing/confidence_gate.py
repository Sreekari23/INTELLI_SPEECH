"""Dynamic dual-path confidence gating and path routing.

Implements Project TIDE confidence gating:

  - Fast Path (direct release): confidence_score > threshold
  - Hard Path (deep reranking): confidence_score <= threshold

The gate decision is based on the top-1 posterior probability P_norm(Y_1)
and the top-2 margin delta Delta = S(Y_1) - S(Y_2).
"""

from typing import Any, Dict


def decide_routing(
    confidence_score: float,
    confidence_margin: float,
    threshold: float,
) -> Dict[str, Any]:
    """Decide FAST vs HARD routing from ASR confidence metrics.

    Args:
        confidence_score:   Top-1 normalized posterior probability P_norm(Y_1) in [0, 1].
        confidence_margin:  Posterior margin delta Delta = S(Y_1) - S(Y_2).
        threshold:          Confidence cutoff above which we take the Fast Path.

    Returns:
        Dict with keys:
            path:            "FAST" or "HARD"
            gate_decision:   Human-readable decision string
            confidence_score: echo
            confidence_margin: echo
            threshold:        echo
    """
    if confidence_score > threshold:
        path = "FAST"
        gate_decision = "Direct Release (High Confidence)"
    else:
        path = "HARD"
        gate_decision = "Deep Fusion Rerank (Uncertainty Detected)"

    return {
        "path": path,
        "gate_decision": gate_decision,
        "confidence_score": float(confidence_score),
        "confidence_margin": float(confidence_margin),
        "threshold": float(threshold),
    }


def gate_name(path: str) -> str:
    """Return a human-readable label for a path token."""
    return {
        "FAST": "FAST_PATH (DIRECT RELEASE)",
        "HARD": "HARD_PATH (DEEP RERANKING)",
    }.get(path, path)
