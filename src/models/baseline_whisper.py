"""Baseline Automatic Speech Recognition (ASR) Pipeline using Whisper.

Supports:
1. Greedy Decoding (K=1): Fast Top-1 linguistic baseline.
2. Beam Search Decoding (K=5): N-best candidate hypotheses with sequence log-probs.
3. Uncertainty & Confidence Scoring:
   - Length-normalized sequence log-probability: S_ASR(Y)
   - Normalized posterior over candidate beam: P_norm(Y_1)
   - Confidence margin delta: Delta = S(Y_1) - S(Y_2)
4. WER / CER Evaluation: Word Error Rate computation via jiwer.
"""

import time
from typing import Dict, List, Any, Optional, Union
import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import jiwer


class WhisperBaseline:
    """Whisper ASR baseline engine for greedy and beam search inference."""

    def __init__(
        self,
        model_name: str = "openai/whisper-tiny.en",
        device: Optional[str] = None,
        length_penalty: float = 0.7,
    ):
        """Initializes Whisper model and processor.

        Args:
            model_name: HuggingFace model checkpoint (default: tiny.en for fast CPU testing).
            device: 'cuda', 'cpu', or auto-detected if None.
            length_penalty: Exponent rho for length normalization (default 0.7).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.length_penalty = length_penalty

        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _prepare_inputs(self, audio: Union[np.ndarray, torch.Tensor], sample_rate: int = 16000) -> torch.Tensor:
        """Prepares audio into 80-channel log-Mel feature tensor for Whisper."""
        if isinstance(audio, torch.Tensor):
            if audio.dim() > 1:
                audio = audio.squeeze().detach().cpu().numpy()
            else:
                audio = audio.detach().cpu().numpy()

        input_features = self.processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).input_features
        return input_features.to(self.device)

    def transcribe_greedy(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """Runs fast greedy decoding (K=1)."""
        input_features = self._prepare_inputs(audio, sample_rate)

        start_time = time.perf_counter()
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                num_beams=1,
                do_sample=False,
                max_new_tokens=128,
            )
        latency_sec = time.perf_counter() - start_time

        transcript = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

        return {
            "transcript": transcript.upper(),
            "latency_sec": round(latency_sec, 4),
            "beam_width": 1,
        }

    def transcribe_beam(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        beam_width: int = 5,
        sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """Runs beam search decoding returning Top-K hypotheses and confidence scores."""
        input_features = self._prepare_inputs(audio, sample_rate)

        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                input_features,
                num_beams=beam_width,
                num_return_sequences=beam_width,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=128,
            )
        latency_sec = time.perf_counter() - start_time

        sequences = outputs.sequences
        # Raw sequence scores from HuggingFace beam search (sum of log probs / length_penalty)
        raw_scores = outputs.sequences_scores.cpu().numpy() if hasattr(outputs, "sequences_scores") else np.zeros(beam_width)

        decoded_texts = self.processor.batch_decode(sequences, skip_special_tokens=True)

        candidates: List[Dict[str, Any]] = []
        for rank, (text, score) in enumerate(zip(decoded_texts, raw_scores), start=1):
            candidates.append({
                "rank": rank,
                "transcript": text.strip().upper(),
                "sequence_log_prob": float(score),
            })

        # Compute confidence metrics over the candidate beam
        # Softmax-normalized posterior over beam:
        scores_arr = np.array([c["sequence_log_prob"] for c in candidates])
        exp_scores = np.exp(scores_arr - np.max(scores_arr))  # Numerical stability
        softmax_probs = exp_scores / np.sum(exp_scores)

        p_norm_top1 = float(softmax_probs[0]) if len(softmax_probs) > 0 else 1.0
        delta_margin = float(scores_arr[0] - scores_arr[1]) if len(scores_arr) > 1 else 0.0

        return {
            "top1_transcript": candidates[0]["transcript"] if candidates else "",
            "candidates": candidates,
            "confidence_metrics": {
                "p_norm_top1": round(p_norm_top1, 4),
                "delta_margin": round(delta_margin, 4),
            },
            "latency_sec": round(latency_sec, 4),
            "beam_width": beam_width,
        }

    @staticmethod
    def compute_evaluation_metrics(
        references: List[str],
        hypotheses: List[str]
    ) -> Dict[str, float]:
        """Computes Word Error Rate (WER) and Character Error Rate (CER)."""
        clean_refs = [r.strip().upper() for r in references]
        clean_hyps = [h.strip().upper() for h in hypotheses]

        wer = jiwer.wer(clean_refs, clean_hyps)
        cer = jiwer.cer(clean_refs, clean_hyps)

        return {
            "wer": round(float(wer), 4),
            "cer": round(float(cer), 4),
        }
