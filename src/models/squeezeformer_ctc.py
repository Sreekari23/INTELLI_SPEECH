"""Squeezeformer / Conformer CTC Acoustic Model for Dysarthric Edge Inference.

Features:
1. 1D-CNN Subsampling (4x sequence reduction for edge-latency).
2. Torchaudio Conformer Backbone (Parallelized acoustic modeling).
3. Linear CTC Classification Head (Non-autoregressive inference).
4. Fast Prefix CTC Beam Search (Extracts Top-K candidates & confidence margins in <3ms).
"""

import math
from collections import defaultdict
import torch
import torch.nn as nn
import numpy as np
import torchaudio.models as models
from typing import Tuple, List, Dict, Any

# Standard 29-character CTC Vocabulary
# Index 0 is reserved for the CTC <blank> token
VOCAB = ["<blank>", " ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", 
         "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]

char2idx = {char: idx for idx, char in enumerate(VOCAB)}
idx2char = {idx: char for idx, char in enumerate(VOCAB)}


def _log_add(a: float, b: float) -> float:
    if a == -float("inf"):
        return b
    if b == -float("inf"):
        return a
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))


class SqueezeformerCTC(nn.Module):
    def __init__(
        self, 
        input_dim: int = 80,
        num_classes: int = 29,
        encoder_dim: int = 256, 
        num_layers: int = 8,
        num_heads: int = 4,
        depthwise_conv_kernel: int = 31,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 1. Subsampling: Reduces sequence length by 4x to accelerate Conformer math
        self.subsample = nn.Sequential(
            nn.Conv1d(input_dim, encoder_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(encoder_dim, encoder_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU()
        )
        
        # 2. Conformer Backbone: Captures local convolutions and global attention
        self.conformer = models.Conformer(
            input_dim=encoder_dim,
            num_heads=num_heads,
            ffn_dim=encoder_dim * 4,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel,
            dropout=dropout,
        )
        
        # 3. CTC Classification Head: Maps features directly to character probabilities
        self.ctc_head = nn.Linear(encoder_dim, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        x = self.subsample(x)
        x = x.transpose(1, 2)
        
        out_lengths = lengths.clone()
        out_lengths = ((out_lengths - 1) // 2 + 1)
        out_lengths = ((out_lengths - 1) // 2 + 1)
        
        x, _ = self.conformer(x, out_lengths)
        logits = self.ctc_head(x)
        
        return logits, out_lengths

    @torch.no_grad()
    def decode_greedy(self, logits: torch.Tensor, lengths: torch.Tensor) -> List[str]:
        """Runs fast greedy CTC decoding on the output logits."""
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        lengths_np = lengths.cpu().numpy()
        
        decoded_transcripts = []
        for batch_idx in range(len(predictions)):
            pred = predictions[batch_idx][:lengths_np[batch_idx]]
            collapsed = []
            prev_idx = -1
            for idx in pred:
                if idx != prev_idx and idx != 0:
                    collapsed.append(idx2char[idx])
                prev_idx = idx
            decoded_transcripts.append("".join(collapsed).strip())
            
        return decoded_transcripts

    @torch.no_grad()
    def decode_beam(
        self,
        logits: torch.Tensor,
        lengths: torch.Tensor,
        beam_width: int = 5,
        top_k_tokens: int = 8
    ) -> Dict[str, Any]:
        """Runs lightweight CTC Prefix Beam Search on single-utterance logits (1, T, C).
        
        Returns Top-K candidate transcripts, sequence log-probs, and confidence gate metrics
        compatible with Project TIDE's Dual-Path router.
        """
        log_probs = torch.nn.functional.log_softmax(logits[0, :lengths[0]], dim=-1).cpu().numpy()
        T_steps, V = log_probs.shape

        # beams maps prefix_tuple -> (prob_blank, prob_non_blank) in log space
        beams = {(): (0.0, -float("inf"))}

        for t in range(T_steps):
            next_beams = defaultdict(lambda: (-float("inf"), -float("inf")))
            # Prune token alphabet per frame to top_k_tokens for <3ms speed
            top_indices = np.argsort(log_probs[t])[-top_k_tokens:]

            for prefix, (p_b, p_nb) in beams.items():
                p_total = _log_add(p_b, p_nb)

                # Case 1: Emit <blank> (index 0)
                p_blank_emit = p_total + float(log_probs[t, 0])
                cur_b, cur_nb = next_beams[prefix]
                next_beams[prefix] = (_log_add(cur_b, p_blank_emit), cur_nb)

                # Case 2: Emit characters
                for c in top_indices:
                    if c == 0:
                        continue
                    p_char = float(log_probs[t, c])
                    new_prefix = prefix + (int(c),)

                    if len(prefix) > 0 and prefix[-1] == c:
                        # Repeat character: if collapsing into current prefix, only from p_nb
                        cur_b_same, cur_nb_same = next_beams[prefix]
                        next_beams[prefix] = (cur_b_same, _log_add(cur_nb_same, p_nb + p_char))
                        # Extending with duplicate requires prior blank p_b
                        cur_b_new, cur_nb_new = next_beams[new_prefix]
                        next_beams[new_prefix] = (cur_b_new, _log_add(cur_nb_new, p_b + p_char))
                    else:
                        cur_b_new, cur_nb_new = next_beams[new_prefix]
                        next_beams[new_prefix] = (cur_b_new, _log_add(cur_nb_new, p_total + p_char))

            # Keep top beam_width prefixes
            sorted_beams = sorted(
                next_beams.items(),
                key=lambda kv: _log_add(kv[1][0], kv[1][1]),
                reverse=True
            )
            beams = dict(sorted_beams[:beam_width])

        candidates = []
        for rank, (prefix, (p_b, p_nb)) in enumerate(beams.items(), start=1):
            score = _log_add(p_b, p_nb)
            text = "".join(idx2char[idx] for idx in prefix).strip().upper()
            candidates.append({
                "rank": rank,
                "transcript": text if text else " ",
                "sequence_log_prob": round(float(score), 4),
            })

        scores_arr = np.array([c["sequence_log_prob"] for c in candidates])
        exp_scores = np.exp(scores_arr - np.max(scores_arr))
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
            "beam_width": beam_width,
        }
