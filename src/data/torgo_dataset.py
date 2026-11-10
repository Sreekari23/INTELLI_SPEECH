import os
import re
import json
import random
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T
import soundfile as sf

from src.models.acoustic_1dcnn import PhonemeTokenizer, LogMelSpectrogramTransform, NUM_PHONEMES
from src.frontend.mel_extractor import LogMelFeatureExtractor
from src.models.squeezeformer_ctc import char2idx


class DysarthricContrastiveDataset(Dataset):
    """
    Loads real TORGO manifest records and returns:
      - mel_features: (80, T_frames) normalized Log-Mel spectrogram
      - true_phn: (L_true,) ground-truth phoneme IDs
      - neg_phn: (L_neg,) hallucinated negative phoneme IDs
    """
    def __init__(self, manifest_path: str, max_seconds: float = 10.0):
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)
        self.base_dir = os.path.dirname(manifest_path)
        self.tokenizer = PhonemeTokenizer()
        self.mel_extractor = LogMelSpectrogramTransform()
        self.max_samples = int(16000 * max_seconds)

    def __len__(self) -> int:
        return len(self.records)

    def _hallucinate_negative(self, phn_ids: List[int]) -> List[int]:
        """Simulates dysarthric articulatory substitutions and consonant omissions."""
        max_valid_id = (NUM_PHONEMES - 2)
        if len(phn_ids) <= 2:
            return phn_ids + [5]
        neg = list(phn_ids)
        if random.random() < 0.7:
            neg.pop(random.randint(0, len(neg) - 1))
        if random.random() < 0.5 and len(neg) > 1:
            neg[random.randint(0, len(neg) - 1)] = random.randint(3, max_valid_id)
        return neg if neg else [2]

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        wav_path = os.path.join(self.base_dir, rec["file"])
        if not os.path.exists(wav_path):
            wav_path = rec["file"]

        audio_np, sr = sf.read(wav_path, dtype="float32")
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        if len(audio_np) > self.max_samples:
            audio_np = audio_np[:self.max_samples]

        waveform = torch.from_numpy(audio_np).float()
        mel = self.mel_extractor(waveform).squeeze(0)

        text = rec.get("text") or rec.get("transcript", "")
        true_phn = self.tokenizer.encode(text)
        if not true_phn:
            true_phn = [2]
        neg_phn = self._hallucinate_negative(true_phn)

        return (
            mel,
            torch.LongTensor(true_phn),
            torch.LongTensor(neg_phn),
        )


def collate_contrastive_batch(batch):
    """
    Pads (80, T) Mel spectrograms along Time dimension -> (Batch, 80, T_max)
    and returns exact unpadded frame lengths for CTC loss & decoding.
    """
    mels, trues, negs = zip(*batch)
    mel_lengths = torch.LongTensor([m.size(-1) for m in mels])
    true_lengths = torch.LongTensor([len(t) for t in trues])
    neg_lengths = torch.LongTensor([len(n) for n in negs])

    mels_t = [m.transpose(0, 1) for m in mels]
    mels_pad = torch.nn.utils.rnn.pad_sequence(mels_t, batch_first=True, padding_value=0.0).transpose(1, 2).contiguous()

    trues_pad = torch.nn.utils.rnn.pad_sequence(trues, batch_first=True, padding_value=0)
    negs_pad = torch.nn.utils.rnn.pad_sequence(negs, batch_first=True, padding_value=0)

    return mels_pad, mel_lengths, trues_pad, true_lengths, negs_pad, neg_lengths


# =====================================================================
# SQUEEZEFORMER CTC DATASET & COLLATOR (NON-AUTOREGRESSIVE)
# =====================================================================

class TorgoCTCDataset(Dataset):
    """
    Loads TORGO manifest records for Squeezeformer CTC training.
    Outputs:
      - mel: (T_frames, 80) CMVN-normalized Log-Mel spectrogram
      - target_ids: (L_chars,) 29-class character IDs
      - clean_text: str ground-truth transcript for WER/CER evaluation
    """
    def __init__(
        self,
        manifest_path: str,
        max_seconds: float = 12.0,
        augment: bool = False,
        config_path: Optional[str] = None,
    ):
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            raw_records = json.load(f)

        # Support both raw list of dicts or {"samples": [...]} / {"cases": [...]}
        if isinstance(raw_records, dict):
            self.records = raw_records.get("samples", raw_records.get("cases", []))
        else:
            self.records = raw_records

        self.base_dir = os.path.dirname(manifest_path)
        self.mel_extractor = LogMelFeatureExtractor(config_path=config_path)
        self.max_samples = int(16000 * max_seconds)
        self.augment = augment

        # SpecAugment transforms for training robustness on dysarthric speech
        self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
        self.time_mask = T.TimeMasking(time_mask_param=35)

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def encode_text(text: str) -> Tuple[List[int], str]:
        """Normalizes text to lowercase [a-z ' ] and maps to char2idx."""
        clean = text.lower().strip()
        clean = re.sub(r"[^a-z' ]+", "", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        ids = [char2idx[c] for c in clean if c in char2idx]
        if not ids:
            ids = [char2idx[" "]]
        return ids, clean

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.LongTensor, str]:
        rec = self.records[idx]
        rel_path = rec.get("file") or rec.get("audio_filepath") or rec.get("filename", "")
        wav_path = os.path.join(self.base_dir, rel_path)
        if not os.path.exists(wav_path):
            wav_path = rel_path

        audio_np, sr = sf.read(wav_path, dtype="float32")
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        if len(audio_np) > self.max_samples:
            audio_np = audio_np[:self.max_samples]

        waveform = torch.from_numpy(audio_np).float()
        if sr != 16000:
            waveform = T.Resample(orig_freq=sr, new_freq=16000)(waveform)

        # (1, 80, T_frames) -> (80, T_frames)
        mel = self.mel_extractor(waveform).squeeze(0)

        # Utterance-level CMVN (Cepstral Mean and Variance Normalization)
        mean = mel.mean(dim=-1, keepdim=True)
        std = mel.std(dim=-1, keepdim=True).clamp(min=1e-5)
        mel = (mel - mean) / std

        if self.augment:
            mel = self.freq_mask(mel)
            mel = self.time_mask(mel)

        raw_text = rec.get("text") or rec.get("transcript") or rec.get("reference", "")
        target_ids, clean_text = self.encode_text(raw_text)

        # Ensure T_frames is at least 4 * len(target_ids) + 4 so 4x subsampling never violates CTC length rules
        min_frames = len(target_ids) * 4 + 4
        if mel.size(-1) < min_frames:
            pad_amt = min_frames - mel.size(-1)
            mel = torch.nn.functional.pad(mel, (0, pad_amt), value=0.0)

        # Transpose to (T_frames, 80) as expected by SqueezeformerCTC.forward()
        mel_time_first = mel.transpose(0, 1).contiguous()

        return mel_time_first, torch.LongTensor(target_ids), clean_text


def collate_ctc_batch(batch):
    """
    Pads (T_frames, 80) features into (Batch, T_max, 80) and returns
    (features_pad, feature_lengths, targets_pad, target_lengths, transcripts).
    """
    mels, targets, texts = zip(*batch)
    feature_lengths = torch.LongTensor([m.size(0) for m in mels])
    target_lengths = torch.LongTensor([len(t) for t in targets])

    features_pad = torch.nn.utils.rnn.pad_sequence(mels, batch_first=True, padding_value=0.0)
    targets_pad = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)

    return features_pad, feature_lengths, targets_pad, target_lengths, list(texts)
