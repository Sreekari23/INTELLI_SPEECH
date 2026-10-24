import os
import json
import random
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
import soundfile as sf

from src.models.acoustic_1dcnn import PhonemeTokenizer, LogMelSpectrogramTransform, NUM_PHONEMES

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
        # Extract normalized (1, 80, T) -> squeeze to (80, T)
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

    # Transpose to (T, 80) for pad_sequence, then back to (Batch, 80, T_max)
    mels_t = [m.transpose(0, 1) for m in mels]
    mels_pad = torch.nn.utils.rnn.pad_sequence(mels_t, batch_first=True, padding_value=0.0).transpose(1, 2).contiguous()

    trues_pad = torch.nn.utils.rnn.pad_sequence(trues, batch_first=True, padding_value=0)
    negs_pad = torch.nn.utils.rnn.pad_sequence(negs, batch_first=True, padding_value=0)

    return mels_pad, mel_lengths, trues_pad, true_lengths, negs_pad, neg_lengths
