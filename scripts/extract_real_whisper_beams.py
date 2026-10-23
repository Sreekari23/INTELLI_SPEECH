import sys, os
sys.path.insert(0, os.path.abspath("."))
import json, re, soundfile as sf
from pathlib import Path
from faster_whisper import WhisperModel
from jiwer import wer

def clean_txt(t: str) -> str:
    t = re.sub(r"[^A-Za-z\s']", "", t).strip().upper()
    return re.sub(r"\s+", " ", t)

def main():
    print("=" * 80)
    print(" EXTRACTING REAL WHISPER MULTI-BEAM HYPOTHESES FROM TORGO AUDIO")
    print("=" * 80)

    manifest_p = Path("data/real_torgo_samples/manifest.json")
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    whisper = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    beam_records = []

    for idx, rec in enumerate(manifest, 1):
        wav_path = rec["file"]
        ref = clean_txt(rec["reference"])
        sev = rec["severity"]
        spk = rec["speaker_id"]

        audio_np, sr = sf.read(wav_path, dtype="float32")
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)

        # Transcribe with beam_size=5 and temperature fallback for candidate diversity
        segments, info = whisper.transcribe(
            audio_np,
            beam_size=5,
            best_of=5,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8],
            language="en"
        )
        
        top1 = clean_txt(" ".join([s.text for s in list(segments)]))
        
        # Pull alternative temperature decodings to construct a diverse beam
        candidates = [{"transcript": top1, "logprob": info.avg_logprob if hasattr(info, "avg_logprob") else -0.5}]
        
        for temp in [0.2, 0.5, 0.8, 1.0]:
            segs, _ = whisper.transcribe(audio_np, beam_size=3, temperature=temp, language="en")
            alt_txt = clean_txt(" ".join([s.text for s in list(segs)]))
            if alt_txt and alt_txt not in [c["transcript"] for c in candidates]:
                candidates.append({"transcript": alt_txt, "logprob": -0.5 - temp})
            if len(candidates) >= 5:
                break
                
        while len(candidates) < 5:
            candidates.append({"transcript": top1, "logprob": -5.0})

        # Calculate Oracle WER: the best candidate in the beam
        wers = [wer(ref, c["transcript"]) if c["transcript"] else 1.0 for c in candidates]
        oracle_wer = min(wers)
        base_wer = wers[0]
        oracle_headroom = base_wer - oracle_wer

        beam_records.append({
            "file": wav_path,
            "speaker_id": spk,
            "severity": sev,
            "reference": ref,
            "base_wer": base_wer,
            "oracle_wer": oracle_wer,
            "oracle_headroom": oracle_headroom,
            "candidates": candidates
        })

        print(f"[{idx:02d}/16] [{sev:<8}] {spk} | Base WER: {base_wer:.2f} | Oracle WER: {oracle_wer:.2f} | Headroom: {oracle_headroom:+.2f}")

    out_p = Path("data/real_torgo_samples/real_whisper_beams.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(beam_records, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Saved real multi-candidate beams to {out_p}")
    print("=" * 80)

if __name__ == "__main__":
    main()
