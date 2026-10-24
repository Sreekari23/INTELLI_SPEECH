import os
import glob
import io
import re
import json
import pyarrow.parquet as pq
import soundfile as sf
from pathlib import Path

def main():
    print("=" * 80)
    print(" PHASE 3: BUILDING CLINICAL SPEAKER-STRATIFIED DATA SPLITS")
    print("=" * 80)

    # 1. Define Strict Speaker Splits (Zero Leakage)
    SPLITS = {
        # TRAIN: M02 (Severe), F03 (Mod), MC03, MC04, FC03 (Controls)
        "M02": "train", "F03": "train", "MC03": "train", "MC04": "train", "FC03": "train",
        # VAL: F01 (Severe), M04 (Mod), MC02, FC02 (Controls)
        "F01": "val", "M04": "val", "MC02": "val", "FC02": "val",
        # TEST: M01 (Severe), F04 (Mod), M05 (Mild), MC01, FC01 (Controls)
        "M01": "test", "F04": "test", "M05": "test", "MC01": "test", "FC01": "test"
    }

    base = os.path.expanduser("~/.cache/huggingface/hub/datasets--abnerh--TORGO-database")
    files = sorted(glob.glob(os.path.join(base, "**", "*.parquet"), recursive=True))
    
    if not files:
        raise FileNotFoundError("Could not find Hugging Face TORGO parquet cache!")

    # 2. Setup output directories
    out_base = Path("data/processed")
    manifests = {"train": [], "val": [], "test": []}
    
    for split in manifests.keys():
        (out_base / split / "wavs").mkdir(parents=True, exist_ok=True)

    print(f"[*] Found {len(files)} parquet shards. Extracting and routing audio...")

    counts = {"train": 0, "val": 0, "test": 0, "skipped": 0}

    # 3. Process Shards
    for f in files:
        t = pq.ParquetFile(f).read()
        d = t.to_pydict()
        
        for i in range(t.num_rows):
            txt = str(d["transcription"][i] or "").strip()
            # For acoustic training, we keep any valid utterance (>= 1 word)
            if not txt or len(txt) < 2:
                counts["skipped"] += 1
                continue

            audio = d["audio"][i]
            path_val = audio.get("path") if isinstance(audio, dict) else ""
            if isinstance(path_val, list) and path_val:
                path_val = path_val[0]
            path_str = str(path_val or "")

            # Filter for clean head-mounted mic only
            if "headmic" not in path_str.lower():
                counts["skipped"] += 1
                continue

            # Extract Speaker ID
            m = re.match(r"^([A-Z0-9]+)_", path_str.split("/")[-1])
            if not m:
                counts["skipped"] += 1
                continue
            
            spk = m.group(1).upper()
            split_name = SPLITS.get(spk)
            
            if not split_name:
                counts["skipped"] += 1
                continue

            # Extract and decode audio bytes
            raw_b = audio.get("bytes")
            if isinstance(raw_b, list) and raw_b:
                raw_b = raw_b[0]
            if not isinstance(raw_b, (bytes, bytearray)):
                counts["skipped"] += 1
                continue

            arr, sr = sf.read(io.BytesIO(raw_b), dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=1) # Downmix stereo to mono if necessary

            # Write WAV to disk
            fn = f"{spk}_{counts[split_name]:04d}.wav"
            dest_path = out_base / split_name / "wavs" / fn
            sf.write(str(dest_path), arr, sr)

            manifests[split_name].append({
                "file": f"wavs/{fn}",
                "speaker_id": spk,
                "text": txt,
                "duration_sec": round(len(arr) / sr, 3)
            })
            
            counts[split_name] += 1

    # 4. Save Manifests
    for split, records in manifests.items():
        mf_path = out_base / split / "manifest.json"
        with open(mf_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    print("\n" + "=" * 80)
    print(" EXTRACTION COMPLETE: STRICT SPEAKER SPLITS")
    print("-" * 80)
    print(f" Train set:      {counts['train']:>5} files  --> data/processed/train/")
    print(f" Validation set: {counts['val']:>5} files  --> data/processed/val/")
    print(f" Test set:       {counts['test']:>5} files  --> data/processed/test/")
    print(f" Skipped:        {counts['skipped']:>5} files  (array mics, missing text, unknown spk)")
    print("=" * 80)

if __name__ == "__main__":
    main()
