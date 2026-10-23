import pyarrow.parquet as pq, glob, os, io, re, json, soundfile as sf
from pathlib import Path

base = os.path.expanduser("~/.cache/huggingface/hub/datasets--abnerh--TORGO-database")
files = sorted(glob.glob(os.path.join(base, "**", "*.parquet"), recursive=True))

SEVERITY = {
    "FC01": "Control", "FC02": "Control", "FC03": "Control",
    "MC01": "Control", "MC02": "Control", "MC03": "Control", "MC04": "Control",
    "M05": "Mild",
    "F03": "Moderate", "F04": "Moderate", "M04": "Moderate",
    "F01": "Severe", "M01": "Severe", "M02": "Severe"
}

out_dir = Path("data/real_torgo_samples")
out_dir.mkdir(parents=True, exist_ok=True)
manifest = []
buckets = {"Control": [], "Mild": [], "Moderate": [], "Severe": []}

for f in files:
    t = pq.ParquetFile(f).read()
    d = t.to_pydict()
    for i in range(t.num_rows):
        txt = str(d["transcription"][i] or "").strip()
        if len(txt.split()) < 3:
            continue

        audio = d["audio"][i]
        path_val = audio.get("path") if isinstance(audio, dict) else ""
        if isinstance(path_val, list) and path_val:
            path_val = path_val[0]
        path_str = str(path_val or "")

        if "headmic" not in path_str.lower():
            continue

        m = re.match(r"^([A-Z0-9]+)_", path_str.split("/")[-1])
        if not m:
            continue
        spk = m.group(1).upper()
        sev = SEVERITY.get(spk)
        if not sev or len(buckets[sev]) >= 4:
            continue

        raw_b = audio.get("bytes")
        if isinstance(raw_b, list) and raw_b:
            raw_b = raw_b[0]
        if not isinstance(raw_b, (bytes, bytearray)):
            continue

        arr, sr = sf.read(io.BytesIO(raw_b), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)

        idx = len(buckets[sev]) + 1
        fn = f"{sev.lower()}_case_{idx:02d}_{spk}.wav"
        dest_path = out_dir / fn
        sf.write(str(dest_path), arr, sr)

        entry = {
            "file": str(dest_path),
            "filename": fn,
            "speaker_id": spk,
            "severity": sev,
            "reference": txt,
            "sample_rate": sr,
            "duration_sec": round(len(arr) / sr, 2)
        }
        buckets[sev].append(entry)
        manifest.append(entry)
        print(f"[+] Extracted {sev} ({idx}/4): {fn} -> \"{txt}\"")

    if all(len(v) >= 4 for v in buckets.values()):
        break

with open(out_dir / "manifest.json", "w") as mf:
    json.dump(manifest, mf, indent=2)

print("\n[OK] Extracted 16 real head-mic samples to data/real_torgo_samples/manifest.json")
