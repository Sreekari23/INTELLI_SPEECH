"""TORGO Dataset Channel Audit & Audio-Text Hygiene Pipeline.

Tasks accomplished:
- Day 3: Channel isolation (headMic vs arrayMic), corrupted file detection.
- Day 4: Sample rate standardization (16 kHz mono), text normalization,
         and structured metadata manifest generation.
"""

import os
import re
import json
import argparse
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import soundfile as sf
import torch
import torchaudio

# Standard TORGO demographic and clinical severity mapping
TORGO_SPEAKER_METADATA: Dict[str, Dict[str, Any]] = {
    "F01": {"gender": "F", "severity": "Severe", "dysarthric": True},
    "F03": {"gender": "F", "severity": "Moderate", "dysarthric": True},
    "F04": {"gender": "F", "severity": "Moderate", "dysarthric": True},
    "M01": {"gender": "M", "severity": "Severe", "dysarthric": True},
    "M02": {"gender": "M", "severity": "Severe", "dysarthric": True},
    "M04": {"gender": "M", "severity": "Moderate", "dysarthric": True},
    "M05": {"gender": "M", "severity": "Mild", "dysarthric": True},
    "FC01": {"gender": "F", "severity": "Control", "dysarthric": False},
    "FC02": {"gender": "F", "severity": "Control", "dysarthric": False},
    "FC03": {"gender": "F", "severity": "Control", "dysarthric": False},
    "MC01": {"gender": "M", "severity": "Control", "dysarthric": False},
    "MC02": {"gender": "M", "severity": "Control", "dysarthric": False},
    "MC03": {"gender": "M", "severity": "Control", "dysarthric": False},
    "MC04": {"gender": "M", "severity": "Control", "dysarthric": False},
}


def normalize_transcript(raw_text: str) -> str:
    """Cleans and standardizes raw text transcripts.

    Operations:
    1. Removes parenthetical/bracketed non-speech annotations: [cough], (laughter), etc.
    2. Strips punctuation marks while preserving intra-word apostrophes.
    3. Normalizes whitespace and converts text to uppercase for ASR consistency.
    """
    if not raw_text:
        return ""

    # Remove annotations enclosed in brackets or parentheses
    text = re.sub(r"\[.*?\]", " ", raw_text)
    text = re.sub(r"\(.*?\)", " ", text)

    # Remove non-alphanumeric characters except apostrophes and whitespace
    text = re.sub(r"[^a-zA-Z0-9'\s]", " ", text)

    # Normalize multiple whitespaces into a single space
    text = re.sub(r"\s+", " ", text).strip()

    return text.upper()


def is_head_mic_path(audio_path: Union[str, Path]) -> bool:
    """Identifies if an audio path belongs to the close-talking headMic channel.

    TORGO contains simultaneous recordings across 'wav_headMic' and 'wav_arrayMic'.
    We strictly isolate the head-mounted microphone to avoid room reverberation.
    """
    path_str = str(audio_path).replace("\\", "/").lower()
    
    # Explicit rejection of array mic recordings
    if "arraymic" in path_str or "array_mic" in path_str:
        return False

    # Positive match for headMic directory or file naming convention
    if "headmic" in path_str or "head_mic" in path_str:
        return True

    # Fallback: if 'array' is in path, reject; otherwise return True if unspecified
    return "array" not in path_str


def get_speaker_id_from_path(file_path: Union[str, Path]) -> Optional[str]:
    """Extracts the standardized speaker ID (e.g. F01, M05, FC01) from path."""
    parts = Path(file_path).parts
    for part in parts:
        part_clean = part.upper()
        if part_clean in TORGO_SPEAKER_METADATA:
            return part_clean
        # Pattern match for F01, M02, FC01, MC04
        match = re.match(r"^(F|M|FC|MC)\d{2}$", part_clean)
        if match:
            return part_clean
    return None


def inspect_audio_file(audio_path: Path) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Inspects audio file validity, channel count, sample rate, and duration.

    Returns:
        (is_valid, audio_info_dict, error_reason)
    """
    if not audio_path.exists() or audio_path.stat().st_size == 0:
        return False, None, "File is missing or 0 bytes"

    try:
        info = sf.info(str(audio_path))
        if info.duration < 0.15:
            return False, None, f"Duration too short ({info.duration:.2f}s < 0.15s)"
        if info.duration > 25.0:
            return False, None, f"Duration exceeds ceiling ({info.duration:.2f}s > 25.0s)"

        metadata = {
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "duration": round(info.duration, 3),
            "format": info.format,
            "subtype": info.subtype,
        }
        return True, metadata, None
    except Exception as exc:
        return False, None, f"Soundfile read error: {str(exc)}"


def audit_torgo_corpus(
    dataset_root: Union[str, Path],
    min_duration_sec: float = 0.15,
    max_duration_sec: float = 25.0
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Audits entire TORGO directory hierarchy.

    Scans for:
    - Channel isolation (extracts headMic, drops arrayMic)
    - Transcript pairing (matches audio with prompts/*.txt)
    - Audio format integrity and duration bounds
    """
    dataset_path = Path(dataset_root)
    valid_records: List[Dict[str, Any]] = []
    
    stats = {
        "total_files_scanned": 0,
        "head_mic_files_found": 0,
        "array_mic_files_dropped": 0,
        "missing_prompt_dropped": 0,
        "empty_transcript_dropped": 0,
        "corrupted_audio_dropped": 0,
        "valid_utterances": 0,
        "total_valid_duration_seconds": 0.0,
        "speaker_breakdown": {},
        "severity_breakdown": {"Control": 0, "Mild": 0, "Moderate": 0, "Severe": 0},
    }

    if not dataset_path.exists():
        return valid_records, stats

    # Traverse all .wav files in the dataset
    for wav_file in dataset_path.rglob("*.wav"):
        stats["total_files_scanned"] += 1

        # 1. Channel Isolation: Head-Mounted Mic vs Room Array Mic
        if not is_head_mic_path(wav_file):
            stats["array_mic_files_dropped"] += 1
            continue

        stats["head_mic_files_found"] += 1

        # 2. Extract Speaker Identity
        speaker_id = get_speaker_id_from_path(wav_file)
        if not speaker_id:
            # Fallback check against parent directory
            speaker_id = "UNKNOWN"

        speaker_meta = TORGO_SPEAKER_METADATA.get(
            speaker_id,
            {"gender": "Unknown", "severity": "Unknown", "dysarthric": False}
        )

        # 3. Locate Corresponding Transcript Prompt File
        # TORGO structure typically: <session>/wav_headMic/<id>.wav -> <session>/prompts/<id>.txt
        stem = wav_file.stem
        prompt_candidates = [
            wav_file.parent.parent / "prompts" / f"{stem}.txt",
            wav_file.parent / f"{stem}.txt",
            wav_file.with_suffix(".txt"),
        ]

        prompt_file = None
        for candidate in prompt_candidates:
            if candidate.exists():
                prompt_file = candidate
                break

        if not prompt_file:
            stats["missing_prompt_dropped"] += 1
            continue

        # 4. Clean & Normalize Transcript
        try:
            raw_text = prompt_file.read_text(encoding="utf-8", errors="ignore").strip()
            clean_text = normalize_transcript(raw_text)
        except Exception:
            clean_text = ""

        if not clean_text:
            stats["empty_transcript_dropped"] += 1
            continue

        # 5. Audio Integrity & Duration Verification
        is_valid_audio, audio_meta, err_reason = inspect_audio_file(wav_file)
        if not is_valid_audio or not audio_meta:
            stats["corrupted_audio_dropped"] += 1
            continue

        # Record valid utterance
        utterance_id = f"{speaker_id}_{wav_file.parent.name}_{stem}"
        record = {
            "utterance_id": utterance_id,
            "speaker_id": speaker_id,
            "gender": speaker_meta["gender"],
            "severity": speaker_meta["severity"],
            "dysarthric": speaker_meta["dysarthric"],
            "raw_audio_path": str(wav_file.resolve()),
            "transcript": clean_text,
            "duration": audio_meta["duration"],
            "original_sample_rate": audio_meta["sample_rate"],
            "channels": audio_meta["channels"],
        }
        valid_records.append(record)

        # Update Statistics
        stats["valid_utterances"] += 1
        stats["total_valid_duration_seconds"] += audio_meta["duration"]

        if speaker_id not in stats["speaker_breakdown"]:
            stats["speaker_breakdown"][speaker_id] = {
                "utterances": 0,
                "duration_seconds": 0.0,
                "severity": speaker_meta["severity"],
            }
        stats["speaker_breakdown"][speaker_id]["utterances"] += 1
        stats["speaker_breakdown"][speaker_id]["duration_seconds"] += audio_meta["duration"]

        severity = speaker_meta.get("severity", "Unknown")
        if severity in stats["severity_breakdown"]:
            stats["severity_breakdown"][severity] += 1

    stats["total_valid_duration_seconds"] = round(stats["total_valid_duration_seconds"], 2)
    stats["total_valid_hours"] = round(stats["total_valid_duration_seconds"] / 3600.0, 3)

    return valid_records, stats


def standardize_and_resample_audio(
    source_audio_path: str,
    target_audio_path: str,
    target_sample_rate: int = 16000
) -> bool:
    """Resamples audio to 16 kHz mono float32 and saves as standardized PCM WAV."""
    try:
        # Load audio using soundfile (cross-platform, zero external codec dependency)
        data, sr = sf.read(source_audio_path, dtype="float32")

        # Standardize shape to (channels, samples)
        if data.ndim == 1:
            waveform = torch.from_numpy(data).unsqueeze(0)  # (1, samples)
        elif data.ndim == 2:
            waveform = torch.from_numpy(data.T)  # (channels, samples)
        else:
            raise ValueError(f"Unsupported audio tensor dimensions: {data.ndim}")

        # Convert to mono if multi-channel
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if sample rate doesn't match target
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
            waveform = resampler(waveform)

        # Convert to 1D float array for soundfile writing
        out_audio = waveform.squeeze(0).numpy()

        # Ensure output directory exists and write 16-bit PCM WAV
        target_path = Path(target_audio_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(target_path), out_audio, samplerate=target_sample_rate, subtype="PCM_16")
        return True
    except Exception as exc:
        print(f"Error standardizing {source_audio_path}: {exc}")
        return False


def export_audit_artifacts(
    records: List[Dict[str, Any]],
    stats: Dict[str, Any],
    manifest_output_path: str,
    report_output_path: str
) -> None:
    """Saves the inventory manifest and summary audit report."""
    manifest_p = Path(manifest_output_path)
    manifest_p.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    report_p = Path(report_output_path)
    report_p.parent.mkdir(parents=True, exist_ok=True)
    with open(report_p, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"[AUDIT COMPLETE] Manifest saved to: {manifest_p.resolve()}")
    print(f"[AUDIT COMPLETE] Report saved to:   {report_p.resolve()}")
    print(f"Valid Utterances: {stats['valid_utterances']}")
    print(f"Total Speech Duration: {stats.get('total_valid_hours', 0)} hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit and clean TORGO dataset.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to raw TORGO directory")
    parser.add_argument("--output_manifest", type=str, default="data/manifests/torgo_inventory.json")
    parser.add_argument("--output_report", type=str, default="data/manifests/torgo_audit_report.json")
    args = parser.parse_args()

    records, stats = audit_torgo_corpus(args.input_dir)
    export_audit_artifacts(records, stats, args.output_manifest, args.output_report)
