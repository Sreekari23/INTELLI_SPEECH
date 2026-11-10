import os
import sys
from pathlib import Path
import torch
import torch.onnx

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.squeezeformer_ctc import SqueezeformerCTC, VOCAB

def export_squeezeformer_to_onnx():
    print("[*] Exporting SqueezeformerCTC to ONNX format...")
    device = torch.device("cpu")
    
    model = SqueezeformerCTC(
        input_dim=80,
        num_classes=len(VOCAB),
        encoder_dim=256,
        num_layers=8
    ).to(device)
    
    ckpt_path = REPO_ROOT / "checkpoints" / "best_squeezeformer_ctc1.pth"
    if not ckpt_path.exists():
        print(f"[!] Checkpoint not found at {ckpt_path}. Run training or place dry-run weights first.")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Dummy inputs matching typical spectrogram lengths: (Batch=1, Time=300, Freq=80)
    dummy_input = torch.randn(1, 300, 80, device=device)
    dummy_lens = torch.LongTensor([300], device=device)

    onnx_dir = REPO_ROOT / "checkpoints" / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "squeezeformer_ctc.onnx"

    torch.onnx.export(
        model,
        (dummy_input, dummy_lens),
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["features", "feature_lengths"],
        output_names=["logits", "output_lengths"],
        dynamic_axes={
            "features": {0: "batch_size", 1: "time_frames"},
            "feature_lengths": {0: "batch_size"},
            "logits": {0: "batch_size", 1: "time_frames_subsampled"},
            "output_lengths": {0: "batch_size"}
        }
    )

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"[+] Successfully exported ONNX model to {onnx_path} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    export_squeezeformer_to_onnx()
