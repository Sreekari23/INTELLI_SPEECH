import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
onnx_dir = REPO_ROOT / "checkpoints" / "onnx"
model_path = onnx_dir / "squeezeformer_ctc.onnx"
quantized_path = onnx_dir / "squeezeformer_ctc_int8.onnx"

print(f"[*] Loading original ONNX model from {model_path}...")
model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
print(f"[*] Original ONNX Size: {model_size_mb:.2f} MB")

print("[*] Applying INT8 Dynamic Quantization...")
quantize_dynamic(
    model_input=str(model_path),
    model_output=str(quantized_path),
    weight_type=QuantType.QUInt8
)

quant_size_mb = os.path.getsize(quantized_path) / (1024 * 1024)
print(f"[+] Quantized INT8 ONNX successfully saved to {quantized_path}")
print(f"[+] Quantized ONNX Size: {quant_size_mb:.2f} MB (Compression ratio: {model_size_mb/quant_size_mb:.2f}x)")