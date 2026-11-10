import zipfile
import os
from pathlib import Path

zip_path = "tide_training_data.zip"
if os.path.exists(zip_path):
    os.remove(zip_path)

print("[*] Creating Kaggle-compatible zip file (data/processed, src, scripts)...")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for folder in ["data/processed", "src", "scripts"]:
        if os.path.exists(folder):
            for path in Path(folder).rglob("*"):
                if path.is_file() and "__pycache__" not in str(path) and not path.name.endswith(".pyc"):
                    arcname = str(path).replace("\\", "/")
                    zf.write(path, arcname)

size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print(f"[OK] Saved {zip_path} ({size_mb:.2f} MB) with Linux-friendly paths!")
