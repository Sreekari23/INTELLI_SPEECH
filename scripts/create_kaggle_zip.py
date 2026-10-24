import zipfile
import os
from pathlib import Path

zip_path = "tide_training_data.zip"
if os.path.exists(zip_path):
    os.remove(zip_path)

print("[*] Creating Kaggle-compatible zip file...")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for folder in ["data/processed", "src"]:
        if os.path.exists(folder):
            for path in Path(folder).rglob("*"):
                if path.is_file():
                    # Force Unix-style forward slashes for Kaggle
                    arcname = str(path).replace("\\", "/")
                    zf.write(path, arcname)

print(f"[OK] Saved {zip_path} with Linux-friendly paths!")
