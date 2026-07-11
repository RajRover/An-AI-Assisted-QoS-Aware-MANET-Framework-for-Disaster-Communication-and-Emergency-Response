#!/usr/bin/env python3
"""
scripts/download_model.py
--------------------------
Downloads the trained disaster-message classifier weights from Hugging Face
Hub into the exact local folder Disaster_Prediction/classifier.py expects
(Disaster_Prediction/models/multi_class_disaster_model/), so the rest of
the codebase needs zero changes.

Model source:
    https://huggingface.co/rajvishwakarmaNIT/disaster-message-classifier

Usage (run once after cloning the repo):
    pip install huggingface_hub
    python scripts/download_model.py
"""

import os
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Missing dependency. Install it first:\n    pip install huggingface_hub")
    sys.exit(1)

REPO_ID = "rajvishwakarmaNIT/disaster-message-classifier"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
TARGET_DIR = os.path.join(
    _PROJECT_ROOT, "Disaster_Prediction", "models", "multi_class_disaster_model"
)


def main() -> None:
    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"Downloading '{REPO_ID}' into:\n    {TARGET_DIR}\n")

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=TARGET_DIR,
        local_dir_use_symlinks=False,  # real files, not symlinks into the HF cache
    )

    required = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
    missing = [f for f in required if not os.path.exists(os.path.join(TARGET_DIR, f))]

    if missing:
        print(f"\n⚠️  Download finished, but expected file(s) missing: {missing}")
        print("   Check the file names in the Hugging Face repo match classifier.py's expectations.")
        sys.exit(1)

    print("\n✅ Model downloaded successfully. You can now run the app/dashboard.")


if __name__ == "__main__":
    main()