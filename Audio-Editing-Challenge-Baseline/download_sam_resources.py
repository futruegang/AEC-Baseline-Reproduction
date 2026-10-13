"""Fetch only the SAM-Audio resources used by the baseline's default calls.

Run with a Python environment containing modelscope and huggingface_hub.
Use SAM_AUDIO_MODE=lean (one candidate, no automatic span prediction).
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MODELSCOPE_CACHE", str(ROOT / "ckpts" / ".modelscope-cache"))
os.environ.setdefault("MODELSCOPE_DOWNLOAD_PARALLELS", "8")

def main():
    from modelscope import snapshot_download
    jobs = [
        ("facebook/sam-audio-large", "sam-audio-large", None),
        ("AI-ModelScope/t5-base", "t5-base", ["*.json", "spiece.model", "model.safetensors"]),
    ]
    manifest = []
    for repo, folder, patterns in jobs:
        dest = ROOT / "ckpts" / folder
        print(f"Downloading {repo} -> {dest}", flush=True)
        snapshot_download(repo, local_dir=str(dest), allow_patterns=patterns)
        manifest.append({"repo": repo, "path": str(dest.relative_to(ROOT)), "source": "modelscope"})
    (ROOT / "ckpts" / "sam-resources.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Resources downloaded for the baseline's default single-candidate text separation.", flush=True)

if __name__ == "__main__":
    main()
