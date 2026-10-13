"""Restore required baseline weights/data only; does not install environments."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-audio", action="store_true")
    args = parser.parse_args()
    from modelscope import snapshot_download
    jobs = [
        ("Tencent-Hunyuan/AuK", "AuK/ckpts/AuK", None),
        ("Qwen/Qwen2.5-Omni-3B", "AuK/ckpts/Qwen2.5-Omni-3B", None),
        ("iic/SenseVoiceSmall", "ckpts/SenseVoiceSmall", None),
    ]
    for repo, folder, patterns in jobs:
        snapshot_download(repo, local_dir=str(ROOT / folder), allow_patterns=patterns)
    from download_sam_resources import main as download_sam
    download_sam()
    if not args.skip_audio:
        from huggingface_hub import snapshot_download as hf_download
        hf_download("BoJack/MMAE", repo_type="dataset", allow_patterns=["wav/**"],
                    local_dir=str(ROOT / "MMAE"))

if __name__ == "__main__":
    main()
