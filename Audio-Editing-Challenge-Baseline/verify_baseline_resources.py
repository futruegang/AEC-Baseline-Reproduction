"""Validate local weight sizes/hashes against ModelScope metadata; no GPU needed."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = [
    ("Tencent-Hunyuan/AuK", "AuK/ckpts/AuK", ["auk_base.safetensors", "vae.safetensors", "config.yaml"]),
    ("Qwen/Qwen2.5-Omni-3B", "AuK/ckpts/Qwen2.5-Omni-3B", None),
    ("iic/SenseVoiceSmall", "ckpts/SenseVoiceSmall", ["model.pt", "config.yaml", "tokens.json"]),
    ("facebook/sam-audio-large", "ckpts/sam-audio-large", ["checkpoint.pt", "config.json"]),
    ("AI-ModelScope/t5-base", "ckpts/t5-base", ["model.safetensors", "config.json", "spiece.model", "tokenizer.json"]),
]

def main():
    from modelscope.hub.api import HubApi
    api = HubApi()
    report = []
    for repo, folder, selected in MODELS:
        print(f"Checking {repo}", flush=True)
        files = api.get_model_files(repo, recursive=True)
        if selected is None:
            index = json.loads((ROOT / folder / "model.safetensors.index.json").read_text(encoding="utf-8"))
            selected = sorted(set(index["weight_map"].values())) + ["config.json", "tokenizer.json", "preprocessor_config.json"]
        available = {meta["Path"] for meta in files if meta["Type"] == "blob"}
        absent = set(selected) - available
        if absent:
            raise RuntimeError(f"Required files absent from {repo} metadata: {sorted(absent)}")
        for meta in files:
            if meta["Type"] != "blob" or meta["Path"] not in selected:
                continue
            path = ROOT / folder / meta["Path"]
            result = {"repo": repo, "file": str(path.relative_to(ROOT)), "expected_bytes": meta["Size"], "expected_sha256": meta.get("Sha256")}
            result["size_ok"] = path.is_file() and path.stat().st_size == meta["Size"]
            if result["size_ok"]:
                digest = hashlib.sha256()
                with path.open("rb") as fin:
                    for block in iter(lambda: fin.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                result["actual_sha256"] = digest.hexdigest()
                result["hash_ok"] = not meta.get("Sha256") or result["actual_sha256"] == meta["Sha256"]
            else:
                result["hash_ok"] = False
            print(f"  {'OK' if result['size_ok'] and result['hash_ok'] else 'MISSING/CORRUPT'} {meta['Path']}", flush=True)
            report.append(result)
    (ROOT / "ckpts" / "baseline-verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not all(row["size_ok"] and row["hash_ok"] for row in report):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
