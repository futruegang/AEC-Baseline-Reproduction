"""Export AEC sources and small resources, with an exact omitted-file inventory."""
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT.parent / ".aec-github-submission"
PROJECT = ROOT / "Audio-Editing-Challenge-Baseline"
LIMIT = 100 * 1024 * 1024
SKIP_DIRS = {".git", ".cache", ".venv", "__pycache__", ".download-logs", ".pytest_cache", ".ruff_cache", "._____temp"}

def main():
    DEST.mkdir(exist_ok=True)
    skipped = []
    included = []
    for folder, dirs, names in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(names):
            src = Path(folder) / name
            rel = src.relative_to(ROOT)
            if name in {"未提交.md", "submission-manifest.json"}:
                continue
            size = src.stat().st_size
            reason = None
            if name in {".msc", ".mv", ".DS_Store"} or name.endswith(".pyc"):
                reason = "本地缓存/下载状态文件，不参与推理"
            elif size >= LIMIT:
                reason = "达到或超过普通 Git 单文件 100 MiB 限制"
            elif rel.parts[:3] == ("Audio-Editing-Challenge-Baseline", "MMAE", "wav"):
                reason = "MMAE 音频数据，体积较大，使用下载脚本恢复"
            if reason:
                skipped.append({"path": rel.as_posix(), "bytes": size, "reason": reason})
                continue
            dst = DEST / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            included.append({"path": rel.as_posix(), "bytes": size})
    revisions = {}
    for folder in [PROJECT, PROJECT / "AuK", PROJECT / "MMAE", PROJECT / "sam-audio", PROJECT / "sam-audio/external/dacvae", PROJECT / "sam-audio/external/perception_models"]:
        revisions[folder.relative_to(ROOT).as_posix()] = subprocess.check_output(["git", "-C", str(folder), "rev-parse", "HEAD"], text=True).strip()
    manifest = {"included": included, "omitted": skipped, "source_revisions": revisions}
    (ROOT / "submission-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 未提交文件明细", "", "路径相对于本地 `AEC/` 目录。所有列出的资源仍保留在本地，没有删除。", "", f"共 {len(skipped)} 个未提交文件，合计 {sum(f['bytes'] for f in skipped):,} 字节。嵌套 Git 历史、运行环境和临时缓存目录不属于项目源码，不打包；源码以普通目录提交，不依赖 Git submodule。", "", "## 恢复方法", "", "在已具备依赖的 Python 环境中进入 `Audio-Editing-Challenge-Baseline/`，执行：", "", "```bash", "python download_baseline_resources.py", "python verify_baseline_resources.py", "```", "", "下载脚本仅下载资源，不安装环境。权重来自 ModelScope，MMAE 音频来自 Hugging Face 的公开数据仓库。只恢复权重时使用 `--skip-audio`。", "", "## 明细", "", "| 本地文件 | 大小（字节） | 未提交原因 |", "| --- | ---: | --- |"]
    lines += [f"| `{f['path']}` | {f['bytes']:,} | {f['reason']} |" for f in skipped]
    (ROOT / "未提交.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name in ["未提交.md", "submission-manifest.json"]:
        shutil.copy2(ROOT / name, DEST / name)
    print(f"Exported {len(included)} files; omitted {len(skipped)} files; destination {DEST}")

if __name__ == "__main__":
    main()
