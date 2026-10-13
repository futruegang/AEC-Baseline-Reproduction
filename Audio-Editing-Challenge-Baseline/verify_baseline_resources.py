"""Verify copied baseline files offline; no downloads or environment installation."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-root", type=Path,
                        help="optional project directory containing separately stored weights")
    args = parser.parse_args()
    records = json.loads((ROOT / "ckpts/baseline-verification.json").read_text(encoding="utf-8"))
    if len(records) != 18 or len({r["file"] for r in records}) != 18:
        raise SystemExit("Expected the complete 18-file reference inventory")
    failures = []
    for record in records:
        rel = Path(record["file"].replace("\\", "/"))
        path = ROOT / rel
        if not path.is_file() and args.resource_root:
            path = args.resource_root / rel
        expected = record["expected_sha256"]
        valid = bool(expected) and path.is_file() and path.stat().st_size == record["expected_bytes"]
        if valid:
            digest = hashlib.sha256()
            with path.open("rb") as fin:
                for block in iter(lambda: fin.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            valid = digest.hexdigest() == expected
        print(f"{'OK' if valid else 'MISSING/CORRUPT'} {rel.as_posix()}", flush=True)
        if not valid:
            failures.append(rel.as_posix())
    if failures:
        raise SystemExit(1)
    print("All 18 reference files verified.")

if __name__ == "__main__":
    main()
