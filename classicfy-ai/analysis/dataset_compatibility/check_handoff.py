"""Verify the tracked handoff artifacts using only the Python standard library."""
import csv
import hashlib
from pathlib import Path

BASE = Path(__file__).resolve().parent


def main():
    checked = 0
    for manifest in sorted(BASE.glob("*/artifact_manifest.csv")):
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            path = manifest.parent / row["path"]
            if not path.is_file():
                raise SystemExit(f"Missing artifact: {path}")
            with path.open("rb") as stream:
                prefix = stream.read(128)
                if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
                    raise SystemExit(f"Git LFS object not downloaded: {path.name}. Run git lfs pull.")
                stream.seek(0)
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if path.stat().st_size != int(row["bytes"]) or digest != row["sha256"]:
                raise SystemExit(f"Artifact integrity mismatch: {path}")
            checked += 1
    if not checked:
        raise SystemExit("No artifact manifests found")
    print(f"PASS: {checked} handoff artifacts, including Git LFS sequence caches")


if __name__ == "__main__":
    main()
