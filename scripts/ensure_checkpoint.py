#!/usr/bin/env python3
"""Materialize and verify the Wentian checkpoint stored in Git LFS."""

from __future__ import annotations

import gzip
import hashlib
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "weights" / "wentian_beta.pth.gz"
CHECKPOINT = ROOT / "weights" / "wentian_beta.pth"
EXPECTED_SIZE = 1_382_298_257
EXPECTED_SHA256 = "56b68db5ae3b64e698bcccce1b552dc4a60caa0c0113d1372365b0c9a6198120"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if CHECKPOINT.is_file() and CHECKPOINT.stat().st_size == EXPECTED_SIZE:
        print(f"checkpoint ready: {CHECKPOINT}")
        return
    if not ARCHIVE.is_file():
        raise SystemExit("compressed checkpoint is missing; run git lfs pull")

    temporary = CHECKPOINT.with_suffix(".pth.tmp")
    try:
        with gzip.open(ARCHIVE, "rb") as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        if temporary.stat().st_size != EXPECTED_SIZE:
            raise SystemExit(
                f"restored checkpoint size mismatch: {temporary.stat().st_size} != {EXPECTED_SIZE}"
            )
        actual = _sha256(temporary)
        if actual != EXPECTED_SHA256:
            raise SystemExit(f"restored checkpoint sha256 mismatch: {actual}")
        os.replace(temporary, CHECKPOINT)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"checkpoint restored: {CHECKPOINT}")


if __name__ == "__main__":
    main()
