#!/usr/bin/env python3
"""Verify repository artifacts without importing the target-specific torch build."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "weights" / "wentian_beta.pth"
CHECKPOINT_ARCHIVE = ROOT / "weights" / "wentian_beta.pth.gz"
CONSTANTS = ROOT / "src/wentian/data/constants/era5_infer.npz"
EXPECTED_SIZE = 1_382_298_257
EXPECTED_SHA256 = "56b68db5ae3b64e698bcccce1b552dc4a60caa0c0113d1372365b0c9a6198120"
EXPECTED_ARCHIVE_SIZE = 274_911_774
EXPECTED_ARCHIVE_SHA256 = "9071dbe34001aa620cfa1378e1870d1d35960fa426ead6d7ba07ff5cbb461a3b"
EXPECTED_CONSTANTS_SHA256 = "72041544f06252dcf0f753533c895ef381f6084db65cbde0bb7816b73c607419"
REQUIRED = (
    ROOT / "src/wentian/data/preprocess.py",
    ROOT / "src/wentian/forecast.py",
    ROOT / "src/wentian/model/config.py",
    ROOT / "src/wentian/model/layers.py",
    ROOT / "src/wentian/model/loading.py",
    ROOT / "src/wentian/model/network.py",
    ROOT / "src/wentian/model/utils.py",
    ROOT / "src/wentian/reference.py",
    ROOT / "src/wentian/settings.py",
    ROOT / "src/wentian/runtime/executor.py",
    ROOT / "src/wentian/runtime/numa.py",
    ROOT / "src/wentian/runtime/operators.py",
    ROOT / "src/wentian/runtime/workers.py",
    ROOT / "run_fp32.sh",
    ROOT / "run_fp64.sh",
    ROOT / "scripts/ensure_checkpoint.py",
    ROOT / "scripts/submit_920f.sh",
    ROOT / "scripts/run_920f.sbatch",
    ROOT / "scripts/run_920f_job.sh",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="skip the full checkpoint hash")
    args = parser.parse_args()

    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.is_file()]
    if missing:
        raise SystemExit("missing required files: " + ", ".join(missing))
    if not CHECKPOINT_ARCHIVE.is_file():
        raise SystemExit("compressed checkpoint is missing; run git lfs pull")
    if CHECKPOINT_ARCHIVE.stat().st_size != EXPECTED_ARCHIVE_SIZE:
        raise SystemExit(
            f"checkpoint archive size mismatch: {CHECKPOINT_ARCHIVE.stat().st_size} "
            f"!= {EXPECTED_ARCHIVE_SIZE}; the checkout may contain an LFS pointer"
        )
    if not args.fast:
        archive_sha256 = sha256(CHECKPOINT_ARCHIVE)
        if archive_sha256 != EXPECTED_ARCHIVE_SHA256:
            raise SystemExit(f"checkpoint archive sha256 mismatch: {archive_sha256}")
    if CHECKPOINT.is_file():
        if CHECKPOINT.stat().st_size != EXPECTED_SIZE:
            raise SystemExit(
                f"restored checkpoint size mismatch: {CHECKPOINT.stat().st_size} != {EXPECTED_SIZE}"
            )
        if not args.fast:
            actual = sha256(CHECKPOINT)
            if actual != EXPECTED_SHA256:
                raise SystemExit(f"restored checkpoint sha256 mismatch: {actual}")
    constants_sha256 = sha256(CONSTANTS)
    if constants_sha256 != EXPECTED_CONSTANTS_SHA256:
        raise SystemExit(f"constants sha256 mismatch: {constants_sha256}")
    print(
        f"artifacts OK: checkpoint_archive={CHECKPOINT_ARCHIVE.stat().st_size} bytes "
        f"constants={CONSTANTS.stat().st_size} bytes" + (" (fast)" if args.fast else "")
    )


if __name__ == "__main__":
    main()
