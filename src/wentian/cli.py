"""Minimal command-line entry point for Wentian forecasts."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from wentian.runtime.topology import RuntimeTopology
from wentian.settings import DEFAULT_CHECKPOINT, SUPPORTED_PRECISIONS

FORECAST_STEPS = 60
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wentian",
        description="Run a 15-day Wentian forecast on a Kunpeng 920F node.",
    )
    parser.add_argument("--version", action="version", version="wentian 1.1.0")
    parser.add_argument("precision", choices=SUPPORTED_PRECISIONS, help="fp32 or fp64")
    parser.add_argument("input_root", type=Path, help="directory containing raw input folders")
    parser.add_argument("timestamp", help="latest input time in YYYYMMDDHH format")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    checkpoint = DEFAULT_CHECKPOINT.resolve()
    input_root = args.input_root.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}; run 'git lfs pull'")
    if not input_root.is_dir():
        raise SystemExit(f"input directory not found: {input_root}")

    env = os.environ.copy()
    topology = RuntimeTopology.from_environment(args.precision, {})
    output_root = REPOSITORY_ROOT / "outputs" / args.precision
    env.update(topology.environment())
    env.update(
        {
            "WENTIAN_CHECKPOINT": str(checkpoint),
            "WENTIAN_PRECISION": args.precision,
            "WENTIAN_INPUT_ROOT": str(input_root),
            "WENTIAN_TIMESTAMP": args.timestamp,
            "WENTIAN_FORECAST_OUTPUT_DIR": str(output_root),
            "WENTIAN_FORECAST_STEPS": str(FORECAST_STEPS),
            "LOCALCOPY": "1",
        }
    )
    os.execvpe(sys.executable, [sys.executable, "-m", "wentian.forecast"], env)


if __name__ == "__main__":
    main()
