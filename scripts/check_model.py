#!/usr/bin/env python3
"""Instantiate Wentian and strictly load the repository checkpoint."""

from __future__ import annotations

import os
from pathlib import Path

from wentian.model.loading import load_wentian_model

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path(os.environ.get("WENTIAN_CHECKPOINT", REPO_ROOT / "weights" / "wentian_beta.pth"))


def main() -> None:
    model = load_wentian_model(CHECKPOINT)
    parameters = sum(value.numel() for value in model.parameters())
    print(f"model OK: parameters={parameters} checkpoint={CHECKPOINT}")


if __name__ == "__main__":
    main()
