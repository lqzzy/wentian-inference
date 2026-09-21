"""Floating-point precision selection for Wentian inference."""

import torch

from wentian.settings import SUPPORTED_PRECISIONS

_DTYPES = {
    "fp32": torch.float32,
    "fp64": torch.float64,
}


def precision_dtype(precision: str) -> torch.dtype:
    """Return the PyTorch dtype for a validated precision name."""
    try:
        return _DTYPES[precision]
    except KeyError as exc:
        choices = ", ".join(SUPPORTED_PRECISIONS)
        raise ValueError(f"precision must be one of {choices}, got {precision!r}") from exc
