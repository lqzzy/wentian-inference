"""Independent standard Wentian forward pass used as the correctness reference."""

import math
from dataclasses import dataclass

import torch

from wentian.model.loading import load_wentian_model
from wentian.precision import precision_dtype
from wentian.settings import MAX_ALLOWED_RMSE_PERCENT


@dataclass(frozen=True)
class VerificationResult:
    plevel_rmse_percent: float
    surface_rmse_percent: float


def _relative_rmse_percent(actual, reference) -> float:
    rmse = (actual - reference).pow(2).mean().sqrt()
    scale = reference.abs().max()
    if scale.item() == 0.0:
        return 0.0 if rmse.item() == 0.0 else float("inf")
    return (rmse / scale * 100).item()


def standard_forward(checkpoint, plevel, surface, lead_time, precision: str = "fp32"):
    """Load a fresh canonical model and run its unmodified ``forward`` method."""
    dtype = precision_dtype(precision)
    model = load_wentian_model(checkpoint).to(dtype=dtype)
    plevel = plevel.to(dtype=dtype)
    surface = surface.to(dtype=dtype)
    if torch.is_tensor(lead_time):
        lead_time = lead_time.to(dtype=dtype)
    with torch.no_grad():
        return model(plevel, surface, lead_time)


def verify_outputs(
    optimized_plevel,
    optimized_surface,
    reference_plevel,
    reference_surface,
    max_rmse_percent: float,
) -> VerificationResult:
    """Compare optimized outputs with the standard result and enforce the limit."""
    if (
        not math.isfinite(max_rmse_percent)
        or max_rmse_percent < 0.0
        or max_rmse_percent > MAX_ALLOWED_RMSE_PERCENT
    ):
        raise ValueError(
            f"max_rmse_percent must be finite and between 0 and {MAX_ALLOWED_RMSE_PERCENT:.3f}"
        )

    output_pairs = (
        ("plevel", optimized_plevel, reference_plevel),
        ("surface", optimized_surface, reference_surface),
    )
    for name, optimized, reference in output_pairs:
        if optimized.shape != reference.shape:
            raise ValueError(
                f"{name} output shape mismatch: {tuple(optimized.shape)} != "
                f"{tuple(reference.shape)}"
            )

    result = VerificationResult(
        plevel_rmse_percent=_relative_rmse_percent(optimized_plevel, reference_plevel),
        surface_rmse_percent=_relative_rmse_percent(optimized_surface, reference_surface),
    )
    rmse_values = (result.plevel_rmse_percent, result.surface_rmse_percent)
    if (
        any(not math.isfinite(value) for value in rmse_values)
        or max(rmse_values) > max_rmse_percent
    ):
        raise RuntimeError(
            "standard reference mismatch: "
            f"plevel={result.plevel_rmse_percent:.9f}% "
            f"surface={result.surface_rmse_percent:.9f}% "
            f"limit={max_rmse_percent:.9f}%"
        )
    return result
