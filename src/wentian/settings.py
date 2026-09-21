"""Validated settings shared by the CLI and inference entry points."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wentian.runtime.topology import RuntimeTopology

DEFAULT_BENCHMARK_LEAD_TIME = 0.1
DEFAULT_INFERENCE_LEAD_TIME = 0.0
MAX_ALLOWED_RMSE_PERCENT = 0.001
DEFAULT_MAX_RMSE_PERCENT = MAX_ALLOWED_RMSE_PERCENT
DEFAULT_WORKER_TIMEOUT_SECONDS = 300.0
SUPPORTED_PRECISIONS = ("fp32", "fp64")
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "weights" / "wentian_beta.pth"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, "1" if default else "0")
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


@dataclass(frozen=True)
class HBMSettings:
    enabled: bool
    required: bool
    use_buffers: bool
    use_cache: bool
    base_node: int
    cache_mb: float
    probe_mb: float

    @classmethod
    def from_env(cls) -> HBMSettings:
        settings = cls(
            enabled=_env_flag("WT_HBM", default=True),
            required=_env_flag("WT_HBM_REQUIRED"),
            use_buffers=_env_flag("WT_HBM_BUF", default=True),
            use_cache=_env_flag("WT_HBM_CACHE", default=True),
            base_node=int(os.environ.get("GD_HBM_BASE", "16")),
            cache_mb=float(os.environ.get("GD_HBM_CACHE_MB", "96")),
            probe_mb=float(os.environ.get("WT_HBM_PROBE_MB", "16")),
        )
        if settings.required and not settings.enabled:
            raise ValueError("WT_HBM_REQUIRED=1 conflicts with WT_HBM=0")
        if not math.isfinite(settings.cache_mb) or settings.cache_mb < 0.0:
            raise ValueError("GD_HBM_CACHE_MB must be finite and non-negative")
        if not math.isfinite(settings.probe_mb) or settings.probe_mb <= 0.0:
            raise ValueError("WT_HBM_PROBE_MB must be finite and positive")
        return settings


@dataclass(frozen=True)
class RuntimeSettings:
    checkpoint: Path
    precision: str
    topology: RuntimeTopology
    lead_time: float
    max_rmse_percent: float
    verify: bool
    copy_model_per_worker: bool
    warmups: int
    repeats: int
    worker_timeout_seconds: float
    hbm: HBMSettings

    @classmethod
    def from_env(cls) -> RuntimeSettings:
        from wentian.runtime.topology import RuntimeTopology

        precision = os.environ.get("WENTIAN_PRECISION", "fp32").lower()
        if precision not in SUPPORTED_PRECISIONS:
            choices = ", ".join(SUPPORTED_PRECISIONS)
            raise ValueError(f"WENTIAN_PRECISION must be one of {choices}")
        settings = cls(
            checkpoint=Path(os.environ.get("WENTIAN_CHECKPOINT", DEFAULT_CHECKPOINT)).resolve(),
            precision=precision,
            topology=RuntimeTopology.from_environment(precision),
            lead_time=float(os.environ.get("WENTIAN_LEAD_TIME", DEFAULT_INFERENCE_LEAD_TIME)),
            max_rmse_percent=float(
                os.environ.get("WENTIAN_MAX_RMSE_PERCENT", DEFAULT_MAX_RMSE_PERCENT)
            ),
            verify=_env_flag("WENTIAN_VERIFY"),
            copy_model_per_worker=_env_flag("LOCALCOPY", default=True),
            warmups=max(0, int(os.environ.get("WENTIAN_WARMUPS", "0"))),
            repeats=max(1, int(os.environ.get("WENTIAN_REPEATS", "1"))),
            worker_timeout_seconds=float(
                os.environ.get("WENTIAN_WORKER_TIMEOUT_SECONDS", DEFAULT_WORKER_TIMEOUT_SECONDS)
            ),
            hbm=HBMSettings.from_env(),
        )
        if not math.isfinite(settings.lead_time):
            raise ValueError("WENTIAN_LEAD_TIME must be finite")
        if (
            not math.isfinite(settings.max_rmse_percent)
            or settings.max_rmse_percent < 0.0
            or settings.max_rmse_percent > MAX_ALLOWED_RMSE_PERCENT
        ):
            raise ValueError(
                "WENTIAN_MAX_RMSE_PERCENT must be finite and between "
                f"0 and {MAX_ALLOWED_RMSE_PERCENT:.3f}"
            )
        if (
            not math.isfinite(settings.worker_timeout_seconds)
            or settings.worker_timeout_seconds <= 0.0
        ):
            raise ValueError("WENTIAN_WORKER_TIMEOUT_SECONDS must be finite and positive")
        return settings
