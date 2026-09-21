"""Run the canonical 60-step Wentian autoregressive forecast."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.multiprocessing as mp

from wentian.data import INPUT_INTERVAL_HOURS, load_constants, load_raw_inputs
from wentian.precision import precision_dtype
from wentian.runtime.executor import NUMAInferenceSession
from wentian.settings import RuntimeSettings

MAX_FORECAST_STEPS = 60


def _forecast_timestamp(initial_timestamp: str, step: int) -> str:
    initial = datetime.strptime(initial_timestamp, "%Y%m%d%H")
    return (initial + timedelta(hours=INPUT_INTERVAL_HOURS * step)).strftime("%Y%m%d%H")


def _denormalize_outputs(
    plevel: torch.Tensor,
    surface: torch.Tensor,
    constants: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    plevel = plevel * constants["plevel_std"] + constants["plevel_mean"]
    surface = surface * constants["surface_std"] + constants["surface_mean"]
    torch.nan_to_num_(plevel)
    torch.nan_to_num_(surface)
    plevel[:, 1].clamp_(min=0.0)
    return plevel, surface


def _advance_inputs(
    plevel_history: torch.Tensor,
    surface_history: torch.Tensor,
    plevel_output: torch.Tensor,
    surface_output: torch.Tensor,
    constants: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    next_plevel = (plevel_output - constants["plevel_mean"]) / constants["plevel_std"]
    next_surface = (surface_output - constants["surface_mean"]) / constants["surface_std"]
    next_surface = torch.concat(
        [next_surface, constants["auxiliary"].unsqueeze(0)],
        dim=1,
    )
    torch.nan_to_num_(next_plevel)
    torch.nan_to_num_(next_surface)

    return (
        torch.stack([plevel_history[1], next_plevel[0]]).contiguous(),
        torch.stack([surface_history[1], next_surface[0]]).contiguous(),
    )


def _save_step(
    output_directory: Path,
    timestamp: str,
    step: int,
    lead_time: float,
    wall_seconds: float,
    plevel: torch.Tensor,
    surface: torch.Tensor,
    precision: str,
) -> None:
    torch.save(
        {
            "plevel": plevel[0].contiguous(),
            "surface": surface[0].contiguous(),
            "timestamp": timestamp,
            "step": step,
            "lead_time": lead_time,
            "wall_seconds": wall_seconds,
            "normalized": False,
            "precision": precision,
        },
        output_directory / f"{timestamp}.pt",
    )


def main() -> None:
    mp.set_start_method("fork")
    torch.set_num_threads(1)
    settings = RuntimeSettings.from_env()
    input_root = os.environ.get("WENTIAN_INPUT_ROOT")
    initial_timestamp = os.environ.get("WENTIAN_TIMESTAMP")
    output_root = os.environ.get("WENTIAN_FORECAST_OUTPUT_DIR")
    constants_path = os.environ.get("WENTIAN_CONSTANTS")
    steps = int(os.environ.get("WENTIAN_FORECAST_STEPS", str(MAX_FORECAST_STEPS)))
    if not input_root or not initial_timestamp or not output_root:
        raise ValueError(
            "WENTIAN_INPUT_ROOT, WENTIAN_TIMESTAMP, and WENTIAN_FORECAST_OUTPUT_DIR are required"
        )
    if not 1 <= steps <= MAX_FORECAST_STEPS:
        raise ValueError(f"forecast steps must be between 1 and {MAX_FORECAST_STEPS}")

    output_directory = Path(output_root).resolve() / f"{initial_timestamp}_pred"
    output_directory.mkdir(parents=True, exist_ok=True)
    existing = list(output_directory.glob("*.pt"))
    if existing:
        raise FileExistsError(
            f"forecast output directory already contains step files: {output_directory}"
        )

    constants = load_constants(constants_path)
    plevel_history, surface_history = load_raw_inputs(
        input_root,
        initial_timestamp,
        constants_path,
    )
    dtype = precision_dtype(settings.precision)
    constants = {name: value.to(dtype=dtype) for name, value in constants.items()}
    plevel_history = plevel_history.to(dtype=dtype)
    surface_history = surface_history.to(dtype=dtype)
    forward_walls = []
    forecast_started = time.perf_counter()
    with NUMAInferenceSession(settings) as runner:
        for step_index in range(steps):
            lead_time = step_index / MAX_FORECAST_STEPS
            runner.timings.clear()
            normalized_plevel, normalized_surface, wall_seconds = runner.forward(
                plevel_history,
                surface_history,
                lead_time,
            )
            plevel_output, surface_output = _denormalize_outputs(
                normalized_plevel,
                normalized_surface,
                constants,
            )
            step = step_index + 1
            timestamp = _forecast_timestamp(initial_timestamp, step)
            _save_step(
                output_directory,
                timestamp,
                step,
                lead_time,
                wall_seconds,
                plevel_output,
                surface_output,
                settings.precision,
            )
            plevel_history, surface_history = _advance_inputs(
                plevel_history,
                surface_history,
                plevel_output,
                surface_output,
                constants,
            )
            forward_walls.append(wall_seconds)
            print(
                f"FORECAST step={step}/{steps} timestamp={timestamp} "
                f"lead_time={lead_time:.9f} wall_seconds={wall_seconds:.6f}",
                flush=True,
            )

    elapsed_seconds = time.perf_counter() - forecast_started
    manifest = {
        "initial_timestamp": initial_timestamp,
        "steps": steps,
        "interval_hours": INPUT_INTERVAL_HOURS,
        "final_timestamp": _forecast_timestamp(initial_timestamp, steps),
        "forward_wall_seconds": forward_walls,
        "mean_forward_wall_seconds": sum(forward_walls) / len(forward_walls),
        "elapsed_seconds": elapsed_seconds,
        "source_commit": os.environ.get("WENTIAN_SOURCE_COMMIT"),
        "checkpoint": str(settings.checkpoint),
        "precision": settings.precision,
        "output_format": f"physical {settings.precision} torch tensors",
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"FORECAST complete steps={steps} final_timestamp={manifest['final_timestamp']} "
        f"mean_forward_wall_seconds={manifest['mean_forward_wall_seconds']:.6f} "
        f"elapsed_seconds={elapsed_seconds:.3f} output={output_directory}",
        flush=True,
    )


if __name__ == "__main__":
    main()
