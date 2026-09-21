"""NUMA-aware 2D Wentian inference executor for Kunpeng 920F systems.

Each worker owns one spatial tile. A block runs in two synchronized phases:
convolution writes tile interiors to a shared frame, then window attention reads
the complete frame and writes each owned window to the next shared buffer.
"""

from __future__ import annotations

import copy
import gc
import os
import sys
import time
import traceback
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp
from einops import rearrange

from wentian.data import load_raw_inputs
from wentian.model.loading import load_wentian_model
from wentian.model.utils import crop_3d, pad_2d, pad_3d
from wentian.precision import precision_dtype
from wentian.reference import standard_forward, verify_outputs
from wentian.runtime import operators
from wentian.runtime.numa import configure_hbm
from wentian.runtime.plan import RuntimePlan, StagePlan, model_stage_blocks
from wentian.runtime.topology import CPUAssignment, RuntimeTopology
from wentian.runtime.workers import WorkerGroup
from wentian.settings import HBMSettings, RuntimeSettings

PLEVEL_SHAPE = (2, 5, 13, 721, 1440)
SURFACE_SHAPE = (2, 7, 721, 1440)


@dataclass(frozen=True)
class SharedStageBuffers:
    first: dict[str, torch.Tensor]
    second: dict[str, torch.Tensor]
    post_conv: dict[str, torch.Tensor]


@dataclass(frozen=True)
class StageWorkBuffers:
    post_conv: torch.Tensor
    padded_conv_input: torch.Tensor


@dataclass(frozen=True)
class WorkerContext:
    model: torch.nn.Module
    cache: dict
    plan: RuntimePlan
    buffers: SharedStageBuffers
    control: torch.Tensor
    block_barrier: Any
    io_barrier: Any
    copy_model: bool
    lead_time: torch.Tensor
    dtype: torch.dtype
    hbm_settings: HBMSettings


@contextmanager
def _torch_thread_count(count: int) -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(count)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _load_tensor(path: str, expected_dims: int, name: str) -> torch.Tensor:
    # User-provided tensors are trusted inputs and may contain more than weights.
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, dict):
        value = next((value[key] for key in (name, "data", "tensor") if key in value), value)
    if not torch.is_tensor(value):
        raise TypeError(f"{name} input must contain a torch.Tensor")
    if value.ndim == expected_dims + 1 and value.shape[0] == 1:
        value = value.squeeze(0)
    if value.ndim != expected_dims:
        raise ValueError(
            f"{name} input has shape {tuple(value.shape)}; expected {expected_dims} dimensions"
        )
    return value.detach().to(device="cpu").contiguous()


def _raw_timestamp(timestamp: str, hours: int) -> str:
    parsed = datetime.strptime(timestamp, "%Y%m%d%H")
    return (parsed + timedelta(hours=hours)).strftime("%Y%m%d%H")


def _load_inputs(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    plevel_path = os.environ.get("WENTIAN_PLEVEL_INPUT")
    surface_path = os.environ.get("WENTIAN_SURFACE_INPUT")
    input_root = os.environ.get("WENTIAN_INPUT_ROOT")
    timestamp = os.environ.get("WENTIAN_TIMESTAMP")

    if bool(plevel_path) != bool(surface_path):
        raise ValueError("WENTIAN_PLEVEL_INPUT and WENTIAN_SURFACE_INPUT must be set together")
    if bool(input_root) != bool(timestamp):
        raise ValueError("WENTIAN_INPUT_ROOT and WENTIAN_TIMESTAMP must be set together")
    if plevel_path and input_root:
        raise ValueError("prepared tensor inputs and raw timestamp inputs are mutually exclusive")

    if plevel_path:
        plevel = _load_tensor(plevel_path, expected_dims=5, name="plevel")
        surface = _load_tensor(surface_path, expected_dims=4, name="surface")
    elif input_root:
        plevel, surface = load_raw_inputs(
            input_root,
            timestamp,
            os.environ.get("WENTIAN_CONSTANTS"),
        )
        print(
            f"INPUT raw_root={input_root} timestamps={_raw_timestamp(timestamp, -6)},{timestamp}",
            flush=True,
        )
    else:
        random = np.random.RandomState(int(os.environ.get("WENTIAN_SEED", "0")))
        plevel = torch.from_numpy((random.randn(*PLEVEL_SHAPE) * 0.3).astype(np.float32))
        surface = torch.from_numpy((random.randn(*SURFACE_SHAPE) * 0.3).astype(np.float32))

    plevel = plevel.to(dtype=dtype)
    surface = surface.to(dtype=dtype)

    if tuple(plevel.shape) != PLEVEL_SHAPE:
        raise ValueError(f"plevel input shape must be {PLEVEL_SHAPE}, got {tuple(plevel.shape)}")
    if tuple(surface.shape) != SURFACE_SHAPE:
        raise ValueError(f"surface input shape must be {SURFACE_SHAPE}, got {tuple(surface.shape)}")
    return plevel, surface


def _round_up(value: int, unit: int) -> int:
    return ((value + unit - 1) // unit) * unit


def _padded_height(height: int) -> int:
    return _round_up(height, operators.WINDOW_HEIGHT)


def _stage_shape(stage: StagePlan) -> tuple[int, ...]:
    batch, depth, height, width, channels = stage.shape
    return batch, depth, _padded_height(height), width, channels


def _pad_height(tensor: torch.Tensor, padded_height: int) -> torch.Tensor:
    batch, depth, height, width, channels = tensor.shape
    if height >= padded_height:
        return tensor[:, :, :padded_height]
    padded = tensor.new_zeros(batch, depth, padded_height, width, channels)
    padded[:, :, :height] = tensor
    return padded


def _first_touch_shared_buffers(
    buffers: SharedStageBuffers,
    plan: RuntimePlan,
    height_partition: int,
    width_partition: int,
) -> None:
    """Place each tile's shared pages on its worker's NUMA node."""
    for stage in plan.stages:
        padded_height = buffers.first[stage.name].shape[2]
        row_start, row_stop = stage.height_bounds[height_partition]
        col_start, col_stop = stage.width_bounds[width_partition]
        if height_partition == plan.topology.height_partitions - 1:
            row_stop = padded_height
        region = (
            slice(None),
            slice(None),
            slice(row_start, row_stop),
            slice(col_start, col_stop),
            slice(None),
        )
        buffers.first[stage.name][region].zero_()
        buffers.second[stage.name][region].zero_()
        buffers.post_conv[stage.name][region].zero_()


def _allocate_work_buffers(
    plan: RuntimePlan,
    height_partition: int,
    width_partition: int,
    dtype: torch.dtype,
) -> dict[str, StageWorkBuffers]:
    """Allocate reusable worker-private tensors under the active HBM policy."""
    work_buffers = {}
    with _torch_thread_count(1):
        for stage in plan.stages:
            batch, depth, height, width, channels = stage.shape
            row_start, row_stop = stage.height_bounds[height_partition]
            col_start, col_stop = stage.width_bounds[width_partition]
            halo_row_start = max(0, row_start - plan.topology.halo_height)
            halo_row_stop = min(height, row_stop + plan.topology.halo_height)
            halo_col_start = max(0, col_start - plan.topology.halo_width)
            halo_col_stop = min(width, col_stop + plan.topology.halo_width)
            shape = (
                batch,
                _round_up(depth, operators.WINDOW_SIZE[0]),
                _round_up(halo_row_stop - halo_row_start, operators.WINDOW_SIZE[1]),
                _round_up(halo_col_stop - halo_col_start, operators.WINDOW_SIZE[2]),
                channels,
            )
            work_buffers[stage.name] = StageWorkBuffers(
                post_conv=torch.zeros(shape, dtype=dtype),
                padded_conv_input=torch.zeros(shape, dtype=dtype),
            )
    return work_buffers


def _tensor_bytes(value: object) -> int:
    if not torch.is_tensor(value):
        return 0
    return value.numel() * value.element_size()


def _clone_worker_cache(
    cache: Mapping,
    plan: RuntimePlan,
    stage_blocks: Mapping[str, Sequence[torch.nn.Module]],
    height_partition: int,
    width_partition: int,
    budget_mb: float,
) -> dict:
    """Clone hot owner sidecars after the worker's HBM policy is active."""
    budget_bytes = int(budget_mb * 1024 * 1024)
    if budget_bytes <= 0:
        return {}

    cloned = {}
    used_bytes = 0
    with _torch_thread_count(1):
        for stage in plan.stages:
            for block_index in range(len(stage_blocks[stage.name])):
                if used_bytes >= budget_bytes:
                    return cloned
                key = (
                    "own",
                    stage.name,
                    block_index,
                    height_partition,
                    width_partition,
                )
                source = cache[key]
                owner_cache = {
                    item_name: item.clone() if torch.is_tensor(item) else item
                    for item_name, item in source.items()
                }
                cloned[key] = owner_cache
                used_bytes += sum(_tensor_bytes(item) for item in owner_cache.values())
    return cloned


def _run_worker_stage(
    stage: StagePlan,
    blocks: Sequence[torch.nn.Module],
    context: WorkerContext,
    work_buffers: Mapping[str, StageWorkBuffers],
    local_cache: Mapping,
    height_partition: int,
    width_partition: int,
) -> int:
    name = stage.name
    buffers = context.buffers
    height = stage.shape[2]
    padded_height = buffers.first[name].shape[2]
    width = buffers.first[name].shape[3]
    row_start, row_stop = stage.height_bounds[height_partition]
    col_start, col_stop = stage.width_bounds[width_partition]
    output_row_stop = (
        row_stop
        if height_partition < context.plan.topology.height_partitions - 1
        else padded_height
    )
    current_buffer = 0

    for block_index, block in enumerate(blocks):
        input_tensor = buffers.first[name] if current_buffer == 0 else buffers.second[name]
        output_tensor = buffers.second[name] if current_buffer == 0 else buffers.first[name]
        halo_row_start = max(0, row_start - context.plan.topology.halo_height)
        halo_row_stop = min(height, row_stop + context.plan.topology.halo_height)
        halo_col_start = max(0, col_start - context.plan.topology.halo_width)
        halo_col_stop = min(width, col_stop + context.plan.topology.halo_width)
        tile_input = input_tensor[
            :, :, halo_row_start:halo_row_stop, halo_col_start:halo_col_stop, :
        ]

        persistent_input = work_buffers[name].padded_conv_input if name in work_buffers else None
        post_conv = operators.block_conv(
            block,
            tile_input,
            context.lead_time,
            persistent_input,
        )
        if name in work_buffers:
            local_post_conv = work_buffers[name].post_conv
            local_post_conv.copy_(post_conv)
        else:
            local_post_conv = post_conv

        buffers.post_conv[name][:, :, row_start:output_row_stop, col_start:col_stop, :] = (
            local_post_conv[
                :,
                :,
                row_start - halo_row_start : output_row_stop - halo_row_start,
                col_start - halo_col_start : col_stop - halo_col_start,
                :,
            ]
        )
        context.block_barrier.wait()

        owner_key = ("own", name, block_index, height_partition, width_partition)
        owner_cache = local_cache.get(owner_key, context.cache[owner_key])
        operators.window_phase(
            block,
            buffers.post_conv[name],
            input_tensor,
            owner_cache,
            output_tensor,
        )
        context.block_barrier.wait()
        current_buffer = 1 - current_buffer

    return current_buffer


def _worker_main(
    worker_id: int,
    assignment: CPUAssignment,
    context: WorkerContext,
) -> None:
    try:
        os.sched_setaffinity(0, set(assignment.cpus))
        torch.set_num_threads(len(assignment.cpus))
        height_partition, width_partition = divmod(
            worker_id,
            context.plan.topology.width_partitions,
        )
        _first_touch_shared_buffers(
            context.buffers,
            context.plan,
            height_partition,
            width_partition,
        )

        hbm_placement = configure_hbm(worker_id, assignment.domain, context.hbm_settings)
        model = copy.deepcopy(context.model) if context.copy_model else context.model
        stage_blocks = model_stage_blocks(model)
        work_buffers = (
            _allocate_work_buffers(
                context.plan,
                height_partition,
                width_partition,
                context.dtype,
            )
            if context.hbm_settings.use_buffers
            else {}
        )
        local_cache = (
            _clone_worker_cache(
                context.cache,
                context.plan,
                stage_blocks,
                height_partition,
                width_partition,
                context.hbm_settings.cache_mb,
            )
            if context.hbm_settings.use_cache
            else {}
        )

        context.io_barrier.wait()
        while True:
            context.io_barrier.wait()
            if int(context.control[0]) == 0:
                break
            stage = context.plan.stages[int(context.control[1])]
            with torch.no_grad():
                context.control[2] = _run_worker_stage(
                    stage,
                    stage_blocks[stage.name],
                    context,
                    work_buffers,
                    local_cache,
                    height_partition,
                    width_partition,
                )
            context.io_barrier.wait()
        hbm_placement.report()
    except Exception:
        error = traceback.format_exc()
        try:
            Path.cwd().joinpath(f"_w2err_{worker_id}.log").write_text(error)
        except OSError:
            sys.stderr.write(error)
            sys.stderr.flush()
        for barrier in (context.block_barrier, context.io_barrier):
            with suppress(Exception):
                barrier.abort()
        raise


def _create_shared_buffers(plan: RuntimePlan, dtype: torch.dtype) -> SharedStageBuffers:
    buffers = SharedStageBuffers(first={}, second={}, post_conv={})
    for stage in plan.stages:
        shape = _stage_shape(stage)
        buffers.first[stage.name] = torch.zeros(shape, dtype=dtype).share_memory_()
        buffers.second[stage.name] = torch.zeros(shape, dtype=dtype).share_memory_()
        buffers.post_conv[stage.name] = torch.zeros(shape, dtype=dtype).share_memory_()
    return buffers


def _worker_cpu_assignments(topology: RuntimeTopology) -> tuple[CPUAssignment, ...]:
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is None:
        cpu_count = os.cpu_count()
        if cpu_count is None:
            raise RuntimeError("could not determine the available CPU count")
        available_cpus = set(range(cpu_count))
    else:
        available_cpus = set(affinity(0))
    return topology.cpu_assignments(available_cpus)


class ParallelInference:
    """Run one optimized forward pass using an initialized worker group."""

    def __init__(
        self,
        model: torch.nn.Module,
        cache: dict,
        plan: RuntimePlan,
        buffers: SharedStageBuffers,
        control: torch.Tensor,
        workers: WorkerGroup,
        lead_time: torch.Tensor,
        dtype: torch.dtype,
    ) -> None:
        self.model = model
        self.cache = cache
        self.plan = plan
        self.buffers = buffers
        self.control = control
        self.workers = workers
        self.lead_time = lead_time
        self.dtype = dtype
        self.timings: defaultdict[str, float] = defaultdict(float)
        self.debug = os.environ.get("WT_DBG") == "1"

    def run_stage(self, name: str, input_tensor: torch.Tensor) -> torch.Tensor:
        copy_started = time.perf_counter()
        self.buffers.first[name].copy_(input_tensor)
        self.buffers.second[name].zero_()
        copy_seconds = time.perf_counter() - copy_started

        self.control[0] = 1
        self.control[1] = self.plan.stage_index(name)
        worker_started = time.perf_counter()
        self.workers.wait()
        self.workers.wait()
        worker_seconds = time.perf_counter() - worker_started

        output = (
            self.buffers.first[name] if int(self.control[2]) == 0 else self.buffers.second[name]
        )
        clone_started = time.perf_counter()
        result = output.clone()
        clone_seconds = time.perf_counter() - clone_started
        self.timings[f"stage_{name}"] += worker_seconds
        self.timings["copyclone"] += copy_seconds + clone_seconds
        print(
            f"  stage {name}: worker={worker_seconds:.2f}s "
            f"copy_={copy_seconds:.2f}s clone={clone_seconds:.2f}s",
            flush=True,
        )
        return result

    def _debug_stage(
        self,
        name: str,
        blocks: Sequence[torch.nn.Module],
        original: torch.Tensor,
        optimized: torch.Tensor,
        shape: tuple[int, int, int],
    ) -> None:
        if not self.debug:
            return
        expected = operators.run_stage(
            self.plan.stage(name),
            blocks,
            original,
            self.cache,
            self.lead_time,
        )
        max_diff = (crop_3d(optimized, shape) - expected).abs().max().item()
        print(f"{name} diff max={max_diff:.5g}", flush=True)

    def forward(
        self,
        plevel: torch.Tensor,
        surface: torch.Tensor,
        lead_time: float,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        if tuple(plevel.shape) != PLEVEL_SHAPE:
            raise ValueError(
                f"plevel input shape must be {PLEVEL_SHAPE}, got {tuple(plevel.shape)}"
            )
        if tuple(surface.shape) != SURFACE_SHAPE:
            raise ValueError(
                f"surface input shape must be {SURFACE_SHAPE}, got {tuple(surface.shape)}"
            )
        self.lead_time.fill_(lead_time)

        started = time.perf_counter()
        with torch.no_grad():
            io_started = time.perf_counter()
            plevel = plevel.to(dtype=self.dtype)
            surface = surface.to(dtype=self.dtype)
            padded_plevel = pad_3d(plevel, operators.PATCH_SIZE, channel_last=False)
            padded_surface = pad_2d(
                surface,
                operators.PATCH_SIZE[1:],
                channel_last=False,
            )
            features = operators.matmul_input(
                self.model.input_layer,
                padded_plevel,
                padded_surface,
            )
            features = rearrange(
                features,
                "(b t) z h w c -> b z h w (t c)",
                t=2,
            )
            features = self.model.seq_len_linear(features)
            self.timings["io_input"] += time.perf_counter() - io_started

            down0_input = features.clone()
            features = self.run_stage("d0", _pad_height(features, _padded_height(181)))
            self._debug_stage(
                "d0",
                self.model.down_blocks[0].blocks,
                down0_input,
                features,
                (8, 181, 360),
            )
            skip = crop_3d(features, (8, 181, 360)).clone()

            io_started = time.perf_counter()
            down1_input = self.model.downsamples[0](crop_3d(features, (8, 181, 360)))
            self.timings["io_down"] += time.perf_counter() - io_started
            features = self.run_stage(
                "d1",
                _pad_height(down1_input, _padded_height(91)),
            )
            self._debug_stage(
                "d1",
                self.model.down_blocks[1].blocks,
                down1_input,
                features,
                (8, 91, 180),
            )

            features = self.run_stage("u0", features)
            io_started = time.perf_counter()
            features = self.model.upsamples[0](crop_3d(features, (8, 91, 180)))
            self.timings["io_up"] += time.perf_counter() - io_started
            features = self.run_stage("u1", _pad_height(features, _padded_height(182)))
            features = crop_3d(features, (8, 182, 360))
            features = crop_3d(features, skip.shape[1:-1])

            io_started = time.perf_counter()
            plevel_output, surface_output = operators.matmul_output(
                self.model.output_layer,
                torch.concat([skip, features], dim=-1),
            )
            self.timings["io_output"] += time.perf_counter() - io_started

        return plevel_output, surface_output, time.perf_counter() - started


class NUMAInferenceSession:
    """Own a persistent worker group for one or more Wentian forwards."""

    def __init__(self, settings: RuntimeSettings) -> None:
        torch.set_num_threads(1)
        self.dtype = precision_dtype(settings.precision)
        self.plan = RuntimePlan.from_topology(settings.topology)
        self.model = load_wentian_model(settings.checkpoint).to(dtype=self.dtype)
        self.cache = operators.build_cache(self.model, self.plan)
        self.buffers = _create_shared_buffers(self.plan, self.dtype)
        self.control = torch.zeros(3, dtype=torch.int).share_memory_()
        self.lead_time = torch.zeros(1, dtype=self.dtype).share_memory_()
        self.block_barrier = mp.Barrier(
            settings.topology.worker_count,
            timeout=settings.worker_timeout_seconds,
        )
        self.io_barrier = mp.Barrier(
            settings.topology.worker_count + 1,
            timeout=settings.worker_timeout_seconds,
        )
        worker_context = WorkerContext(
            model=self.model,
            cache=self.cache,
            plan=self.plan,
            buffers=self.buffers,
            control=self.control,
            block_barrier=self.block_barrier,
            io_barrier=self.io_barrier,
            copy_model=settings.copy_model_per_worker,
            lead_time=self.lead_time,
            dtype=self.dtype,
            hbm_settings=settings.hbm,
        )
        processes = [
            mp.Process(target=_worker_main, args=(worker_id, assignment, worker_context))
            for worker_id, assignment in enumerate(_worker_cpu_assignments(settings.topology))
        ]
        self.workers = WorkerGroup(
            processes,
            self.control,
            self.block_barrier,
            self.io_barrier,
            settings.worker_timeout_seconds,
        )

    def __enter__(self) -> ParallelInference:
        self.workers.__enter__()
        try:
            self.workers.wait()
        except BaseException:
            self.workers.__exit__(*sys.exc_info())
            raise
        torch.set_num_threads(self.plan.topology.main_threads)
        return ParallelInference(
            self.model,
            self.cache,
            self.plan,
            self.buffers,
            self.control,
            self.workers,
            self.lead_time,
            self.dtype,
        )

    def __exit__(self, exc_type, exc_value, traceback_value) -> bool:
        return self.workers.__exit__(exc_type, exc_value, traceback_value)


def _run_samples(
    runner: ParallelInference,
    settings: RuntimeSettings,
    plevel: torch.Tensor,
    surface: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[float], list[dict[str, float]]]:
    walls = []
    timing_samples = []
    plevel_output = surface_output = None
    topology = runner.plan.topology
    for step in range(settings.warmups + settings.repeats):
        runner.timings.clear()
        plevel_output, surface_output, wall_seconds = runner.forward(
            plevel,
            surface,
            settings.lead_time,
        )
        if step < settings.warmups:
            print(
                f"PERF warmup={step + 1} wall_seconds={wall_seconds:.6f} "
                f"precision={settings.precision} "
                f"PH={topology.height_partitions} PW={topology.width_partitions} "
                f"workers={topology.worker_count}",
                flush=True,
            )
            continue
        walls.append(wall_seconds)
        timing_samples.append(dict(runner.timings))
        print(
            f"PERF step={step - settings.warmups + 1} wall_seconds={wall_seconds:.6f} "
            f"precision={settings.precision} "
            f"PH={topology.height_partitions} PW={topology.width_partitions} "
            f"workers={topology.worker_count}",
            flush=True,
        )
    return plevel_output, surface_output, walls, timing_samples


def _verify(
    settings: RuntimeSettings,
    plevel_output: torch.Tensor,
    surface_output: torch.Tensor,
    plevel_input: torch.Tensor,
    surface_input: torch.Tensor,
    lead_time: torch.Tensor,
) -> tuple[float, float]:
    torch.backends.mkldnn.enabled = False
    reference_plevel, reference_surface = standard_forward(
        settings.checkpoint,
        plevel_input.reshape(1, *PLEVEL_SHAPE),
        surface_input.reshape(1, *SURFACE_SHAPE),
        lead_time,
        precision=settings.precision,
    )
    verification = verify_outputs(
        plevel_output,
        surface_output,
        reference_plevel,
        reference_surface,
        settings.max_rmse_percent,
    )
    print(
        "VERIFY reference=wentian.reference.standard_forward "
        f"precision={settings.precision} "
        f"status=PASS limit={settings.max_rmse_percent:.9f}%",
        flush=True,
    )
    return verification.plevel_rmse_percent, verification.surface_rmse_percent


def _average_timings(samples: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = set().union(*(sample.keys() for sample in samples))
    return {key: sum(sample.get(key, 0.0) for sample in samples) / len(samples) for key in keys}


def _save_output(
    settings: RuntimeSettings,
    plevel_output: torch.Tensor,
    surface_output: torch.Tensor,
    wall_seconds: float,
    walls: list[float],
    plevel_rmse: float,
    surface_rmse: float,
) -> None:
    output = os.environ.get("WENTIAN_OUTPUT")
    if not output:
        return
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "plevel": plevel_output,
        "surface": surface_output,
        "precision": settings.precision,
        "wall_seconds": wall_seconds,
        "wall_seconds_samples": walls,
        "rmse_plevel_percent": plevel_rmse,
        "rmse_surface_percent": surface_rmse,
        "normalized": True,
        "lead_time": settings.lead_time,
        "reference": "wentian.reference.standard_forward" if settings.verify else None,
        "max_rmse_percent": settings.max_rmse_percent if settings.verify else None,
    }
    if os.environ.get("WENTIAN_TIMESTAMP"):
        result["input_timestamp"] = os.environ["WENTIAN_TIMESTAMP"]
    torch.save(result, output_path)


def _print_summary(
    wall_seconds: float,
    average: Mapping[str, float],
    precision: str,
    plevel_rmse: float,
    surface_rmse: float,
    verified: bool,
    topology: RuntimeTopology,
) -> None:
    if verified:
        result = f"plevel RMSE={plevel_rmse:.6f}% surface={surface_rmse:.6f}%"
    else:
        result = "verification=SKIPPED"
    print(
        f"2D多进程 precision={precision} PH={topology.height_partitions} "
        f"PW={topology.width_partitions} P={topology.worker_count}  "
        f"{result}  墙钟={wall_seconds:.2f}s",
        flush=True,
    )

    stage_seconds = sum(value for key, value in average.items() if key.startswith("stage_"))
    print(
        f"  分解: stages合计={stage_seconds:.2f}s "
        f"(d0={average['stage_d0']:.2f} d1={average['stage_d1']:.2f} "
        f"u0={average['stage_u0']:.2f} u1={average['stage_u1']:.2f}), "
        f"io(main)={wall_seconds - stage_seconds:.2f}s",
        flush=True,
    )
    print(
        f"  io细分: input={average['io_input']:.2f} down={average['io_down']:.2f} "
        f"up={average['io_up']:.2f} output={average['io_output']:.2f} "
        f"copyclone={average['copyclone']:.2f}",
        flush=True,
    )


def main() -> None:
    mp.set_start_method("fork")
    torch.set_num_threads(1)
    settings = RuntimeSettings.from_env()
    plevel, surface = _load_inputs(precision_dtype(settings.precision))
    lead_time = torch.tensor([settings.lead_time], dtype=plevel.dtype)
    with NUMAInferenceSession(settings) as runner:
        plevel_output, surface_output, walls, timing_samples = _run_samples(
            runner,
            settings,
            plevel,
            surface,
        )

    wall_seconds = sum(walls) / len(walls)
    print(
        f"PERF mean_wall_seconds={wall_seconds:.6f} "
        f"precision={settings.precision} "
        f"samples={','.join(f'{sample:.6f}' for sample in walls)}",
        flush=True,
    )

    if settings.verify:
        del runner
        gc.collect()
        plevel_rmse, surface_rmse = _verify(
            settings,
            plevel_output,
            surface_output,
            plevel,
            surface,
            lead_time,
        )
    else:
        plevel_rmse = surface_rmse = float("nan")

    _save_output(
        settings,
        plevel_output,
        surface_output,
        wall_seconds,
        walls,
        plevel_rmse,
        surface_rmse,
    )
    _print_summary(
        wall_seconds,
        _average_timings(timing_samples),
        settings.precision,
        plevel_rmse,
        surface_rmse,
        settings.verify,
        settings.topology,
    )


if __name__ == "__main__":
    main()
