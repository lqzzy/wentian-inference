"""Tensor operators and spatial decomposition metadata for optimized inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from einops import rearrange

from wentian.model.utils import pad_3d
from wentian.runtime.plan import (
    RuntimePlan,
    StagePlan,
    locality_balanced_owners,
    model_stage_blocks,
)

WINDOW_SIZE = (2, 6, 12)
WINDOW_DEPTH, WINDOW_HEIGHT, WINDOW_WIDTH = WINDOW_SIZE
WINDOW_VOLUME = WINDOW_DEPTH * WINDOW_HEIGHT * WINDOW_WIDTH
SHIFT_SIZE = (1, 3, 6)
PATCH_SIZE = (2, 4, 4)


@dataclass(frozen=True)
class HeightWindowMap:
    indices: list[int]
    has_phantom_window: bool


def _round_up(value: int, unit: int) -> int:
    return ((value + unit - 1) // unit) * unit


def _shifted_depthwise_conv2d(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Evaluate Wentian's 7x7 depthwise convolution as fused spatial shifts."""
    height, width = tensor.shape[2:]
    padded = functional.pad(tensor, (3, 3, 3, 3))
    output = padded[:, :, :height, :width] * kernel[None, :, 0, 0, None, None]
    for row in range(7):
        for column in range(7):
            if row == 0 and column == 0:
                continue
            output.addcmul_(
                padded[:, :, row : row + height, column : column + width],
                kernel[None, :, row, column, None, None],
            )
    return output.add_(bias[None, :, None, None])


_compiled_shifted_depthwise_conv2d = torch.compile(
    _shifted_depthwise_conv2d,
    fullgraph=True,
    dynamic=False,
    mode="reduce-overhead",
)


def _depthwise_conv2d(module: torch.nn.Conv2d, tensor: torch.Tensor) -> torch.Tensor:
    """Run FP64 depthwise convolution without PyTorch's slow grouped-conv fallback."""
    if tensor.dtype != torch.float64:
        return module(tensor)
    if (
        module.groups != tensor.shape[1]
        or module.in_channels != module.out_channels
        or module.in_channels != module.groups
        or module.kernel_size != (7, 7)
        or module.padding != (3, 3)
        or module.stride != (1, 1)
        or module.dilation != (1, 1)
        or module.bias is None
        or module.padding_mode != "zeros"
        or module.weight.dtype != tensor.dtype
        or module.bias.dtype != tensor.dtype
        or module.weight.device != tensor.device
        or module.bias.device != tensor.device
        or tensor.device.type != "cpu"
    ):
        return module(tensor)
    return _compiled_shifted_depthwise_conv2d(tensor, module.weight[:, 0], module.bias)


def block_conv(
    block: torch.nn.Module,
    tensor: torch.Tensor,
    lead_time: torch.Tensor,
    padded_input: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply lead-time embedding, depthwise convolution, and normalization.

    A persistent padded input avoids allocating and copying a separate ``x + t``
    tensor on every block. Its padding remains zero and its interior is fully
    overwritten before each use.
    """
    lead_time_embedding = block.lead_time_embedding(lead_time)
    batch, depth, height, width, channels = tensor.shape
    padded_depth = _round_up(depth, WINDOW_DEPTH)
    padded_height = _round_up(height, WINDOW_HEIGHT)
    padded_width = _round_up(width, WINDOW_WIDTH)
    padded_shape = batch, padded_depth, padded_height, padded_width, channels
    if padded_input is None:
        padded_input = tensor.new_zeros(padded_shape)
    elif padded_input.shape != padded_shape:
        raise ValueError(
            f"padded input has shape {tuple(padded_input.shape)}, expected {padded_shape}"
        )
    torch.add(
        tensor,
        lead_time_embedding,
        out=padded_input[:, :depth, :height, :width, :],
    )

    conv_input = padded_input.view(
        batch * padded_depth,
        padded_height,
        padded_width,
        channels,
    ).permute(0, 3, 1, 2)
    post_conv = _depthwise_conv2d(block.large_kernel_conv, conv_input) + conv_input
    post_conv = post_conv.permute(0, 2, 3, 1).reshape(
        batch,
        padded_depth,
        padded_height,
        padded_width,
        channels,
    )
    return block.norm_conv(post_conv)


def window_phase(
    block: torch.nn.Module,
    post_conv: torch.Tensor,
    shortcut_source: torch.Tensor,
    owner_cache: Mapping[str, object],
    output: torch.Tensor,
) -> None:
    """Compute this tile's owned windows and scatter each output exactly once."""
    window_indices = owner_cache["window_indices"]
    if window_indices.numel() == 0:
        return

    padded_height = post_conv.shape[2]
    width = post_conv.shape[3]
    channels = post_conv.shape[4]
    sequence_length = owner_cache["sequence_length"]
    depth_indices = owner_cache["depth_indices"]
    row_indices = owner_cache["row_indices"]
    column_indices = owner_cache["column_indices"]
    valid_positions = owner_cache["valid_positions"]
    flat_indices = (depth_indices * padded_height + row_indices) * width + column_indices
    windows = (
        post_conv[0]
        .reshape(-1, channels)
        .index_select(0, flat_indices)
        .reshape(window_indices.numel(), sequence_length, channels)
    )

    attention = block.attention
    packed_qkv = attention.linear1(windows)
    window_count, sequence_length, packed_channels = packed_qkv.shape
    attention_channels = packed_channels // 3
    head_count = attention.head_number
    head_channels = attention_channels // head_count
    query = (
        packed_qkv.narrow(-1, 0, attention_channels)
        .view(window_count, sequence_length, head_count, head_channels)
        .transpose(1, 2)
    )
    key = (
        packed_qkv.narrow(-1, attention_channels, attention_channels)
        .view(window_count, sequence_length, head_count, head_channels)
        .transpose(1, 2)
    )
    value = (
        packed_qkv.narrow(-1, 2 * attention_channels, attention_channels)
        .view(window_count, sequence_length, head_count, head_channels)
        .transpose(1, 2)
    )
    attended = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=owner_cache["attention_bias"],
        scale=attention.scale,
    )
    attended = rearrange(attended, "b h s d -> b s (h d)")
    attended = attention.linear2(attended).reshape(-1, channels)

    owned_depth = depth_indices[valid_positions]
    owned_rows = row_indices[valid_positions]
    owned_columns = column_indices[valid_positions]
    block_output = attended[valid_positions]
    shortcut = shortcut_source[0][owned_depth, owned_rows, owned_columns, :]
    block_output = shortcut + block.norm1(block_output)
    block_output = block_output + block.norm2(block.linear(block_output))
    output[0][owned_depth, owned_rows, owned_columns, :] = block_output


def matmul_output(
    output_layer: torch.nn.Module,
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace non-overlapping transposed convolutions with matrix products."""
    patch_depth, patch_height, patch_width = output_layer.patch_size
    plevel_weight = output_layer.conv.weight
    plevel_channels = plevel_weight.shape[1]
    plevel = features[:, :-1] @ plevel_weight.reshape(plevel_weight.shape[0], -1)
    plevel = rearrange(
        plevel,
        "b z h w (c pd ph pw) -> b c (z pd) (h ph) (w pw)",
        c=plevel_channels,
        pd=patch_depth,
        ph=patch_height,
        pw=patch_width,
    ) + output_layer.conv.bias.reshape(1, plevel_channels, 1, 1, 1)
    plevel = plevel[
        :,
        :,
        : output_layer.output_size[0],
        : output_layer.output_size[1],
        : output_layer.output_size[2],
    ]

    surface_weight = output_layer.conv_surface.weight
    surface_channels = surface_weight.shape[1]
    surface = features[:, -1] @ surface_weight.reshape(surface_weight.shape[0], -1)
    surface = rearrange(
        surface,
        "b h w (c ph pw) -> b c (h ph) (w pw)",
        c=surface_channels,
        ph=patch_height,
        pw=patch_width,
    ) + output_layer.conv_surface.bias.reshape(1, surface_channels, 1, 1)
    surface = surface[
        :,
        :,
        : output_layer.output_size[1],
        : output_layer.output_size[2],
    ]
    return plevel, surface


def matmul_input(
    input_layer: torch.nn.Module,
    plevel: torch.Tensor,
    surface: torch.Tensor,
) -> torch.Tensor:
    """Replace non-overlapping patch convolutions with matrix products."""
    patch_depth, patch_height, patch_width = input_layer.patch_size
    plevel_patches = rearrange(
        plevel,
        "b c (z pd) (h ph) (w pw) -> b z h w (c pd ph pw)",
        pd=patch_depth,
        ph=patch_height,
        pw=patch_width,
    )
    plevel_weight = rearrange(
        input_layer.conv.weight,
        "oc ic pd ph pw -> (ic pd ph pw) oc",
    )
    plevel_embedding = plevel_patches @ plevel_weight + input_layer.conv.bias

    surface_patches = rearrange(
        surface,
        "b c (h ph) (w pw) -> b h w (c ph pw)",
        ph=patch_height,
        pw=patch_width,
    )
    surface_weight = rearrange(
        input_layer.conv_surface.weight,
        "oc ic ph pw -> (ic ph pw) oc",
    )
    surface_embedding = (
        surface_patches @ surface_weight + input_layer.conv_surface.bias
    ).unsqueeze(1)
    return torch.concat([plevel_embedding, surface_embedding], dim=1)


def _height_window_maps(
    block: torch.nn.Module,
    height: int,
    height_window_count: int,
    padded_height: int,
    bounds: Sequence[tuple[int, int]],
    halo_height: int,
) -> list[HeightWindowMap]:
    maps = []
    for partition, (start, stop) in enumerate(bounds):
        halo_start = max(0, start - halo_height)
        halo_stop = min(height, stop + halo_height)
        tile_height = halo_stop - halo_start
        local_window_count = _round_up(tile_height, WINDOW_HEIGHT) // WINDOW_HEIGHT
        if block.roll and partition == 0:
            indices = list(range(local_window_count)) + [height_window_count - 1]
            has_phantom_window = True
        elif block.roll:
            padded_tile_height = local_window_count * WINDOW_HEIGHT
            indices = [
                (
                    (
                        halo_start
                        + (WINDOW_HEIGHT * local_window + SHIFT_SIZE[1]) % padded_tile_height
                    )
                    % padded_height
                    - SHIFT_SIZE[1]
                )
                % padded_height
                // WINDOW_HEIGHT
                for local_window in range(local_window_count)
            ]
            has_phantom_window = False
        else:
            indices = [
                halo_start // WINDOW_HEIGHT + local_window
                for local_window in range(local_window_count)
            ]
            has_phantom_window = False
        maps.append(HeightWindowMap(indices, has_phantom_window))
    return maps


def _width_window_maps(
    block: torch.nn.Module,
    width: int,
    width_window_count: int,
    bounds: Sequence[tuple[int, int]],
    halo_width: int,
) -> list[list[int]]:
    maps = []
    for start, stop in bounds:
        halo_start = start - halo_width
        halo_stop = stop + halo_width
        local_window_count = _round_up(halo_stop - halo_start, WINDOW_WIDTH) // WINDOW_WIDTH
        if block.roll:
            padded_tile_width = local_window_count * WINDOW_WIDTH
            indices = [
                (
                    (halo_start + (WINDOW_WIDTH * local_window + SHIFT_SIZE[2]) % padded_tile_width)
                    % width
                    - SHIFT_SIZE[2]
                )
                % width
                // WINDOW_WIDTH
                for local_window in range(local_window_count)
            ]
        else:
            indices = [
                (halo_start // WINDOW_WIDTH + local_window) % width_window_count
                for local_window in range(local_window_count)
            ]
        maps.append(indices)
    return maps


def _position_bias(block: torch.nn.Module) -> torch.Tensor:
    attention = block.attention
    return rearrange(
        attention.abs_position_embedding[attention.position_index],
        "(q k) t h -> t h q k",
        q=WINDOW_VOLUME,
        k=WINDOW_VOLUME,
    )


def _owner_sidecar(
    block: torch.nn.Module,
    attention_bias: torch.Tensor,
    owned: Sequence[bool],
    height_windows: Sequence[int],
    width_windows: Sequence[int],
    depth_window_count: int,
    padded_height: int,
    height: int,
    width: int,
) -> dict[str, object]:
    window_indices = torch.tensor(owned, dtype=torch.bool).nonzero(as_tuple=True)[0]
    local_height_windows = len(height_windows)
    local_width_windows = len(width_windows)
    depth_window = window_indices // (local_height_windows * local_width_windows)
    local_height_window = (window_indices // local_width_windows) % local_height_windows
    local_width_window = window_indices % local_width_windows
    global_height_window = torch.tensor(height_windows).index_select(0, local_height_window)
    global_width_window = torch.tensor(width_windows).index_select(0, local_width_window)

    depth_shift, height_shift, width_shift = SHIFT_SIZE if block.roll else (0, 0, 0)
    depth_offsets = torch.arange(WINDOW_DEPTH)
    row_offsets = torch.arange(WINDOW_HEIGHT)
    column_offsets = torch.arange(WINDOW_WIDTH)
    depth_indices = (
        depth_window[:, None, None, None] * WINDOW_DEPTH
        + depth_offsets[None, :, None, None]
        + depth_shift
    ) % (depth_window_count * WINDOW_DEPTH)
    row_indices = (
        global_height_window[:, None, None, None] * WINDOW_HEIGHT
        + row_offsets[None, None, :, None]
        + height_shift
    ) % padded_height
    column_indices = (
        global_width_window[:, None, None, None] * WINDOW_WIDTH
        + column_offsets[None, None, None, :]
        + width_shift
    ) % width
    depth_indices, row_indices, column_indices = torch.broadcast_tensors(
        depth_indices,
        row_indices,
        column_indices,
    )
    depth_indices = depth_indices.reshape(-1).contiguous()
    row_indices = row_indices.reshape(-1).contiguous()
    column_indices = column_indices.reshape(-1).contiguous()
    return {
        "window_indices": window_indices,
        "attention_bias": attention_bias.index_select(0, window_indices),
        "depth_indices": depth_indices,
        "row_indices": row_indices,
        "column_indices": column_indices,
        "valid_positions": row_indices < height,
        "sequence_length": WINDOW_VOLUME,
    }


@torch.no_grad()
def build_cache(model: torch.nn.Module, plan: RuntimePlan) -> dict:
    """Build immutable owner maps and attention masks outside the hot path."""
    cache = {}
    blocks_by_stage = model_stage_blocks(model)
    for stage in plan.stages:
        _, depth, height, width, _ = stage.shape
        depth_window_count = depth // WINDOW_DEPTH
        height_window_count = _round_up(height, WINDOW_HEIGHT) // WINDOW_HEIGHT
        width_window_count = width // WINDOW_WIDTH
        padded_height = height_window_count * WINDOW_HEIGHT
        bias_type_count = depth_window_count * height_window_count

        for block_index, block in enumerate(blocks_by_stage[stage.name]):
            block.large_kernel_conv.to(memory_format=torch.channels_last)
            base_bias = _position_bias(block)
            height_maps = _height_window_maps(
                block,
                height,
                height_window_count,
                padded_height,
                stage.height_bounds,
                plan.topology.halo_height,
            )
            width_maps = _width_window_maps(
                block,
                width,
                width_window_count,
                stage.width_bounds,
                plan.topology.halo_width,
            )
            real_height_maps = [
                item.indices[:-1] if item.has_phantom_window else item.indices
                for item in height_maps
            ]
            height_owners = locality_balanced_owners(
                real_height_maps,
                WINDOW_HEIGHT,
                SHIFT_SIZE[1] if block.roll else 0,
                padded_height,
                stage.height_bounds,
            )
            width_owners = locality_balanced_owners(
                width_maps,
                WINDOW_WIDTH,
                SHIFT_SIZE[2] if block.roll else 0,
                width,
                stage.width_bounds,
            )

            for height_partition, height_map in enumerate(height_maps):
                for width_partition, width_map in enumerate(width_maps):
                    bias_indices = []
                    mask_indices = []
                    owned = []
                    for depth_window in range(depth_window_count):
                        for local_height, global_height in enumerate(height_map.indices):
                            real_height = not (
                                height_map.has_phantom_window
                                and local_height == len(height_map.indices) - 1
                            )
                            owns_height = (
                                real_height and height_owners.get(global_height) == height_partition
                            )
                            for global_width in width_map:
                                global_index = (
                                    depth_window * height_window_count + global_height
                                ) * width_window_count + global_width
                                bias_indices.append(global_index % bias_type_count)
                                mask_indices.append(global_index)
                                owned.append(
                                    owns_height
                                    and width_owners.get(global_width) == width_partition
                                )

                    attention_bias = base_bias[torch.tensor(bias_indices)]
                    if block.roll:
                        shift_mask = block.attn_mask[torch.tensor(mask_indices)]
                        attention_bias = (attention_bias + shift_mask[:, None, :, :]).contiguous()
                    key = (
                        "own",
                        stage.name,
                        block_index,
                        height_partition,
                        width_partition,
                    )
                    cache[key] = _owner_sidecar(
                        block,
                        attention_bias,
                        owned,
                        height_map.indices,
                        width_map,
                        depth_window_count,
                        padded_height,
                        height,
                        width,
                    )
    return cache


def run_stage(
    stage: StagePlan,
    blocks: Sequence[torch.nn.Module],
    tensor: torch.Tensor,
    cache: Mapping,
    lead_time: torch.Tensor,
) -> torch.Tensor:
    """Run a stage in one process using the same owner sidecars as workers."""
    for block_index, block in enumerate(blocks):
        shortcut = pad_3d(tensor, WINDOW_SIZE)
        post_conv = block_conv(block, tensor, lead_time)
        output = torch.zeros_like(tensor)
        for height_partition in range(len(stage.height_bounds)):
            for width_partition in range(len(stage.width_bounds)):
                key = (
                    "own",
                    stage.name,
                    block_index,
                    height_partition,
                    width_partition,
                )
                window_phase(block, post_conv, shortcut, cache[key], output)
        tensor = output
    return tensor
