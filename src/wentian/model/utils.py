"""Tensor shape helpers used by the canonical Wentian model."""

from __future__ import annotations

import torch
from einops import rearrange
from torch.nn import functional


def _padding(length: int, divisor: int) -> int:
    return (-length) % divisor


def get_padded_shape_3d(
    input_shape: tuple[int, int, int] = (13, 720, 1440),
    division_size: tuple[int, int, int] = (4, 4, 4),
) -> tuple[int, int, int]:
    """Return the smallest divisible 3D shape that contains ``input_shape``."""
    return tuple(
        length + _padding(length, divisor) for length, divisor in zip(input_shape, division_size)
    )


def pad_3d(
    tensor: torch.Tensor,
    division_size: tuple[int, int, int] = (4, 4, 4),
    channel_last: bool = True,
) -> torch.Tensor:
    """Pad depth, height, and width on their right edge with zeros."""
    if channel_last:
        depth, height, width = tensor.shape[1:4]
        padding = (
            0,
            0,
            0,
            _padding(width, division_size[2]),
            0,
            _padding(height, division_size[1]),
            0,
            _padding(depth, division_size[0]),
        )
    else:
        depth, height, width = tensor.shape[2:5]
        padding = (
            0,
            _padding(width, division_size[2]),
            0,
            _padding(height, division_size[1]),
            0,
            _padding(depth, division_size[0]),
        )
    return functional.pad(tensor, padding)


def pad_2d(
    tensor: torch.Tensor,
    division_size: tuple[int, int] = (4, 4),
    channel_last: bool = True,
) -> torch.Tensor:
    """Pad height and width on their right edge with zeros."""
    if channel_last:
        height, width = tensor.shape[1:3]
        padding = (
            0,
            0,
            0,
            _padding(width, division_size[1]),
            0,
            _padding(height, division_size[0]),
        )
    else:
        height, width = tensor.shape[2:4]
        padding = (
            0,
            _padding(width, division_size[1]),
            0,
            _padding(height, division_size[0]),
        )
    return functional.pad(tensor, padding)


def crop_3d(
    tensor: torch.Tensor,
    target_shape: tuple[int, int, int] = (4, 4, 4),
    channel_last: bool = True,
) -> torch.Tensor:
    """Crop depth, height, and width to ``target_shape``."""
    depth, height, width = target_shape
    if channel_last:
        return tensor[:, :depth, :height, :width, :]
    return tensor[:, :, :depth, :height, :width]


def crop_2d(
    tensor: torch.Tensor,
    target_shape: tuple[int, int] = (4, 4),
    channel_last: bool = True,
) -> torch.Tensor:
    """Crop height and width to ``target_shape``."""
    height, width = target_shape
    if channel_last:
        return tensor[:, :height, :width, :]
    return tensor[:, :, :height, :width]


def resize_3d(
    tensor: torch.Tensor,
    division_size: tuple[int, int, int] | None = None,
    target_shape: tuple[int, int, int] | None = None,
    channel_last: bool = True,
) -> torch.Tensor:
    """Resize a 3D tensor to an explicit or divisible shape."""
    if division_size is None and target_shape is None:
        raise ValueError("division_size or target_shape must be provided")
    if channel_last:
        tensor = rearrange(tensor, "b d h w c -> b c d h w")
    if division_size is not None:
        target_shape = get_padded_shape_3d(tensor.shape[2:5], division_size)
    tensor = functional.interpolate(tensor, size=target_shape, mode="trilinear")
    if channel_last:
        tensor = rearrange(tensor, "b c d h w -> b d h w c")
    return tensor


def resize_2d(
    tensor: torch.Tensor,
    division_size: tuple[int, int] | None = None,
    target_shape: tuple[int, int] | None = None,
    channel_last: bool = True,
) -> torch.Tensor:
    """Resize a 2D tensor to an explicit or divisible shape."""
    if division_size is None and target_shape is None:
        raise ValueError("division_size or target_shape must be provided")
    if channel_last:
        tensor = rearrange(tensor, "b h w c -> b c h w")
    if division_size is not None:
        height, width = tensor.shape[2:4]
        target_shape = (
            height + _padding(height, division_size[0]),
            width + _padding(width, division_size[1]),
        )
    tensor = functional.interpolate(tensor, size=target_shape, mode="bilinear")
    if channel_last:
        tensor = rearrange(tensor, "b c h w -> b h w c")
    return tensor
