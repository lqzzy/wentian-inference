"""Canonical construction and strict checkpoint loading for Wentian."""

from pathlib import Path

import torch

from .config import IMAGE_SIZE, WENTIAN_CONFIG
from .network import WentianModel


def load_wentian_model(checkpoint) -> WentianModel:
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"checkpoint does not contain a 'model' state dict: {checkpoint}")

    model = WentianModel(image_size=IMAGE_SIZE, **WENTIAN_CONFIG)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model
