"""Canonical configuration for the released Wentian checkpoint."""

IMAGE_SIZE = (721, 1440)

WENTIAN_CONFIG = {
    "num_plevel_var": (5, 5),
    "num_surface_var": (7, 4),
    "num_pressure_level": 13,
    "depths": [2, 6, 6, 2],
    "heads": [2, 8, 8, 2],
    "seq_len": 2,
    "dim": 256,
    "patch_size": [2, 4, 4],
    "window_size": [2, 6, 12],
    "use_checkpoint": False,
}
