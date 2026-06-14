"""Runtime helpers for installed or vendored Meta SAM2 source trees."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Optional, Union


DEFAULT_MODEL_CONFIG = "configs/sam2/sam2_hiera_l.yaml"


def _import_build_sam2():
    try:
        return importlib.import_module("sam2.build_sam").build_sam2
    except ModuleNotFoundError as original_error:
        vendored_root = Path(__file__).resolve().parents[2] / "third_party" / "sam2"
        build_module = vendored_root / "sam2" / "build_sam.py"
        if not build_module.is_file():
            raise ImportError(
                "Meta SAM2 is unavailable. Install it or place its repository in ./sam2."
            ) from original_error

        vendored_root_text = str(vendored_root)
        if vendored_root_text not in sys.path:
            sys.path.insert(0, vendored_root_text)
        for module_name in tuple(sys.modules):
            if module_name == "sam2" or module_name.startswith("sam2."):
                del sys.modules[module_name]
        return importlib.import_module("sam2.build_sam").build_sam2


def build_sam2_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
    model_cfg: str = DEFAULT_MODEL_CONFIG,
    device: str = "cpu",
    **kwargs,
):
    """Build SAM2 without relying on the caller's current working directory."""

    build_sam2 = _import_build_sam2()
    checkpoint = None if checkpoint_path is None else str(checkpoint_path)
    return build_sam2(model_cfg, checkpoint, device=device, **kwargs)


def validate_sam2_input(x, stride: int = 32) -> None:
    """Validate the spatial contract required by the Hiera feature pyramid."""

    if x.ndim != 4:
        raise ValueError(f"SAM2-UNet expects a 4D BCHW tensor, got shape {tuple(x.shape)}")
    height, width = x.shape[-2:]
    if height % stride or width % stride:
        raise ValueError(
            f"SAM2-UNet input height and width must be divisible by {stride}, "
            f"got {height}x{width}"
        )


__all__ = ["DEFAULT_MODEL_CONFIG", "build_sam2_model", "validate_sam2_input"]
