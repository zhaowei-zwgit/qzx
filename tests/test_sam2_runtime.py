import importlib
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch
import torch.nn as nn


@pytest.fixture
def fake_sam2_builder(monkeypatch):
    calls = []
    build_module = types.ModuleType("sam2.build_sam")

    def build_sam2(config_file, checkpoint=None, **kwargs):
        calls.append((config_file, checkpoint, kwargs))
        return object()

    build_module.build_sam2 = build_sam2
    monkeypatch.setitem(sys.modules, "sam2", types.ModuleType("sam2"))
    monkeypatch.setitem(sys.modules, "sam2.build_sam", build_module)
    return calls


@pytest.mark.parametrize(
    "module_name", ["sam2unet.baseline", "sam2unet.experimental_darkir"]
)
def test_legacy_models_are_importable_with_vendored_sam2(module_name):
    module = importlib.import_module(module_name)

    assert hasattr(module, "SAM2UNet")


def test_package_import_does_not_eagerly_import_model_modules():
    source_root = Path(__file__).resolve().parents[1] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root)
    code = (
        "import sys, sam2unet; "
        "assert 'sam2unet.baseline' not in sys.modules; "
        "assert 'sam2unet.experimental_darkir' not in sys.modules; "
        "assert 'sam2unet.fusion' not in sys.modules"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_runtime_builder_uses_official_config_key_and_explicit_device(fake_sam2_builder):
    from sam2unet.runtime import build_sam2_model

    build_sam2_model(checkpoint_path=None, device="cpu")

    assert fake_sam2_builder == [
        ("configs/sam2/sam2_hiera_l.yaml", None, {"device": "cpu"})
    ]


@pytest.mark.parametrize("shape", [(3, 64, 64), (1, 3, 65, 64), (1, 3, 64, 67)])
def test_validate_sam2_input_rejects_invalid_shapes(shape):
    from sam2unet.runtime import validate_sam2_input

    with pytest.raises(ValueError, match="divisible by 32|4D"):
        validate_sam2_input(torch.randn(*shape))


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = types.SimpleNamespace(qkv=nn.Linear(4, 12))
        self.projection = nn.Linear(4, 4)

    def forward(self, x):
        return self.projection(x)


class FakeTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([FakeBlock(), FakeBlock()])


def fake_sam2_model_on(device):
    trunk = FakeTrunk().to(device)
    return types.SimpleNamespace(
        image_encoder=types.SimpleNamespace(trunk=trunk, neck=object()),
        sam_mask_decoder=object(),
        sam_prompt_encoder=object(),
        memory_encoder=object(),
        memory_attention=object(),
        mask_downsample=object(),
        obj_ptr_tpos_proj=object(),
        obj_ptr_proj=object(),
    )


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("sam2unet.baseline", "SAM2UNet"),
        ("sam2unet.experimental_darkir", "SAM2UNet"),
        ("sam2unet.fusion", "SAM2UNetFusion"),
    ],
)
def test_inserted_adapters_follow_requested_sam_device(
    monkeypatch, module_name, class_name
):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(
        module, "build_sam2_model", lambda *args, **kwargs: fake_sam2_model_on("meta")
    )

    model = getattr(module, class_name)(sam_device="meta", bridge_mode="static") if (
        class_name == "SAM2UNetFusion"
    ) else getattr(module, class_name)(sam_device="meta")

    assert {parameter.device.type for parameter in model.parameters()} == {"meta"}
