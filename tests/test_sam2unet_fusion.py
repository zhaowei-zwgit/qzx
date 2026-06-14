from pathlib import Path
import sys
import types

import pytest
import torch
import torch.nn as nn

from sam2unet.fusion import Adapter, SAM2UNetFusion


class FakeEncoder(nn.Module):
    def forward(self, x):
        batch, _, height, width = x.shape
        return (
            torch.randn(batch, 144, height // 4, width // 4, device=x.device),
            torch.randn(batch, 288, height // 8, width // 8, device=x.device),
            torch.randn(batch, 576, height // 16, width // 16, device=x.device),
            torch.randn(batch, 1152, height // 32, width // 32, device=x.device),
        )


@pytest.mark.parametrize("mode", ["rfb", "static", "parameternet", "darkir", "full"])
def test_full_model_supports_all_modes_and_output_contract(mode):
    model = SAM2UNetFusion(bridge_mode=mode, num_experts=2, encoder=FakeEncoder())
    x = torch.randn(1, 3, 64, 64)

    out, out1, out2 = model(x)

    assert out.shape == out1.shape == out2.shape == (1, 1, 64, 64)


def test_full_model_can_return_per_bridge_router_stats():
    model = SAM2UNetFusion(
        bridge_mode="full",
        num_experts=3,
        return_router_stats=True,
        encoder=FakeEncoder(),
    )

    outputs, stats = model(torch.randn(2, 3, 64, 64))

    assert len(outputs) == 3
    assert set(stats) == {"bridge1", "bridge2", "bridge3", "bridge4"}
    assert all(item["weights"].shape == (2, 3) for item in stats.values())


def test_checkpoint_round_trip_preserves_config_and_outputs(tmp_path: Path):
    model = SAM2UNetFusion(
        bridge_mode="parameternet",
        num_experts=2,
        enable_spatial=False,
        enable_frequency=True,
        encoder=FakeEncoder(),
    )
    path = tmp_path / "fusion.pt"

    model.save_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert payload["config"] == model.get_config()
    restored = SAM2UNetFusion.from_checkpoint(path, encoder=FakeEncoder())
    assert restored.get_config() == model.get_config()
    for expected, actual in zip(model.state_dict().values(), restored.state_dict().values()):
        torch.testing.assert_close(expected, actual)


def test_injected_encoder_is_used_as_supplied():
    encoder = FakeEncoder()
    model = SAM2UNetFusion(bridge_mode="static", encoder=encoder)

    assert model.encoder is encoder


def test_model_preserves_full_resolution_for_352_input():
    model = SAM2UNetFusion(bridge_mode="static", encoder=FakeEncoder())

    outputs = model(torch.randn(1, 3, 352, 352))

    assert all(output.shape == (1, 1, 352, 352) for output in outputs)


def test_baseline_checkpoint_loads_common_weights_non_strictly(tmp_path: Path):
    model = SAM2UNetFusion(bridge_mode="static", encoder=FakeEncoder())
    checkpoint = tmp_path / "baseline.pt"
    expected_head = torch.full_like(model.head.weight, 0.25)
    torch.save({"head.weight": expected_head}, checkpoint)

    incompatible = model.load_baseline_checkpoint(checkpoint)

    torch.testing.assert_close(model.head.weight, expected_head)
    assert incompatible.missing_keys
    assert not incompatible.unexpected_keys


def test_real_builder_path_freezes_hiera_and_wraps_adapters(monkeypatch):
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

    trunk = FakeTrunk()
    fake_model = types.SimpleNamespace(
        image_encoder=types.SimpleNamespace(trunk=trunk, neck=object()),
        sam_mask_decoder=object(),
    )
    builder_calls = []
    build_module = types.ModuleType("sam2.build_sam")

    def fake_build_sam2(model_cfg, checkpoint=None, **kwargs):
        builder_calls.append((model_cfg, checkpoint, kwargs))
        return fake_model

    build_module.build_sam2 = fake_build_sam2
    monkeypatch.setitem(sys.modules, "sam2", types.ModuleType("sam2"))
    monkeypatch.setitem(sys.modules, "sam2.build_sam", build_module)

    model = SAM2UNetFusion(bridge_mode="static")

    assert builder_calls == [
        ("configs/sam2/sam2_hiera_l.yaml", None, {"device": "cpu"})
    ]
    assert all(isinstance(block, Adapter) for block in model.encoder.blocks)
    assert all(
        not parameter.requires_grad
        for block in model.encoder.blocks
        for parameter in block.block.parameters()
    )
    assert all(
        parameter.requires_grad
        for block in model.encoder.blocks
        for parameter in block.prompt_learn.parameters()
    )
