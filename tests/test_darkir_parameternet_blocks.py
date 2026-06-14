import pytest
import torch
import torch.nn as nn

from sam2unet.fusion import (
    DarkIRFeatureEnhancer,
    FeatureBridge,
    ParameterNetDynamicProjection,
)


def test_dynamic_projection_returns_normalized_per_sample_routes():
    layer = ParameterNetDynamicProjection(6, 4, num_experts=3)
    x = torch.randn(2, 6, 5, 7)

    y, routes = layer(x, return_router=True)

    assert y.shape == (2, 4, 5, 7)
    assert routes.shape == (2, 3)
    torch.testing.assert_close(routes.sum(dim=1), torch.ones(2))
    assert torch.isfinite(y).all()


def test_single_expert_dynamic_projection_matches_conv2d():
    layer = ParameterNetDynamicProjection(3, 5, num_experts=1)
    conv = nn.Conv2d(3, 5, 1)
    with torch.no_grad():
        layer.expert_weight[0].copy_(conv.weight)
        layer.expert_bias[0].copy_(conv.bias)

    x = torch.randn(3, 3, 4, 6)

    torch.testing.assert_close(layer(x), conv(x))


def test_dynamic_projection_sends_gradients_to_experts_and_router():
    layer = ParameterNetDynamicProjection(4, 3, num_experts=2)
    y = layer(torch.randn(2, 4, 5, 5))

    y.square().mean().backward()

    assert layer.expert_weight.grad is not None
    assert layer.expert_weight.grad.abs().sum() > 0
    assert all(parameter.grad is not None for parameter in layer.router.parameters())


def test_darkir_enhancer_is_initially_identity_and_finite():
    layer = DarkIRFeatureEnhancer(8)
    x = torch.randn(2, 8, 9, 11, requires_grad=True)

    y = layer(x)

    torch.testing.assert_close(y, x)
    assert torch.isfinite(y).all()
    y.mean().backward()
    assert x.grad is not None


def test_darkir_spatial_branches_are_strict_depthwise_convolutions():
    layer = DarkIRFeatureEnhancer(8, dw_expand=2)

    assert len(layer.spatial_branches) == 3
    assert all(branch.groups == branch.in_channels == branch.out_channels for branch in layer.spatial_branches)


def test_darkir_frequency_branch_returns_real_finite_tensor():
    layer = DarkIRFeatureEnhancer(8)
    x = torch.randn(2, 8, 9, 11)

    y = layer.frequency_branch(x)

    assert not y.is_complex()
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.skipif(not hasattr(torch, "autocast"), reason="autocast is unavailable")
def test_darkir_cpu_mixed_precision_path_is_finite():
    layer = DarkIRFeatureEnhancer(8)
    x = torch.randn(1, 8, 8, 8)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        y = layer(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("mode", ["rfb", "static", "parameternet", "darkir", "full"])
def test_feature_bridge_supports_all_ablation_modes(mode):
    bridge = FeatureBridge(12, 8, bridge_mode=mode, num_experts=2)

    y, router = bridge(torch.randn(2, 12, 8, 8), return_router=True)

    assert y.shape == (2, 8, 8, 8)
    if mode in {"parameternet", "full"}:
        assert router["weights"].shape == (2, 2)
        assert router["mean_usage"].shape == (2,)
        assert router["entropy"].ndim == 0
    else:
        assert router is None


def test_feature_bridge_rejects_unknown_mode():
    with pytest.raises(ValueError, match="bridge_mode"):
        FeatureBridge(12, 8, bridge_mode="unknown")
