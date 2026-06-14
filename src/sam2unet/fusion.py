"""Stable SAM2-UNet fusion of ParameterNet-style routing and DarkIR features."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .runtime import DEFAULT_MODEL_CONFIG, build_sam2_model, validate_sam2_input


class LayerNorm2d(nn.Module):
    """Apply channel-wise layer normalization independently at every pixel."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        normalized = (x - mean) * torch.rsqrt(variance + self.eps)
        return (
            normalized * self.weight.view(1, -1, 1, 1)
            + self.bias.view(1, -1, 1, 1)
        )


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=1)
        return first * second


class BasicConv2d(nn.Module):
    """The convolution and batch-normalization unit used by the baseline RFB."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class RFBModified(nn.Module):
    """A local copy of the immutable baseline RFB for the `rfb` ablation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = BasicConv2d(in_channels, out_channels, 1)
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, (1, 3), padding=(0, 1)),
            BasicConv2d(out_channels, out_channels, (3, 1), padding=(1, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=3, dilation=3),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, (1, 5), padding=(0, 2)),
            BasicConv2d(out_channels, out_channels, (5, 1), padding=(2, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=5, dilation=5),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, (1, 7), padding=(0, 3)),
            BasicConv2d(out_channels, out_channels, (7, 1), padding=(3, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=7, dilation=7),
        )
        self.conv_cat = BasicConv2d(4 * out_channels, out_channels, 3, padding=1)
        self.conv_res = BasicConv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = (
            self.branch0(x),
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
        )
        return self.relu(self.conv_cat(torch.cat(branches, dim=1)) + self.conv_res(x))


class ParameterNetDynamicProjection(nn.Module):
    """Fuse multiple 1x1 experts, then execute one grouped convolution per batch."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 64,
        num_experts: int = 4,
        router_reduction: int = 16,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if router_reduction <= 0:
            raise ValueError("router_reduction must be positive")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        hidden_channels = max(in_channels // router_reduction, 16)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, num_experts),
        )
        self.expert_weight = nn.Parameter(
            torch.empty(num_experts, out_channels, in_channels, 1, 1)
        )
        self.expert_bias = nn.Parameter(torch.empty(num_experts, out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.expert_weight[0], a=math.sqrt(5))
        bound = 1 / math.sqrt(self.in_channels)
        nn.init.uniform_(self.expert_bias[0], -bound, bound)
        with torch.no_grad():
            for expert_index in range(1, self.num_experts):
                self.expert_weight[expert_index].copy_(self.expert_weight[0])
                self.expert_weight[expert_index].add_(
                    torch.randn_like(self.expert_weight[expert_index]) * 1e-4
                )
                self.expert_bias[expert_index].copy_(self.expert_bias[0])
                self.expert_bias[expert_index].add_(
                    torch.randn_like(self.expert_bias[expert_index]) * 1e-4
                )
        final_router_layer = self.router[-1]
        nn.init.zeros_(final_router_layer.weight)
        nn.init.zeros_(final_router_layer.bias)

    def routing_weights(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.router(x), dim=-1)

    def forward(
        self, x: torch.Tensor, return_router: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected input shaped [B, {self.in_channels}, H, W], got {tuple(x.shape)}"
            )

        routes = self.routing_weights(x)
        dynamic_weight = torch.einsum("bm,moihw->boihw", routes, self.expert_weight)
        dynamic_bias = torch.einsum("bm,mo->bo", routes, self.expert_bias)
        batch_size, _, height, width = x.shape
        grouped_input = x.reshape(1, batch_size * self.in_channels, height, width)
        grouped_weight = dynamic_weight.reshape(
            batch_size * self.out_channels, self.in_channels, 1, 1
        )
        output = F.conv2d(
            grouped_input,
            grouped_weight,
            dynamic_bias.reshape(-1),
            groups=batch_size,
        ).reshape(batch_size, self.out_channels, height, width)
        if return_router:
            return output, routes
        return output


class DarkIRFeatureEnhancer(nn.Module):
    """DarkIR-inspired additive spatial and frequency feature enhancement."""

    def __init__(
        self,
        channels: int = 64,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        dilations: Tuple[int, ...] = (1, 4, 9),
        enable_spatial: bool = True,
        enable_frequency: bool = True,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if dw_expand <= 0 or (channels * dw_expand) % 2:
            raise ValueError("channels * dw_expand must be a positive even number")
        if ffn_expand <= 0:
            raise ValueError("ffn_expand must be positive")
        if not dilations or any(dilation <= 0 for dilation in dilations):
            raise ValueError("dilations must contain positive integers")

        self.enable_spatial = enable_spatial
        self.enable_frequency = enable_frequency
        expanded_channels = channels * dw_expand
        gated_channels = expanded_channels // 2

        self.spatial_norm = LayerNorm2d(channels)
        self.spatial_expand = nn.Conv2d(channels, expanded_channels, 1)
        self.spatial_branches = nn.ModuleList(
            [
                nn.Conv2d(
                    expanded_channels,
                    expanded_channels,
                    3,
                    padding=dilation,
                    dilation=dilation,
                    groups=expanded_channels,
                )
                for dilation in dilations
            ]
        )
        self.spatial_gate = SimpleGate()
        self.spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(gated_channels, gated_channels, 1),
        )
        self.spatial_project = nn.Conv2d(gated_channels, channels, 1)

        self.frequency_norm = LayerNorm2d(channels)
        self.frequency_mlp = nn.Sequential(
            nn.Conv2d(channels, channels * ffn_expand, 1),
            nn.GELU(),
            nn.Conv2d(channels * ffn_expand, channels, 1),
        )
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def spatial_branch(self, x: torch.Tensor) -> torch.Tensor:
        expanded = self.spatial_expand(self.spatial_norm(x))
        mixed = self.spatial_branches[0](expanded)
        for branch in self.spatial_branches[1:]:
            mixed = mixed + branch(expanded)
        gated = self.spatial_gate(mixed)
        attended = gated * self.spatial_attention(gated)
        return self.spatial_project(attended)

    def frequency_branch(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        normalized = self.frequency_norm(x).float()
        height, width = normalized.shape[-2:]
        spectrum = torch.fft.rfft2(normalized, norm="backward")
        magnitude = torch.abs(spectrum).clamp_min(1e-8)
        phase = torch.angle(spectrum)
        enhanced_magnitude = self.frequency_mlp(magnitude)
        enhanced_spectrum = torch.complex(
            enhanced_magnitude * torch.cos(phase),
            enhanced_magnitude * torch.sin(phase),
        )
        enhanced = torch.fft.irfft2(
            enhanced_spectrum, s=(height, width), norm="backward"
        )
        return enhanced.to(dtype=original_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x
        if self.enable_spatial:
            output = output + self.beta * self.spatial_branch(output)
        if self.enable_frequency:
            output = output + self.gamma * self.frequency_branch(output)
        return output


RouterStats = Dict[str, torch.Tensor]


class FeatureBridge(nn.Module):
    """Map one Hiera feature level to the decoder's common feature space."""

    MODES = ("rfb", "static", "parameternet", "darkir", "full")

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 64,
        bridge_mode: str = "full",
        num_experts: int = 4,
        router_reduction: int = 16,
        enable_spatial: bool = True,
        enable_frequency: bool = True,
    ) -> None:
        super().__init__()
        if bridge_mode not in self.MODES:
            raise ValueError(
                f"bridge_mode must be one of {self.MODES}, got {bridge_mode!r}"
            )
        self.bridge_mode = bridge_mode
        if bridge_mode == "rfb":
            self.projection: nn.Module = RFBModified(in_channels, out_channels)
        elif bridge_mode in {"parameternet", "full"}:
            self.projection = ParameterNetDynamicProjection(
                in_channels,
                out_channels,
                num_experts=num_experts,
                router_reduction=router_reduction,
            )
        else:
            self.projection = nn.Conv2d(in_channels, out_channels, 1)

        self.enhancer: nn.Module
        if bridge_mode in {"darkir", "full"}:
            self.enhancer = DarkIRFeatureEnhancer(
                out_channels,
                enable_spatial=enable_spatial,
                enable_frequency=enable_frequency,
            )
        else:
            self.enhancer = nn.Identity()

    @staticmethod
    def _router_stats(routes: torch.Tensor) -> RouterStats:
        entropy = -(routes * routes.clamp_min(1e-8).log()).sum(dim=1).mean()
        return {
            "weights": routes,
            "mean_usage": routes.mean(dim=0),
            "entropy": entropy,
        }

    def forward(
        self, x: torch.Tensor, return_router: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[RouterStats]]]:
        stats = None
        if isinstance(self.projection, ParameterNetDynamicProjection):
            projected, routes = self.projection(x, return_router=True)
            stats = self._router_stats(routes)
        else:
            projected = self.projection(x)
        output = self.enhancer(projected)
        if return_router:
            return output, stats
        return output


class DoubleConv(nn.Module):
    """The two-convolution decoder unit used by SAM2-UNet."""

    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None
    ) -> None:
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Up(nn.Module):
    """Upsample a decoder feature, align it, then fuse its skip connection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, decoder_feature: torch.Tensor, skip_feature: torch.Tensor) -> torch.Tensor:
        decoder_feature = self.up(decoder_feature)
        difference_y = skip_feature.size(2) - decoder_feature.size(2)
        difference_x = skip_feature.size(3) - decoder_feature.size(3)
        decoder_feature = F.pad(
            decoder_feature,
            [
                difference_x // 2,
                difference_x - difference_x // 2,
                difference_y // 2,
                difference_y - difference_y // 2,
            ],
        )
        return self.conv(torch.cat([skip_feature, decoder_feature], dim=1))


class Adapter(nn.Module):
    """Trainable SAM2-UNet prompt adapter around one frozen Hiera block."""

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block
        feature_dim = block.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.GELU(),
            nn.Linear(32, feature_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x + self.prompt_learn(x))


def _build_sam2_encoder(
    checkpoint_path: Optional[Union[str, Path]], model_cfg: str, sam_device: str
) -> nn.Module:
    model = build_sam2_model(checkpoint_path, model_cfg, device=sam_device)

    removable_components = (
        "sam_mask_decoder",
        "sam_prompt_encoder",
        "memory_encoder",
        "memory_attention",
        "mask_downsample",
        "obj_ptr_tpos_proj",
        "obj_ptr_proj",
    )
    for component in removable_components:
        if hasattr(model, component):
            delattr(model, component)
    if hasattr(model.image_encoder, "neck"):
        delattr(model.image_encoder, "neck")

    encoder = model.image_encoder.trunk
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    encoder.blocks = nn.Sequential(*(Adapter(block) for block in encoder.blocks))
    encoder.to(sam_device)
    return encoder


class SAM2UNetFusion(nn.Module):
    """SAM2-UNet with independently ablatable dynamic and DarkIR bridges."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        bridge_mode: str = "full",
        num_experts: int = 4,
        return_router_stats: bool = False,
        enable_spatial: bool = True,
        enable_frequency: bool = True,
        encoder: Optional[nn.Module] = None,
        model_cfg: str = DEFAULT_MODEL_CONFIG,
        sam_device: str = "cpu",
        router_reduction: int = 16,
        feature_channels: int = 64,
    ) -> None:
        super().__init__()
        if feature_channels <= 0:
            raise ValueError("feature_channels must be positive")
        self.bridge_mode = bridge_mode
        self.num_experts = num_experts
        self.return_router_stats = return_router_stats
        self.enable_spatial = enable_spatial
        self.enable_frequency = enable_frequency
        self.model_cfg = model_cfg
        self.sam_device = sam_device
        self.router_reduction = router_reduction
        self.feature_channels = feature_channels
        self.encoder = (
            encoder
            if encoder is not None
            else _build_sam2_encoder(checkpoint_path, model_cfg, sam_device)
        )

        bridge_kwargs = {
            "out_channels": feature_channels,
            "bridge_mode": bridge_mode,
            "num_experts": num_experts,
            "router_reduction": router_reduction,
            "enable_spatial": enable_spatial,
            "enable_frequency": enable_frequency,
        }
        self.bridge1 = FeatureBridge(144, **bridge_kwargs)
        self.bridge2 = FeatureBridge(288, **bridge_kwargs)
        self.bridge3 = FeatureBridge(576, **bridge_kwargs)
        self.bridge4 = FeatureBridge(1152, **bridge_kwargs)

        self.up1 = Up(feature_channels * 2, feature_channels)
        self.up2 = Up(feature_channels * 2, feature_channels)
        self.up3 = Up(feature_channels * 2, feature_channels)
        self.side1 = nn.Conv2d(feature_channels, 1, 1)
        self.side2 = nn.Conv2d(feature_channels, 1, 1)
        self.head = nn.Conv2d(feature_channels, 1, 1)
        self.to(sam_device)

    def get_config(self) -> Dict[str, Union[str, int, bool]]:
        return {
            "bridge_mode": self.bridge_mode,
            "num_experts": self.num_experts,
            "return_router_stats": self.return_router_stats,
            "enable_spatial": self.enable_spatial,
            "enable_frequency": self.enable_frequency,
            "model_cfg": self.model_cfg,
            "sam_device": self.sam_device,
            "router_reduction": self.router_reduction,
            "feature_channels": self.feature_channels,
        }

    def checkpoint_dict(self) -> Dict[str, object]:
        return {"config": self.get_config(), "state_dict": self.state_dict()}

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        torch.save(self.checkpoint_dict(), path)

    def load_baseline_checkpoint(
        self,
        path: Union[str, Path],
        map_location: Union[str, torch.device] = "cpu",
    ):
        payload = self._load_payload(path, map_location=map_location)
        state_dict = payload.get("state_dict", payload)
        if not isinstance(state_dict, Mapping):
            raise ValueError("baseline checkpoint must contain a state dictionary")
        normalized_state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
        return self.load_state_dict(normalized_state_dict, strict=False)

    @staticmethod
    def _load_payload(
        path: Union[str, Path], map_location: Union[str, torch.device] = "cpu"
    ) -> Mapping[str, object]:
        try:
            payload = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=map_location)
        if not isinstance(payload, Mapping):
            raise ValueError("checkpoint must contain a mapping")
        return payload

    @classmethod
    def from_checkpoint(
        cls,
        path: Union[str, Path],
        encoder: Optional[nn.Module] = None,
        map_location: Union[str, torch.device] = "cpu",
        strict: bool = True,
        **config_overrides: object,
    ) -> "SAM2UNetFusion":
        payload = cls._load_payload(path, map_location=map_location)
        if "config" not in payload or "state_dict" not in payload:
            raise ValueError("fusion checkpoint must contain config and state_dict")
        config = dict(payload["config"])
        config.update(config_overrides)
        model = cls(encoder=encoder, **config)
        model.load_state_dict(payload["state_dict"], strict=strict)
        return model

    def _bridge_features(
        self, features: Tuple[torch.Tensor, ...]
    ) -> Tuple[Tuple[torch.Tensor, ...], Dict[str, RouterStats]]:
        if len(features) != 4:
            raise ValueError(f"encoder must return four feature levels, got {len(features)}")
        bridges = (self.bridge1, self.bridge2, self.bridge3, self.bridge4)
        bridged_features = []
        router_stats = {}
        for index, (bridge, feature) in enumerate(zip(bridges, features), start=1):
            if self.return_router_stats:
                bridged, stats = bridge(feature, return_router=True)
                if stats is not None:
                    router_stats[f"bridge{index}"] = stats
            else:
                bridged = bridge(feature)
            bridged_features.append(bridged)
        return tuple(bridged_features), router_stats

    def forward(
        self, x: torch.Tensor
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, RouterStats]],
    ]:
        validate_sam2_input(x)
        input_size = x.shape[-2:]
        features = tuple(self.encoder(x))
        bridged_features, router_stats = self._bridge_features(features)
        feature1, feature2, feature3, feature4 = bridged_features

        decoded = self.up1(feature4, feature3)
        out1 = F.interpolate(
            self.side1(decoded), size=input_size, mode="bilinear", align_corners=False
        )
        decoded = self.up2(decoded, feature2)
        out2 = F.interpolate(
            self.side2(decoded), size=input_size, mode="bilinear", align_corners=False
        )
        decoded = self.up3(decoded, feature1)
        out = F.interpolate(
            self.head(decoded), size=input_size, mode="bilinear", align_corners=False
        )
        outputs = (out, out1, out2)
        if self.return_router_stats:
            return outputs, router_stats
        return outputs


__all__ = [
    "Adapter",
    "DarkIRFeatureEnhancer",
    "FeatureBridge",
    "LayerNorm2d",
    "ParameterNetDynamicProjection",
    "RFBModified",
    "SAM2UNetFusion",
    "Up",
]
