# SAM2-UNet DarkIR ParameterNet Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independently testable SAM2-UNet fusion model that combines ParameterNet-style dynamic projection with DarkIR-inspired spatial-frequency feature enhancement while preserving both existing model files.

**Architecture:** A `FeatureBridge` converts each Hiera feature level to 64 channels using RFB, static convolution, or sample-conditioned expert fusion, then optionally applies a zero-initialized DarkIR feature enhancer. `SAM2UNetFusion` owns four bridges and the unchanged three-stage U-Net decoder, supports dependency-injected encoders for testing, and exposes router statistics and checkpoint configuration.

**Tech Stack:** Python 3.9+, PyTorch, pytest, optional Meta SAM2 package for real Hiera-L construction

---

### Task 1: Define block behavior with tests

**Files:**
- Create: `tests/test_darkir_parameternet_blocks.py`
- Create: `SAM2UNet_darkir_parameternet.py`

- [ ] **Step 1: Write failing tests**

Test dynamic projection shape, normalized routing, grouped-convolution equivalence for one expert, gradients, identity initialization, strict depth-wise branches, finite FFT output, mixed precision, and all bridge modes.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_darkir_parameternet_blocks.py -q`

Expected: FAIL during import because `SAM2UNet_darkir_parameternet.py` does not exist.

- [ ] **Step 3: Implement the blocks**

Implement:

```python
ParameterNetDynamicProjection(in_channels, out_channels=64, num_experts=4, router_reduction=16)
DarkIRFeatureEnhancer(channels=64, dw_expand=2, ffn_expand=2, dilations=(1, 4, 9))
FeatureBridge(in_channels, out_channels=64, bridge_mode="full", num_experts=4)
```

Use one batch-grouped `F.conv2d` call for the fused expert kernel. Keep FFT work in float32 and use additive, zero-initialized residual scaling.

- [ ] **Step 4: Run block tests**

Run: `python -m pytest tests/test_darkir_parameternet_blocks.py -q`

Expected: all tests pass.

### Task 2: Integrate the full model

**Files:**
- Modify: `SAM2UNet_darkir_parameternet.py`
- Create: `tests/test_sam2unet_fusion.py`

- [ ] **Step 1: Write failing integration tests**

Use a fake four-level encoder to test every bridge mode without requiring a SAM2 installation or checkpoint. Verify three full-resolution outputs, optional router statistics, encoder freezing behavior through the real builder path, and checkpoint configuration round trips.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_sam2unet_fusion.py -q`

Expected: FAIL because `SAM2UNetFusion` and checkpoint helpers are not implemented.

- [ ] **Step 3: Implement model integration**

Implement:

```python
SAM2UNetFusion(
    checkpoint_path=None,
    bridge_mode="full",
    num_experts=4,
    return_router_stats=False,
    enable_spatial=True,
    enable_frequency=True,
    encoder=None,
)
```

When no encoder is injected, lazily import and build SAM2 Hiera-L, remove unused SAM2 heads, freeze trunk parameters, and add trainable adapters. Save model configuration beside the state dictionary.

- [ ] **Step 4: Run integration tests**

Run: `python -m pytest tests/test_sam2unet_fusion.py -q`

Expected: all tests pass.

### Task 3: Add project usage and dependency metadata

**Files:**
- Create: `requirements.txt`
- Create: `README.md`

- [ ] **Step 1: Document installation and use**

Document block-only testing, optional SAM2 installation, all bridge modes, forward return contracts, checkpoint helpers, and the fact that real training requires user-provided data and a SAM2 checkpoint.

- [ ] **Step 2: Record minimal test dependencies**

Pin no platform-specific PyTorch build. List `torch>=2.1` and `pytest>=7.0`, and document SAM2 as an optional Git dependency because its supported Python/CUDA combinations vary.

### Task 4: Verify requirements and regressions

**Files:**
- Verify: `SAM2UNet.py`
- Verify: `SAM2UNet_dblock_dat_fused_rfbhou.py`
- Verify: all new files

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Run syntax compilation**

Run: `python -m compileall -q SAM2UNet.py SAM2UNet_dblock_dat_fused_rfbhou.py SAM2UNet_darkir_parameternet.py tests`

Expected: exit code 0.

- [ ] **Step 3: Confirm baseline files are unchanged**

Run: `git diff -- SAM2UNet.py SAM2UNet_dblock_dat_fused_rfbhou.py`

Expected: no output.

- [ ] **Step 4: Review requirement coverage**

Confirm all five modes, router statistics, spatial/frequency ablations, finite mixed-precision FFT behavior, configuration checkpoints, and baseline immutability are covered by tests or documentation.
