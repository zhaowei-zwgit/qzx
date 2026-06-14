# Project Structure Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the repository into a standard `src` Python package layout while preserving model behavior, tests, vendored SAM2 loading, documentation, papers, and checkpoints.

**Architecture:** Production Python code moves into `src/sam2unet`, with stable package modules for baseline, experimental, fusion, and runtime behavior. Vendored Meta SAM2 moves into `third_party/sam2`; runtime resolution remains independent of the current working directory. Documentation is grouped by purpose under `docs`.

**Tech Stack:** Python, PyTorch, pytest, Meta SAM2, Git

---

### Task 1: Establish the package import contract

**Files:**
- Modify: `tests/test_darkir_parameternet_blocks.py`
- Modify: `tests/test_sam2unet_fusion.py`
- Modify: `tests/test_sam2_runtime.py`

- [ ] Change imports to `sam2unet.baseline`, `sam2unet.experimental_darkir`, `sam2unet.fusion`, and `sam2unet.runtime`.
- [ ] Run `python -m pytest tests/test_sam2_runtime.py -q`.
- [ ] Confirm failure because the `sam2unet` package does not exist.

### Task 2: Move production source into `src/sam2unet`

**Files:**
- Create: `src/sam2unet/__init__.py`
- Move: `SAM2UNet.py` to `src/sam2unet/baseline.py`
- Move: `SAM2UNet_dblock_dat_fused_rfbhou.py` to `src/sam2unet/experimental_darkir.py`
- Move: `SAM2UNet_darkir_parameternet.py` to `src/sam2unet/fusion.py`
- Move: `sam2_runtime.py` to `src/sam2unet/runtime.py`

- [ ] Update internal imports to package-relative imports.
- [ ] Export the three public model classes from `src/sam2unet/__init__.py`.
- [ ] Ensure direct module execution uses `python -m sam2unet.baseline`.
- [ ] Run the full test suite.

### Task 3: Move vendored SAM2 and preserve runtime resolution

**Files:**
- Move: `sam2/` to `third_party/sam2/`
- Modify: `src/sam2unet/runtime.py`
- Modify: `pyrightconfig.json`

- [ ] Resolve vendored SAM2 from `<repo>/third_party/sam2`.
- [ ] Add `src` and `third_party/sam2` to Pyright paths.
- [ ] Run a real Hiera-L CPU build.

### Task 4: Group documentation and papers

**Files:**
- Move: `DarkIR.pdf` to `docs/papers/DarkIR.pdf`
- Move: `ParameterNet.pdf` to `docs/papers/ParameterNet.pdf`
- Move: `sam2unet.pdf` to `docs/papers/sam2unet.pdf`
- Move: `SAM2UNet融合DarkIR与ParameterNet稳健版设计文档.md` to `docs/design/SAM2UNet融合DarkIR与ParameterNet稳健版设计文档.md`
- Move: `两个Python文件与三篇论文关系分析.md` to `docs/analysis/两个Python文件与三篇论文关系分析.md`
- Modify: `README.md`

- [ ] Update README commands and paths.
- [ ] Update documentation references to the new source paths.
- [ ] Keep checkpoints in `checkpoints/`.

### Task 5: Final verification

- [ ] Run `python -m pytest -q`.
- [ ] Run `D:\Anaconda\python.exe -m pytest -q`.
- [ ] Run syntax compilation over `src/sam2unet` and `tests`.
- [ ] Run `python -m sam2unet.baseline` and `python -m sam2unet.experimental_darkir`.
- [ ] Load `checkpoints/sam2_hiera_large.pt` through `sam2unet.fusion.SAM2UNetFusion` and execute a CUDA forward.
- [ ] Confirm the repository root contains only project-level directories and configuration files.
