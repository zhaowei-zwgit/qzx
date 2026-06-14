# Polyp Training and Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a complete train, evaluate, and smoke-test workflow for SAM2-UNet polyp segmentation.

**Architecture:** A focused dataset module reads existing manifests, while a training module owns losses, metrics, epochs, evaluation, and checkpoint state. A single experiment CLI loads JSON configuration, builds either the real SAM2 model or a lightweight smoke encoder, and writes reproducible artifacts.

**Tech Stack:** Python 3.9+, PyTorch, torchvision, Pillow, pytest, JSON

---

### Task 1: Manifest Dataset

**Files:**
- Create: `tests/test_polyp_training.py`
- Create: `src/sam2unet/polyp_dataset.py`

- [ ] Write failing tests that create a one-row manifest and assert RGB image shape, binary mask shape, resizing, and synchronized forced flips.
- [ ] Run `python -m pytest tests/test_polyp_training.py -q` and confirm import failure.
- [ ] Implement `PolypManifestDataset` with deterministic evaluation transforms and synchronized training flips.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Losses and Metrics

**Files:**
- Modify: `tests/test_polyp_training.py`
- Create: `src/sam2unet/training.py`

- [ ] Write failing tests for finite soft-IoU loss, summed three-output deep supervision, and perfect/empty Dice-IoU-MAE metrics.
- [ ] Run the focused tests and confirm missing-symbol failures.
- [ ] Implement `soft_iou_loss`, `deep_supervision_loss`, and `SegmentationMetrics`.
- [ ] Run the focused tests and confirm they pass.

### Task 3: Epochs and Checkpoints

**Files:**
- Modify: `tests/test_polyp_training.py`
- Modify: `src/sam2unet/training.py`

- [ ] Write failing tests showing `train_one_epoch` updates a tiny model, `evaluate_model` returns the three metrics, and checkpoint state resumes optimizer and epoch.
- [ ] Run the focused tests and confirm failures.
- [ ] Implement training, evaluation, checkpoint save, and checkpoint restore helpers.
- [ ] Run the focused tests and confirm they pass.

### Task 4: Experiment CLI

**Files:**
- Create: `tests/test_experiment_cli.py`
- Create: `src/sam2unet/experiment.py`
- Modify: `configs/polyp_train.json`
- Modify: `README.md`

- [ ] Write a failing smoke-command test using a temporary manifest and output directory.
- [ ] Run the CLI test and confirm import or command failure.
- [ ] Implement config loading, model construction, loaders, train/evaluate/smoke commands, artifact output, and real-forward reporting.
- [ ] Extend the JSON config with batch size, accumulation, clipping, seed, checkpoint path, output root, and smoke limits.
- [ ] Document the three commands and 4 GB GPU defaults.
- [ ] Run CLI and focused training tests.

### Task 5: Real Smoke and Final Verification

- [ ] Run `python -m sam2unet.experiment smoke --config configs/polyp_train.json`.
- [ ] Confirm lightweight smoke creates checkpoint, history, and evaluation JSON.
- [ ] Confirm real SAM2 forward result or environment/memory failure is recorded.
- [ ] Run `python -m pytest -q`.
- [ ] Run `python -m compileall -q src tests`.
- [ ] Run `git diff --check`.
