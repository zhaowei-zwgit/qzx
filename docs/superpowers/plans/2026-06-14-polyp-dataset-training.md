# Polyp Dataset Training Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Download Kvasir-SEG and CVC-ClinicDB, normalize the standard 1,450-image training split, and add reproducible dataset validation and experiment configuration.

**Architecture:** A small dataset-preparation module discovers extracted image/mask pairs, creates deterministic official-sized splits, copies them into a collision-safe normalized layout, and writes CSV manifests. A PowerShell download script obtains the two public Kaggle mirrors and invokes preparation; a JSON configuration records the initial training experiment.

**Tech Stack:** Python 3.9+, Pillow, pytest, PowerShell, Kaggle CLI/API

---

### Task 1: Ignore Local Data

**Files:**
- Modify: `.gitignore`

- [ ] Add `data/` to `.gitignore`.
- [ ] Verify `git status --short --ignored data` reports the directory as ignored.

### Task 2: Dataset Preparation Module

**Files:**
- Create: `tests/test_polyp_data.py`
- Create: `src/sam2unet/data.py`

- [ ] Write failing tests for pair discovery, deterministic official-sized splitting, prefixed combined training names, manifests, and missing-pair errors.
- [ ] Run `python -m pytest tests/test_polyp_data.py -q` and confirm the tests fail because `sam2unet.data` is missing.
- [ ] Implement `discover_pairs`, `prepare_polyp_datasets`, and `validate_prepared_polyp_data`.
- [ ] Run `python -m pytest tests/test_polyp_data.py -q` and confirm all tests pass.

### Task 3: Download and Experiment Configuration

**Files:**
- Create: `scripts/download_polyp_data.ps1`
- Create: `configs/polyp_train.json`
- Modify: `README.md`

- [ ] Add a PowerShell script that downloads both Kaggle datasets, extracts them, invokes the Python preparation API, and validates counts.
- [ ] Add a JSON experiment configuration for 352x352 binary segmentation, 20 epochs, Dice/IoU/MAE, and the `rfb`, `static`, and `full` modes.
- [ ] Document download, preparation, validation, and expected counts in `README.md`.

### Task 4: Download and Prepare Real Data

**Files:**
- Create locally: `data/polyps/`

- [ ] Run `scripts/download_polyp_data.ps1`.
- [ ] Confirm Kvasir-SEG has 900 training and 100 test pairs.
- [ ] Confirm CVC-ClinicDB has 550 training and 62 test pairs.
- [ ] Confirm the combined training set has 1,450 image-mask pairs.

### Task 5: Final Verification

- [ ] Run `python -m pytest -q`.
- [ ] Run `python -m compileall -q src tests`.
- [ ] Run the real-data validation command and record its output.
- [ ] Inspect `git status --short --ignored` to verify downloaded datasets are ignored.
