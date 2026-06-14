# Root Main Entry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a root-level `main.py` that selects and runs smoke, train, or evaluate workflows through the existing experiment CLI.

**Architecture:** Implement a thin command adapter that parses root-level options and forwards them to `sam2unet.experiment.main`. Keep all model, data, training, evaluation, checkpoint, and seed behavior in the existing package.

**Tech Stack:** Python, argparse, pytest, existing `sam2unet.experiment` CLI

---

### Task 1: Define Entry-Point Behavior With Tests

**Files:**
- Create: `tests/test_main.py`

- [x] **Step 1: Write failing tests**

Test that the default command forwards smoke mode and the default config, train
mode forwards its specific options, evaluate mode requires a checkpoint, and
the delegated exit code is returned.

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_main.py -q`

Expected: FAIL because root `main.py` does not exist.

### Task 2: Implement Root Entry Point

**Files:**
- Create: `main.py`

- [x] **Step 1: Implement the minimal adapter**

Create an argparse parser, translate root options into the existing experiment
subcommand format, invoke `sam2unet.experiment.main`, and print start/end time
and duration.

- [x] **Step 2: Run entry-point tests**

Run: `python -m pytest tests/test_main.py -q`

Expected: PASS.

### Task 3: Document and Verify

**Files:**
- Modify: `README.md`

- [x] **Step 1: Add root entry examples**

Document `python main.py`, formal training, resume, and evaluation commands.

- [x] **Step 2: Run focused and full verification**

Run:

```powershell
python main.py --help
python -m pytest tests/test_main.py tests/test_experiment_cli.py -q
python main.py --mode smoke --device cpu --limit-train 1 --limit-test 1
python -m pytest -q
```

Expected: help exits successfully, focused tests pass, smoke report is
generated, and the full test suite passes.
