# Dataset Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Download, normalize, register, and validate every currently missing dataset named in README.md.

**Architecture:** Extend the existing PraNet data module only for additional polyp evaluation sets, keeping the verified training split untouched. Acquire camouflaged-object datasets into their README-defined local hierarchy and validate them independently because the project has no COD training module yet.

**Tech Stack:** Python 3.9, pathlib, csv, Pillow, pytest, PowerShell, official Google Drive/GitHub dataset releases

---

### Task 1: Extend PraNet Test-Set Preparation

**Files:**
- Modify: `tests/test_polyp_data.py`
- Modify: `src/sam2unet/data.py`

- [ ] **Step 1: Write a failing test for all PraNet test sets**

```python
def test_prepare_pranet_polyp_data_keeps_all_supported_test_sets(tmp_path: Path):
    train_source, test_source = _build_pranet_source(tmp_path / "source")
    _write_pair(test_source, "CVC-300", "cvc-300-test")
    _write_pair(test_source, "ETIS-LaribPolypDB", "etis-test")
    summary = prepare_pranet_polyp_data(train_source, test_source, tmp_path / "prepared")
    assert summary["test/CVC-ColonDB"] == 1
    assert summary["test/CVC-300"] == 1
    assert summary["test/ETIS"] == 1
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run: `python -m pytest tests/test_polyp_data.py::test_prepare_pranet_polyp_data_keeps_all_supported_test_sets -q`

Expected: FAIL because the current `TEST_DATASETS` tuple contains only Kvasir and CVC-ClinicDB.

- [ ] **Step 3: Implement canonical source-to-output mappings**

```python
TEST_DATASETS = {
    "Kvasir": "Kvasir",
    "CVC-ClinicDB": "CVC-ClinicDB",
    "CVC-300": "CVC-300",
    "CVC-ColonDB": "CVC-ColonDB",
    "ETIS-LaribPolypDB": "ETIS",
}
```

Iterate through this mapping when copying pairs and writing manifests.

- [ ] **Step 4: Run focused data tests and confirm GREEN**

Run: `python -m pytest tests/test_polyp_data.py -q`

Expected: all data tests pass.

### Task 2: Make Validation Cover Configured Test Sets

**Files:**
- Modify: `tests/test_polyp_data.py`
- Modify: `src/sam2unet/data.py`

- [ ] **Step 1: Write a failing validation test**

```python
def test_validate_prepared_polyp_data_includes_extended_test_sets(tmp_path: Path):
    train_source, test_source = _build_pranet_source(tmp_path / "source")
    _write_pair(test_source, "CVC-300", "cvc-300-test")
    _write_pair(test_source, "ETIS-LaribPolypDB", "etis-test")
    output = tmp_path / "prepared"
    prepare_pranet_polyp_data(train_source, test_source, output)
    summary = validate_prepared_polyp_data(output)
    assert summary["test/CVC-300"] == 1
    assert summary["test/CVC-ColonDB"] == 1
    assert summary["test/ETIS"] == 1
```

- [ ] **Step 2: Run the test and confirm RED**

Run: `python -m pytest tests/test_polyp_data.py::test_validate_prepared_polyp_data_includes_extended_test_sets -q`

Expected: FAIL because validation currently hard-codes three locations.

- [ ] **Step 3: Derive validation locations from the canonical mapping**

```python
locations = {"train": output_root / "train"}
locations.update(
    {f"test/{name}": output_root / "test" / name for name in TEST_DATASETS.values()}
)
```

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run: `python -m pytest tests/test_polyp_data.py -q`

Expected: all tests pass.

### Task 3: Acquire and Register Extended Polyp Tests

**Files:**
- Create locally: `data/polyps/archives/PraNet-TestDataset.zip`
- Create locally: `data/polyps/prepared/test/CVC-300/`
- Create locally: `data/polyps/prepared/test/CVC-ColonDB/`
- Create locally: `data/polyps/prepared/test/ETIS/`
- Create locally: `data/polyps/prepared/manifests/test-CVC-300.csv`
- Create locally: `data/polyps/prepared/manifests/test-CVC-ColonDB.csv`
- Create locally: `data/polyps/prepared/manifests/test-ETIS.csv`
- Modify: `configs/polyp_train.json`
- Modify: `configs/polyp_sources.json`

- [ ] **Step 1: Download PraNet's official TestDataset archive**

Run: `gdown 1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao -O data/polyps/archives/PraNet-TestDataset.zip`

Expected: a valid ZIP containing CVC-300, CVC-ColonDB, and ETIS-LaribPolypDB.

- [ ] **Step 2: Extract only the missing test directories and normalize names**

Use PowerShell `Expand-Archive` into `data/polyps/raw/pranet-test`, then invoke the tested Python pair discovery and manifest writer so output paths follow the existing layout.

- [ ] **Step 3: Register manifests and expected counts**

```json
"CVC-300": "data/polyps/prepared/manifests/test-CVC-300.csv",
"CVC-ColonDB": "data/polyps/prepared/manifests/test-CVC-ColonDB.csv",
"ETIS": "data/polyps/prepared/manifests/test-ETIS.csv"
```

Expected counts are 60, 380, and 196 respectively.

- [ ] **Step 4: Validate real polyp data**

Run: `python -m sam2unet.data validate data/polyps/prepared`

Expected: all six reported counts match the design.

### Task 4: Acquire and Validate COD Benchmarks

**Files:**
- Create locally: `data/cod/archives/`
- Create locally: `data/cod/prepared/train/CAMO/`
- Create locally: `data/cod/prepared/train/COD10K/`
- Create locally: `data/cod/prepared/test/CHAMELEON/`
- Create locally: `data/cod/prepared/test/CAMO/`
- Create locally: `data/cod/prepared/test/COD10K/`
- Create locally: `data/cod/prepared/test/NC4K/`
- Create: `data/cod/dataset_inventory.json`

- [ ] **Step 1: Download official releases**

Use CAMO's author page, the official SINet/COD10K repository links, and the NC4K project repository. Save each transfer to `data/cod/archives/` before extraction.

- [ ] **Step 2: Extract one archive at a time**

After each successful extraction, remove duplicate archives as needed to keep at least 2 GB free working space.

- [ ] **Step 3: Normalize and inventory datasets**

For each split, identify image and binary-mask directories, match files by case-insensitive stem, verify them with Pillow, and write counts and source URLs to `data/cod/dataset_inventory.json`.

- [ ] **Step 4: Compare counts with README**

Expected: CAMO 1,250 train and 250 test; COD10K 3,040 train and 2,026 test; CHAMELEON 76 test; NC4K 4,121 test.

### Task 5: Final Verification and Documentation

**Files:**
- Modify: `README.md`
- Modify: `configs/polyp_sources.json`

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest tests/ -q`

Expected: all tests pass.

- [ ] **Step 2: Run manifest loader spot checks**

Run one sample from every polyp manifest through `PolypManifestDataset` and verify image shape `(3, 352, 352)` and mask shape `(1, 352, 352)`.

- [ ] **Step 3: Update dataset status and source records**

Mark only verified datasets as available. Keep COD training support described as unavailable until actual COD code and configuration exist.

- [ ] **Step 4: Report inaccessible datasets**

List each unsuccessful item with its official source, failure reason, and required user action; omit this list only when all datasets are verified locally.

