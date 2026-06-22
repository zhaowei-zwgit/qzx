# COD Training Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CAMO, COD10K, NC4K, and CHAMELEON fully runnable through the existing `main.py` smoke, train, and evaluate workflows.

**Architecture:** Add a strict directory-backed COD dataset with the same sample contract as the manifest-backed Polyp dataset. Dispatch loaders from `dataset_type` in the shared experiment pipeline, concatenate CAMO and COD10K for training, and keep the model, loss, metrics, monitoring, and checkpoint code shared.

**Tech Stack:** Python 3.9, pathlib, Pillow, PyTorch, torchvision, pytest, JSON

---

### Task 1: Add The COD Directory Dataset

**Files:**
- Create: `src/sam2unet/cod_dataset.py`
- Create: `tests/test_cod_dataset.py`

- [ ] **Step 1: Write failing tests for loading, transforms, and invalid layouts**

Create `tests/test_cod_dataset.py` with temporary image fixtures and these tests:

```python
from pathlib import Path

import pytest
import torch
from PIL import Image

from sam2unet.cod_dataset import CODDirectoryDataset


def _write_pair(root: Path, stem: str) -> None:
    image_dir = root / "images"
    mask_dir = root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (4, 2), "black")
    image.putpixel((0, 0), (255, 0, 0))
    image.save(image_dir / f"{stem}.jpg")
    mask = Image.new("L", (4, 2), 0)
    mask.putpixel((0, 0), 255)
    mask.save(mask_dir / f"{stem}.png")


def test_cod_dataset_loads_resized_binary_sample(tmp_path: Path):
    _write_pair(tmp_path, "sample")
    dataset = CODDirectoryDataset(
        tmp_path, dataset_name="CAMO", split="train", image_size=(8, 8)
    )

    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["image"].shape == (3, 8, 8)
    assert sample["mask"].shape == (1, 8, 8)
    assert set(torch.unique(sample["mask"]).tolist()) <= {0.0, 1.0}
    assert sample["dataset"] == "CAMO"
    assert sample["split"] == "train"


def test_cod_dataset_applies_synchronized_forced_flips(tmp_path: Path):
    _write_pair(tmp_path, "sample")
    dataset = CODDirectoryDataset(
        tmp_path,
        dataset_name="COD10K",
        split="train",
        image_size=(2, 4),
        training=True,
        horizontal_flip_probability=1.0,
        vertical_flip_probability=1.0,
    )

    sample = dataset[0]

    assert sample["mask"][0, -1, -1] == 1
    assert sample["image"][0, -1, -1] > 0.9


def test_cod_dataset_rejects_unmatched_files(tmp_path: Path):
    _write_pair(tmp_path, "sample")
    (tmp_path / "masks" / "sample.png").unlink()

    with pytest.raises(ValueError, match="CAMO.*unmatched"):
        CODDirectoryDataset(tmp_path, dataset_name="CAMO", split="test")


def test_cod_dataset_rejects_duplicate_casefolded_stems(tmp_path: Path):
    _write_pair(tmp_path, "sample")
    Image.new("RGB", (4, 2), "white").save(tmp_path / "images" / "sample.png")

    with pytest.raises(ValueError, match="CAMO.*duplicate"):
        CODDirectoryDataset(tmp_path, dataset_name="CAMO", split="test")


def test_cod_dataset_rejects_empty_directories(tmp_path: Path):
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()

    with pytest.raises(ValueError, match="CAMO.*no image-mask pairs"):
        CODDirectoryDataset(tmp_path, dataset_name="CAMO", split="test")


def test_cod_dataset_rejects_missing_directories(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="CAMO.*missing images directory"):
        CODDirectoryDataset(tmp_path, dataset_name="CAMO", split="test")
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `pytest -q tests/test_cod_dataset.py`

Expected: collection fails with `ModuleNotFoundError: No module named 'sam2unet.cod_dataset'`.

- [ ] **Step 3: Implement the directory-backed dataset**

Create `src/sam2unet/cod_dataset.py` with:

```python
"""Directory-backed camouflaged-object segmentation datasets."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


class CODDirectoryDataset(Dataset):
    """Load strictly paired COD images and binary masks from one dataset root."""

    def __init__(
        self,
        root: Path,
        dataset_name: str,
        split: str,
        image_size: Tuple[int, int] = (352, 352),
        training: bool = False,
        horizontal_flip_probability: float = 0.5,
        vertical_flip_probability: float = 0.5,
        image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.split = split
        self.image_size = tuple(image_size)
        self.training = training
        self.horizontal_flip_probability = horizontal_flip_probability
        self.vertical_flip_probability = vertical_flip_probability
        self.image_mean = image_mean
        self.image_std = image_std
        images = self._index_files(self.root / "images", "images")
        masks = self._index_files(self.root / "masks", "masks")
        if images.keys() != masks.keys():
            image_only = sorted(images.keys() - masks.keys())
            mask_only = sorted(masks.keys() - images.keys())
            raise ValueError(
                f"{dataset_name}: unmatched image-mask files; "
                f"images_without_masks={image_only[:5]}, "
                f"masks_without_images={mask_only[:5]}"
            )
        self.pairs = [(images[key], masks[key]) for key in sorted(images)]
        if not self.pairs:
            raise ValueError(f"{dataset_name}: no image-mask pairs under {self.root}")

    def _index_files(self, directory: Path, role: str) -> Dict[str, Path]:
        if not directory.is_dir():
            raise FileNotFoundError(
                f"{self.dataset_name}: missing {role} directory: {directory}"
            )
        indexed: Dict[str, Path] = {}
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                key = path.stem.casefold()
                if key in indexed:
                    raise ValueError(
                        f"{self.dataset_name}: duplicate sample stem in {directory}: "
                        f"{path.stem}"
                    )
                indexed[key] = path
        return indexed

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, object]:
        image_path, mask_path = self.pairs[index]
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        with Image.open(mask_path) as source_mask:
            mask = source_mask.convert("L")
        if self.training:
            if random.random() < self.horizontal_flip_probability:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() < self.vertical_flip_probability:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
        image = TF.resize(
            image, self.image_size, interpolation=InterpolationMode.BILINEAR
        )
        mask = TF.resize(mask, self.image_size, interpolation=InterpolationMode.NEAREST)
        image_tensor = TF.normalize(TF.to_tensor(image), self.image_mean, self.image_std)
        mask_tensor = (TF.to_tensor(mask) >= 0.5).to(torch.float32)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "dataset": self.dataset_name,
            "split": self.split,
            "image_path": str(image_path),
        }


__all__ = ["CODDirectoryDataset"]
```

- [ ] **Step 4: Run the focused tests and confirm GREEN**

Run: `pytest -q tests/test_cod_dataset.py`

Expected: `6 passed`.

- [ ] **Step 5: Commit the dataset unit**

```powershell
git add src/sam2unet/cod_dataset.py tests/test_cod_dataset.py
git commit -m "feat: add COD directory dataset"
```

### Task 2: Dispatch COD Loaders From The Shared Experiment Pipeline

**Files:**
- Modify: `src/sam2unet/experiment.py:15-125`
- Modify: `tests/test_experiment_cli.py:1-80`

- [ ] **Step 1: Add a failing loader-dispatch test**

In `tests/test_experiment_cli.py`, add `import pytest`, change the experiment import
to `from sam2unet.experiment import build_loaders, main`, and add:

```python
def _make_cod_root(root: Path) -> Path:
    image_dir = root / "images"
    mask_dir = root / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    Image.new("RGB", (32, 32), "white").save(image_dir / "sample.jpg")
    Image.new("L", (32, 32), 255).save(mask_dir / "sample.png")
    return root


def test_build_loaders_combines_cod_training_sets_and_names_tests(tmp_path: Path):
    train_sets = {}
    for name in ("CAMO", "COD10K"):
        root = _make_cod_root(tmp_path / "train" / name)
        train_sets[name] = str(root)
    test_sets = {}
    for name in ("CAMO", "COD10K", "NC4K", "CHAMELEON"):
        root = _make_cod_root(tmp_path / "test" / name)
        test_sets[name] = str(root)
    config = {
        "dataset_type": "cod_directory",
        "train_sets": train_sets,
        "test_sets": test_sets,
        "input_size": [16, 16],
        "batch_size": 1,
        "num_workers": 0,
    }

    train_loader, test_loaders = build_loaders(config, tmp_path / "config.json")

    assert len(train_loader.dataset) == 2
    assert set(test_loaders) == {"CAMO", "COD10K", "NC4K", "CHAMELEON"}
    assert {test_loaders[name].dataset.dataset_name for name in test_loaders} == set(
        test_loaders
    )
```

Also add:

```python
def test_build_loaders_rejects_unknown_dataset_type(tmp_path: Path):
    with pytest.raises(ValueError, match="unsupported dataset_type"):
        build_loaders({"dataset_type": "unknown"}, tmp_path / "config.json")
```

In the same RED phase, add the complete synthetic workflow test:

```python
def test_cod_smoke_command_writes_all_evaluations(tmp_path: Path):
    train_sets = {
        name: str(_make_cod_root(tmp_path / "train" / name))
        for name in ("CAMO", "COD10K")
    }
    test_sets = {
        name: str(_make_cod_root(tmp_path / "test" / name))
        for name in ("CAMO", "COD10K", "NC4K", "CHAMELEON")
    }
    output = tmp_path / "output"
    config_path = tmp_path / "cod-config.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_type": "cod_directory",
                "train_sets": train_sets,
                "test_sets": test_sets,
                "input_size": [32, 32],
                "epochs": 1,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.001,
                "max_grad_norm": 1.0,
                "seed": 7,
                "num_workers": 0,
                "checkpoint_path": "missing.pt",
                "model_cfg": "configs/sam2/sam2_hiera_l.yaml",
                "output_root": str(output),
                "smoke": {
                    "train_samples": 2,
                    "test_samples": 1,
                    "feature_channels": 8,
                    "attempt_real_forward": False,
                },
            }
        ),
        encoding="utf-8",
    )

    assert main(["smoke", "--config", str(config_path), "--device", "cpu"]) == 0

    report = json.loads((output / "smoke" / "smoke_report.json").read_text())
    assert report["workflow"]["status"] == "passed"
    assert set(report["workflow"]["resumed_evaluation"]) == {
        "CAMO",
        "COD10K",
        "NC4K",
        "CHAMELEON",
    }
    assert (output / "smoke" / "latest.pt").is_file()
```

- [ ] **Step 2: Run the dispatch tests and confirm RED**

Run: `pytest -q tests/test_experiment_cli.py -k "build_loaders or cod_smoke"`

Expected: both COD tests FAIL because `build_loaders()` still requires
`train_manifest` and has no `cod_directory` branch.

- [ ] **Step 3: Implement configuration-driven dispatch**

In `src/sam2unet/experiment.py`:

```python
from torch.utils.data import ConcatDataset, DataLoader, Subset

from .cod_dataset import CODDirectoryDataset
```

Extract the current manifest code to `_build_manifest_datasets()`, add
`_build_cod_datasets()`, and make `build_loaders()` select explicitly:

```python
def _build_manifest_datasets(config, config_path, image_size):
    train_dataset = PolypManifestDataset(
        _resolve(config["train_manifest"], config_path),
        image_size=image_size,
        training=True,
    )
    test_datasets = {
        name: PolypManifestDataset(
            _resolve(manifest, config_path), image_size=image_size, training=False
        )
        for name, manifest in dict(config["test_sets"]).items()
    }
    return train_dataset, test_datasets


def _build_cod_datasets(config, config_path, image_size):
    train_datasets = [
        CODDirectoryDataset(
            _resolve(root, config_path),
            dataset_name=name,
            split="train",
            image_size=image_size,
            training=True,
        )
        for name, root in dict(config["train_sets"]).items()
    ]
    if not train_datasets:
        raise ValueError("cod_directory requires at least one train set")
    test_datasets = {
        name: CODDirectoryDataset(
            _resolve(root, config_path),
            dataset_name=name,
            split="test",
            image_size=image_size,
            training=False,
        )
        for name, root in dict(config["test_sets"]).items()
    }
    if not test_datasets:
        raise ValueError("cod_directory requires at least one test set")
    return ConcatDataset(train_datasets), test_datasets
```

At the start of `build_loaders()` select the builder and use the resulting datasets
with the existing limits and `DataLoader` settings:

```python
    dataset_type = str(config.get("dataset_type", "manifest"))
    if dataset_type == "manifest":
        train_dataset, test_datasets = _build_manifest_datasets(
            config, config_path, image_size
        )
    elif dataset_type == "cod_directory":
        train_dataset, test_datasets = _build_cod_datasets(
            config, config_path, image_size
        )
    else:
        raise ValueError(f"unsupported dataset_type: {dataset_type}")
```

Complete `build_loaders()` with the existing limit and loader policy:

```python
    train_loader = DataLoader(
        _limited_dataset(train_dataset, train_limit),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loaders = {
        name: DataLoader(
            _limited_dataset(dataset, test_limit),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
        )
        for name, dataset in test_datasets.items()
    }
    return train_loader, test_loaders
```

- [ ] **Step 4: Run dispatch and Polyp compatibility tests**

Run: `pytest -q tests/test_experiment_cli.py tests/test_polyp_training.py`

Expected: all tests pass, including the COD smoke workflow and the existing config
that omits `dataset_type`.

- [ ] **Step 5: Commit shared experiment dispatch**

```powershell
git add src/sam2unet/experiment.py tests/test_experiment_cli.py
git commit -m "feat: dispatch COD experiment datasets"
```

### Task 3: Add The Real COD Training Configuration

**Files:**
- Create: `configs/cod_train.json`
- Create: `tests/test_cod_config.py`

- [ ] **Step 1: Write a failing portable configuration test**

Create `tests/test_cod_config.py` that parses `configs/cod_train.json` and asserts:

```python
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_cod_training_config_registers_all_validated_splits():
    config = json.loads(
        (PROJECT_ROOT / "configs" / "cod_train.json").read_text(encoding="utf-8")
    )

    assert config["dataset_type"] == "cod_directory"
    assert config["train_sets"] == {
        "CAMO": "data/cod/prepared/train/CAMO",
        "COD10K": "data/cod/prepared/train/COD10K",
    }
    assert set(config["test_sets"]) == {"CAMO", "COD10K", "NC4K", "CHAMELEON"}
    assert config["expected_counts"] == {
        "train/CAMO": 1000,
        "train/COD10K": 3040,
        "test/CAMO": 250,
        "test/COD10K": 2026,
        "test/NC4K": 4121,
        "test/CHAMELEON": 76,
    }
    assert config["output_root"] == "runs/cod"
```

- [ ] **Step 2: Run the config test and confirm RED**

Run: `pytest -q tests/test_cod_config.py`

Expected: FAIL with `FileNotFoundError` because `configs/cod_train.json` is absent.

- [ ] **Step 3: Create the COD configuration**

Create `configs/cod_train.json` with:

```json
{
  "task": "camouflaged_object_segmentation",
  "dataset_type": "cod_directory",
  "data_root": "data/cod/prepared",
  "train_sets": {
    "CAMO": "data/cod/prepared/train/CAMO",
    "COD10K": "data/cod/prepared/train/COD10K"
  },
  "test_sets": {
    "CAMO": "data/cod/prepared/test/CAMO",
    "COD10K": "data/cod/prepared/test/COD10K",
    "NC4K": "data/cod/prepared/test/NC4K",
    "CHAMELEON": "data/cod/prepared/test/CHAMELEON"
  },
  "input_size": [352, 352],
  "epochs": 30,
  "batch_size": 1,
  "gradient_accumulation_steps": 12,
  "num_workers": 0,
  "optimizer": "AdamW",
  "learning_rate": 0.001,
  "weight_decay": 0.01,
  "max_grad_norm": 1.0,
  "seed": 42,
  "checkpoint_path": "checkpoints/sam2_hiera_large.pt",
  "model_cfg": "configs/sam2/sam2_hiera_l.yaml",
  "output_root": "runs/cod",
  "num_experts": 4,
  "metrics": ["dice", "iou", "mae"],
  "bridge_modes": ["rfb", "static", "full"],
  "expected_counts": {
    "train/CAMO": 1000,
    "train/COD10K": 3040,
    "test/CAMO": 250,
    "test/COD10K": 2026,
    "test/NC4K": 4121,
    "test/CHAMELEON": 76
  },
  "smoke": {
    "train_samples": 2,
    "test_samples": 1,
    "feature_channels": 8,
    "attempt_real_forward": false
  }
}
```

- [ ] **Step 4: Run the portable configuration test**

Run: `pytest -q tests/test_cod_config.py`

Expected: `1 passed` without requiring `data/cod/` to exist.

- [ ] **Step 5: Commit the real configuration**

```powershell
git add configs/cod_train.json tests/test_cod_config.py
git commit -m "feat: add COD training configuration"
```

### Task 4: Update Documentation And Run Real-Data Acceptance

**Files:**
- Modify: `README.md:50-59,63-92,200-245`
- Modify: `main.py:1,22-24`
- Modify: `src/sam2unet/experiment.py:1`

- [ ] **Step 1: Update task-neutral descriptions and COD usage docs**

Change module/CLI descriptions from Polyp-only wording to binary segmentation.
In README, remove the planning labels from `cod_train.json` and `cod_dataset.py`,
mark COD training support implemented, and document these commands:

```python
# main.py
"""SAM2-UNet multi-task binary segmentation experiment entrypoint."""

# src/sam2unet/experiment.py
"""Train, evaluate, and smoke-test SAM2-UNet binary segmentation models."""
```

```powershell
python main.py --mode smoke --config configs/cod_train.json --device cpu
python main.py --mode train --config configs/cod_train.json --bridge-mode full --epochs 30
python main.py --mode evaluate --config configs/cod_train.json --bridge-mode full --checkpoint runs/cod/full/best.pt
```

State explicitly that NC4K binary masks are evaluated and instance annotations are
not used by the current binary loss/metrics.

- [ ] **Step 2: Run a real-data count and readability check**

Run:

```powershell
@'
import json
from pathlib import Path
from sam2unet.cod_dataset import CODDirectoryDataset

config = json.loads(Path("configs/cod_train.json").read_text(encoding="utf-8"))
for key, expected in config["expected_counts"].items():
    split, name = key.split("/", 1)
    root = Path(config[f"{split}_sets"][name])
    dataset = CODDirectoryDataset(
        root, dataset_name=name, split=split, image_size=(32, 32)
    )
    if len(dataset) != expected:
        raise SystemExit(f"{key}: expected {expected}, found {len(dataset)}")
    sample = dataset[0]
    assert tuple(sample["image"].shape) == (3, 32, 32)
    assert tuple(sample["mask"].shape) == (1, 32, 32)
    print(f"{key}={len(dataset)}")
'@ | python -
```

Expected output:

```text
train/CAMO=1000
train/COD10K=3040
test/CAMO=250
test/COD10K=2026
test/NC4K=4121
test/CHAMELEON=76
```

- [ ] **Step 3: Run the complete portable test suite**

Run: `pytest -q`

Expected: all tests pass with zero failures.

- [ ] **Step 4: Run the real COD CPU smoke workflow**

Run:

```powershell
python main.py --mode smoke --config configs/cod_train.json --device cpu --limit-train 2 --limit-test 1
```

Expected: exit code 0; `runs/cod/smoke/latest.pt`, `best.pt`, `history.json`, and
`smoke_report.json` exist; the report contains CAMO, COD10K, NC4K, and CHAMELEON.

- [ ] **Step 5: Inspect the final diff**

Run: `git diff --check`

Expected: no whitespace errors. Preserve all pre-existing working-tree changes and
do not stage unrelated files.

- [ ] **Step 6: Commit documentation and any final task-neutral wording changes**

```powershell
git add README.md main.py src/sam2unet/experiment.py
git commit -m "docs: document COD training workflow"
```
