from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from sam2unet.cod_dataset import CODDirectoryDataset


def _write_rgb_image(path: Path, size: tuple[int, int] = (8, 8)) -> None:
    image = Image.new("RGB", size, "black")
    image.putpixel((0, 0), (255, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _write_mask(path: Path, size: tuple[int, int] = (8, 8)) -> None:
    mask = Image.new("L", size, 0)
    mask.putpixel((0, 0), 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(path)


def test_cod_dataset_loads_case_insensitive_stem_pair_with_mismatched_extensions(
    tmp_path: Path,
):
    root = tmp_path / "camo"
    _write_rgb_image(root / "images" / "SampleOne.JPG")
    _write_mask(root / "masks" / "sampleone.png")

    dataset = CODDirectoryDataset(root, dataset_name="CAMO", split="train", image_size=(8, 8))
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["image"].shape == (3, 8, 8)
    assert sample["mask"].shape == (1, 8, 8)
    assert torch.all((sample["mask"] == 0.0) | (sample["mask"] == 1.0))
    assert sample["dataset"] == "CAMO"
    assert sample["split"] == "train"


def test_cod_dataset_applies_synchronized_forced_flips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "camo"
    _write_rgb_image(root / "images" / "flip.png")
    _write_mask(root / "masks" / "flip.png")

    dataset = CODDirectoryDataset(
        root,
        dataset_name="CAMO",
        split="train",
        image_size=(8, 8),
        training=True,
        horizontal_flip_probability=1.0,
        vertical_flip_probability=1.0,
    )
    monkeypatch.setattr("sam2unet.cod_dataset.random.random", lambda: 0.0)

    sample = dataset[0]
    bottom_right = sample["image"][:, -1, -1]
    expected = torch.tensor(
        [
            (1.0 - 0.485) / 0.229,
            (0.0 - 0.456) / 0.224,
            (0.0 - 0.406) / 0.225,
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(bottom_right, expected, atol=1e-6)
    assert sample["mask"][0, -1, -1] == 1.0


def test_cod_dataset_rejects_unmatched_files(tmp_path: Path):
    root = tmp_path / "camo"
    _write_rgb_image(root / "images" / "only-image.jpg")
    _write_mask(root / "masks" / "other-mask.png")

    with pytest.raises(ValueError, match="CAMO.*unmatched"):
        CODDirectoryDataset(root, dataset_name="CAMO", split="train")


def test_cod_dataset_rejects_duplicate_case_insensitive_stems(tmp_path: Path):
    root = tmp_path / "camo"
    _write_rgb_image(root / "images" / "duplicate.jpg")
    _write_rgb_image(root / "images" / "Duplicate.png")
    _write_mask(root / "masks" / "duplicate.tif")

    with pytest.raises(ValueError, match="CAMO.*duplicate"):
        CODDirectoryDataset(root, dataset_name="CAMO", split="train")


def test_cod_dataset_rejects_empty_directories(tmp_path: Path):
    root = tmp_path / "camo"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir(parents=True)

    with pytest.raises(ValueError, match="CAMO.*no image-mask pairs"):
        CODDirectoryDataset(root, dataset_name="CAMO", split="train")


def test_cod_dataset_rejects_missing_images_directory(tmp_path: Path):
    root = tmp_path / "camo"
    (root / "masks").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="CAMO.*missing images directory"):
        CODDirectoryDataset(root, dataset_name="CAMO", split="train")
