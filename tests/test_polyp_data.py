import csv
from pathlib import Path

import pytest
from PIL import Image

from sam2unet.data import (
    main,
    prepare_polyp_data_from_full_sources,
    prepare_pranet_polyp_data,
    validate_prepared_polyp_data,
)


def _write_pair(root: Path, dataset: str, stem: str) -> None:
    image_dir = root / dataset / "images"
    mask_dir = root / dataset / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), "red").save(image_dir / f"{stem}.png")
    Image.new("L", (8, 8), 255).save(mask_dir / f"{stem}.png")


def _build_pranet_source(root: Path) -> tuple[Path, Path]:
    train = root / "TrainDataset"
    for stem in ("train-a", "train-b", "train-c"):
        _write_pair(root, "TrainDataset", stem)

    test = root / "TestDataset"
    _write_pair(test, "Kvasir", "kvasir-test")
    _write_pair(test, "CVC-ClinicDB", "clinic-test")
    _write_pair(test, "CVC-ColonDB", "ignored-test")
    return train, test


def test_prepare_pranet_polyp_data_keeps_only_requested_test_sets(tmp_path: Path):
    train_source, test_source = _build_pranet_source(tmp_path / "source")
    output = tmp_path / "prepared"

    summary = prepare_pranet_polyp_data(train_source, test_source, output)

    assert summary == {
        "train": 3,
        "test/Kvasir": 1,
        "test/CVC-ClinicDB": 1,
    }
    assert len(list((output / "train" / "images").glob("*.png"))) == 3
    assert len(list((output / "test" / "Kvasir" / "images").glob("*.png"))) == 1
    assert not (output / "test" / "CVC-ColonDB").exists()


def test_prepare_pranet_polyp_data_writes_manifest_and_validates(tmp_path: Path):
    train_source, test_source = _build_pranet_source(tmp_path / "source")
    output = tmp_path / "prepared"

    prepare_pranet_polyp_data(train_source, test_source, output)
    summary = validate_prepared_polyp_data(
        output,
        expected_counts={"train": 3, "test/Kvasir": 1, "test/CVC-ClinicDB": 1},
    )

    assert summary["train"] == 3
    with (output / "manifests" / "train.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows[0].keys() == {"dataset", "split", "image", "mask"}
    assert rows[0]["dataset"] == "Kvasir-SEG+CVC-ClinicDB"
    assert rows[0]["split"] == "train"
    assert len(rows) == 3


def test_prepare_pranet_polyp_data_rejects_missing_mask(tmp_path: Path):
    train_source, test_source = _build_pranet_source(tmp_path / "source")
    (train_source / "masks" / "train-a.png").unlink()

    with pytest.raises(ValueError, match="unmatched"):
        prepare_pranet_polyp_data(train_source, test_source, tmp_path / "prepared")


def test_polyp_data_cli_prepares_and_validates(tmp_path: Path, capsys):
    train_source, test_source = _build_pranet_source(tmp_path / "source")
    output = tmp_path / "prepared"

    assert main(["prepare", str(train_source), str(test_source), str(output)]) == 0
    assert main(["validate", str(output), "--train-count", "3", "--kvasir-count", "1",
                 "--clinic-count", "1"]) == 0
    assert '"train": 3' in capsys.readouterr().out


def test_prepare_from_full_sources_uses_official_train_and_exact_complements(
    tmp_path: Path,
):
    source = tmp_path / "source"
    _write_pair(source, "official", "kvasir-train")
    _write_pair(source, "official", "1")
    (source / "official" / "images").rename(source / "official" / "image")

    _write_pair(source, "Kvasir-SEG", "kvasir-train")
    _write_pair(source, "Kvasir-SEG", "kvasir-test")
    _write_pair(source, "CVC-ClinicDB", "1")
    _write_pair(source, "CVC-ClinicDB", "2")
    output = tmp_path / "prepared"

    summary = prepare_polyp_data_from_full_sources(
        source / "official",
        source / "Kvasir-SEG",
        source / "CVC-ClinicDB",
        output,
    )

    assert summary == {
        "train": 2,
        "test/Kvasir": 1,
        "test/CVC-ClinicDB": 1,
    }
    assert (output / "test" / "Kvasir" / "images" / "kvasir-test.png").is_file()
    assert (output / "test" / "CVC-ClinicDB" / "images" / "2.png").is_file()
    with (output / "manifests" / "train.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert {row["dataset"] for row in rows} == {"Kvasir", "CVC-ClinicDB"}
