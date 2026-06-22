import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from sam2unet.experiment import build_loaders, main


def _make_manifest(root: Path, name: str, split: str) -> Path:
    image = root / f"{name}-image.png"
    mask = root / f"{name}-mask.png"
    Image.new("RGB", (32, 32), "white").save(image)
    Image.new("L", (32, 32), 255).save(mask)
    manifest = root / f"{name}.csv"
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=("dataset", "split", "image", "mask")
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": name,
                "split": split,
                "image": image.name,
                "mask": mask.name,
            }
        )
    return manifest


def _make_cod_root(root: Path) -> Path:
    image_path = root / "images" / "sample.jpg"
    mask_path = root / "masks" / "sample.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), "white").save(image_path)
    Image.new("L", (32, 32), 255).save(mask_path)
    return root


def test_build_loaders_combines_cod_training_sets_and_names_tests(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config = {
        "dataset_type": "cod_directory",
        "train_sets": {
            "CAMO": str(_make_cod_root(tmp_path / "train" / "CAMO")),
            "COD10K": str(_make_cod_root(tmp_path / "train" / "COD10K")),
        },
        "test_sets": {
            "CAMO": str(_make_cod_root(tmp_path / "test" / "CAMO")),
            "COD10K": str(_make_cod_root(tmp_path / "test" / "COD10K")),
            "NC4K": str(_make_cod_root(tmp_path / "test" / "NC4K")),
            "CHAMELEON": str(_make_cod_root(tmp_path / "test" / "CHAMELEON")),
        },
        "input_size": [16, 16],
        "batch_size": 1,
        "num_workers": 0,
    }

    train_loader, test_loaders = build_loaders(config, config_path)

    assert len(train_loader.dataset) == 2
    assert set(test_loaders) == {"CAMO", "COD10K", "NC4K", "CHAMELEON"}
    for name, loader in test_loaders.items():
        assert loader.dataset.dataset_name == name


def test_build_loaders_rejects_unknown_dataset_type(tmp_path: Path):
    with pytest.raises(ValueError, match="unsupported dataset_type"):
        build_loaders({"dataset_type": "mystery"}, tmp_path / "config.json")


def test_smoke_command_writes_training_and_evaluation_artifacts(tmp_path: Path):
    train_manifest = _make_manifest(tmp_path, "train", "train")
    test_manifest = _make_manifest(tmp_path, "test", "test")
    output = tmp_path / "output"
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "train_manifest": str(train_manifest),
                "test_sets": {"test": str(test_manifest)},
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
                    "train_samples": 1,
                    "test_samples": 1,
                    "feature_channels": 8,
                    "attempt_real_forward": False
                }
            }
        ),
        encoding="utf-8",
    )

    assert main(["smoke", "--config", str(config), "--device", "cpu"]) == 0

    assert (output / "smoke" / "latest.pt").is_file()
    assert (output / "smoke" / "best.pt").is_file()
    assert (output / "smoke" / "history.json").is_file()
    report = json.loads((output / "smoke" / "smoke_report.json").read_text())
    assert report["workflow"]["status"] == "passed"


def test_cod_smoke_command_writes_all_evaluations(tmp_path: Path):
    train_roots = {
        "CAMO": str(_make_cod_root(tmp_path / "train" / "CAMO")),
        "COD10K": str(_make_cod_root(tmp_path / "train" / "COD10K")),
    }
    test_roots = {
        "CAMO": str(_make_cod_root(tmp_path / "test" / "CAMO")),
        "COD10K": str(_make_cod_root(tmp_path / "test" / "COD10K")),
        "NC4K": str(_make_cod_root(tmp_path / "test" / "NC4K")),
        "CHAMELEON": str(_make_cod_root(tmp_path / "test" / "CHAMELEON")),
    }
    output = tmp_path / "output"
    model_cfg = tmp_path / "model_cfg.yaml"
    model_cfg.write_text("smoke: true\n", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "dataset_type": "cod_directory",
                "train_sets": train_roots,
                "test_sets": test_roots,
                "input_size": [32, 32],
                "epochs": 1,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.001,
                "max_grad_norm": 1.0,
                "seed": 7,
                "num_workers": 0,
                "checkpoint_path": "missing.pt",
                "model_cfg": str(model_cfg),
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

    assert main(["smoke", "--config", str(config), "--device", "cpu"]) == 0

    report = json.loads((output / "smoke" / "smoke_report.json").read_text())
    assert report["workflow"]["status"] == "passed"
    assert set(report["workflow"]["resumed_evaluation"]) == {
        "CAMO",
        "COD10K",
        "NC4K",
        "CHAMELEON",
    }
    assert (output / "smoke" / "latest.pt").is_file()
