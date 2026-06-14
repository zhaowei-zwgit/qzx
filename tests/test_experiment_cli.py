import csv
import json
from pathlib import Path

from PIL import Image

from sam2unet.experiment import main


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
