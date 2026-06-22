from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_cod_training_config_registers_all_validated_splits():
    config = json.loads(
        (PROJECT_ROOT / "configs" / "cod_train.json").read_text(encoding="utf-8")
    )

    assert config["task"] == "camouflaged_object_segmentation"
    assert config["dataset_type"] == "cod_directory"
    assert config["data_root"] == "data/cod/prepared"
    assert config["train_sets"] == {
        "CAMO": "data/cod/prepared/train/CAMO",
        "COD10K": "data/cod/prepared/train/COD10K",
    }
    assert config["test_sets"] == {
        "CAMO": "data/cod/prepared/test/CAMO",
        "COD10K": "data/cod/prepared/test/COD10K",
        "NC4K": "data/cod/prepared/test/NC4K",
        "CHAMELEON": "data/cod/prepared/test/CHAMELEON",
    }
    assert config["expected_counts"] == {
        "train/CAMO": 1000,
        "train/COD10K": 3040,
        "test/CAMO": 250,
        "test/COD10K": 2026,
        "test/NC4K": 4121,
        "test/CHAMELEON": 76,
    }
    assert config["input_size"] == [352, 352]
    assert config["epochs"] == 30
    assert config["batch_size"] == 1
    assert config["gradient_accumulation_steps"] == 12
    assert config["num_workers"] == 0
    assert config["optimizer"] == "AdamW"
    assert config["learning_rate"] == 0.001
    assert config["weight_decay"] == 0.01
    assert config["max_grad_norm"] == 1.0
    assert config["seed"] == 42
    assert config["checkpoint_path"] == "checkpoints/sam2_hiera_large.pt"
    assert config["model_cfg"] == "configs/sam2/sam2_hiera_l.yaml"
    assert config["output_root"] == "runs/cod"
    assert config["num_experts"] == 4
    assert config["metrics"] == ["dice", "iou", "mae"]
    assert config["bridge_modes"] == ["rfb", "static", "full"]
    assert config["smoke"]["train_samples"] == 2
    assert config["smoke"]["test_samples"] == 1
    assert config["smoke"]["feature_channels"] == 8
    assert config["smoke"]["attempt_real_forward"] is False
