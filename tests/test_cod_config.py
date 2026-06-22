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
    assert config["output_root"] == "runs/cod"
    assert config["smoke"]["attempt_real_forward"] is False
