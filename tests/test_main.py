from pathlib import Path

import pytest

import main as root_main


def test_default_mode_forwards_smoke_and_default_config():
    args = root_main.parse_args([])

    forwarded = root_main.build_experiment_argv(args)

    assert forwarded == [
        "smoke",
        "--config",
        str(root_main.DEFAULT_CONFIG),
        "--device",
        "auto",
    ]


def test_train_mode_forwards_training_options(tmp_path: Path):
    config = tmp_path / "config.json"
    output = tmp_path / "runs"
    resume = tmp_path / "latest.pt"
    args = root_main.parse_args(
        [
            "--mode",
            "train",
            "--config",
            str(config),
            "--device",
            "cpu",
            "--bridge-mode",
            "full",
            "--output-root",
            str(output),
            "--limit-train",
            "10",
            "--limit-test",
            "2",
            "--epochs",
            "3",
            "--resume",
            str(resume),
        ]
    )

    assert root_main.build_experiment_argv(args) == [
        "train",
        "--config",
        str(config),
        "--device",
        "cpu",
        "--bridge-mode",
        "full",
        "--output-root",
        str(output),
        "--limit-train",
        "10",
        "--limit-test",
        "2",
        "--epochs",
        "3",
        "--resume",
        str(resume),
    ]


def test_evaluate_mode_requires_checkpoint():
    with pytest.raises(SystemExit) as error:
        root_main.parse_args(["--mode", "evaluate"])

    assert error.value.code == 2


def test_main_returns_delegated_exit_code(capsys):
    received = []

    def fake_experiment_main(argv):
        received.extend(argv)
        return 7

    result = root_main.main(
        ["--mode", "smoke", "--device", "cpu"],
        experiment_main=fake_experiment_main,
    )

    assert result == 7
    assert received[:4] == [
        "smoke",
        "--config",
        str(root_main.DEFAULT_CONFIG),
        "--device",
    ]
    output = capsys.readouterr().out
    assert "Mode: smoke" in output
    assert "Duration:" in output
