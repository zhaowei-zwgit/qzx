"""Train, evaluate, and smoke-test SAM2-UNet polyp segmentation models."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .fusion import SAM2UNetFusion
from .monitor import TrainingMonitor
from .polyp_dataset import PolypManifestDataset
from .runtime import DEFAULT_MODEL_CONFIG
from .training import (
    evaluate_model,
    load_training_checkpoint,
    save_training_checkpoint,
    train_one_epoch,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SmokeEncoder(nn.Module):
    """Small encoder that preserves the four-level Hiera feature contract."""

    def __init__(self) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Conv2d(3, channels, 1) for channels in (144, 288, 576, 1152)
        )

    def forward(self, x: torch.Tensor):
        height, width = x.shape[-2:]
        return tuple(
            projection(
                F.interpolate(
                    x,
                    size=(height // scale, width // scale),
                    mode="bilinear",
                    align_corners=False,
                )
            )
            for projection, scale in zip(self.projections, (4, 8, 16, 32))
        )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve(path_value: str | Path, config_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    project_path = PROJECT_ROOT / path
    if project_path.exists() or not (config_path.parent / path).exists():
        return project_path
    return config_path.parent / path


def load_config(path: Path) -> Dict[str, object]:
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_config_path"] = str(config_path)
    return config


def _device(device_text: str) -> torch.device:
    if device_text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_text)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _limited_dataset(dataset, limit: int | None):
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def build_loaders(
    config: Mapping[str, object],
    config_path: Path,
    train_limit: int | None = None,
    test_limit: int | None = None,
) -> tuple[DataLoader, Dict[str, DataLoader]]:
    image_size = tuple(config.get("input_size", (352, 352)))
    workers = int(config.get("num_workers", 0))
    batch_size = int(config.get("batch_size", 1))
    train_dataset = PolypManifestDataset(
        _resolve(config["train_manifest"], config_path),
        image_size=image_size,
        training=True,
    )
    train_loader = DataLoader(
        _limited_dataset(train_dataset, train_limit),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loaders = {}
    for name, manifest in dict(config["test_sets"]).items():
        dataset = PolypManifestDataset(
            _resolve(manifest, config_path), image_size=image_size, training=False
        )
        test_loaders[name] = DataLoader(
            _limited_dataset(dataset, test_limit),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
        )
    return train_loader, test_loaders


def build_model(
    config: Mapping[str, object],
    device: torch.device,
    bridge_mode: str,
    smoke: bool = False,
) -> SAM2UNetFusion:
    kwargs = {
        "bridge_mode": bridge_mode,
        "num_experts": int(config.get("num_experts", 4)),
        "model_cfg": str(config.get("model_cfg", DEFAULT_MODEL_CONFIG)),
        "sam_device": str(device),
    }
    if smoke:
        smoke_config = dict(config.get("smoke", {}))
        kwargs.update(
            {
                "encoder": SmokeEncoder(),
                "feature_channels": int(smoke_config.get("feature_channels", 8)),
            }
        )
    else:
        config_path = Path(str(config["_config_path"]))
        kwargs["checkpoint_path"] = _resolve(config["checkpoint_path"], config_path)
    return SAM2UNetFusion(**kwargs).to(device)


def _optimizer_and_scheduler(model: nn.Module, config: Mapping[str, object], epochs: int):
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config.get("learning_rate", 0.001)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    return optimizer, scheduler


def _evaluate_sets(
    model: nn.Module, loaders: Mapping[str, DataLoader], device: torch.device
) -> Dict[str, Dict[str, float]]:
    return {name: evaluate_model(model, loader, device) for name, loader in loaders.items()}


def run_training(
    config: Mapping[str, object],
    model: nn.Module,
    train_loader: DataLoader,
    test_loaders: Mapping[str, DataLoader],
    device: torch.device,
    output_dir: Path,
    epochs: int,
    resume: Path | None = None,
    monitor_enabled: bool = True,
) -> Dict[str, object]:
    optimizer, scheduler = _optimizer_and_scheduler(model, config, epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    start_epoch = 0
    best_score = float("-inf")
    history = []
    if resume is not None:
        state = load_training_checkpoint(
            resume, model, optimizer, scheduler, scaler, map_location=device
        )
        start_epoch = int(state.get("epoch", 0))
        best_score = float(state.get("best_score", best_score))
        history = list(state.get("history", []))

    output_dir.mkdir(parents=True, exist_ok=True)
    monitor = TrainingMonitor(output_dir, enabled=monitor_enabled)

    # 训练集总 batch 数，用于进度条
    total_batches = len(train_loader)

    for epoch in range(start_epoch + 1, epochs + 1):
        monitor.on_epoch_start(epoch, total_batches)

        current_lr = optimizer.param_groups[0]["lr"]

        def _batch_callback(batch_idx: int, loss: float, lr: float) -> None:
            monitor.on_batch_end(epoch, batch_idx, loss, lr)

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
            max_grad_norm=float(config.get("max_grad_norm", 1.0)),
            scaler=scaler,
            on_batch_callback=_batch_callback,
        )
        evaluations = _evaluate_sets(model, test_loaders, device)
        average_dice = sum(item["dice"] for item in evaluations.values()) / len(
            evaluations
        )
        scheduler.step()

        monitor.on_epoch_end(epoch, train_loss, evaluations, average_dice)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "average_dice": average_dice,
            "evaluation": evaluations,
        }
        history.append(record)
        model_config = (
            model.get_config() if hasattr(model, "get_config") else {"type": type(model).__name__}
        )
        save_training_checkpoint(
            output_dir / "latest.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            max(best_score, average_dice),
            history,
            model_config,
            scaler,
        )
        if average_dice > best_score:
            best_score = average_dice
            save_training_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_score,
                history,
                model_config,
                scaler,
            )
        _write_json(output_dir / "history.json", history)

    monitor.on_train_end()
    return {"best_score": best_score, "history": history}


def _common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bridge-mode", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    _common_args(train)
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--resume", type=Path, default=None)
    train.add_argument(
        "--no-monitor",
        action="store_true",
        help="禁用 TensorBoard 实时监控和 tqdm 进度条",
    )
    evaluate = subparsers.add_parser("evaluate")
    _common_args(evaluate)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    smoke = subparsers.add_parser("smoke")
    _common_args(smoke)
    return parser


def _output_root(args, config: Mapping[str, object], suffix: str) -> Path:
    config_path = Path(str(config["_config_path"]))
    root = args.output_root or _resolve(config.get("output_root", "runs/polyps"), config_path)
    return Path(root) / suffix


def _train_command(args, config, device, config_path) -> int:
    bridge_mode = args.bridge_mode or str(config.get("bridge_modes", ["full"])[-1])
    epochs = args.epochs or int(config.get("epochs", 20))
    train_loader, test_loaders = build_loaders(
        config, config_path, args.limit_train, args.limit_test
    )
    model = build_model(config, device, bridge_mode)
    output_dir = _output_root(args, config, bridge_mode)
    monitor_enabled = not getattr(args, "no_monitor", False)
    summary = run_training(
        config, model, train_loader, test_loaders, device, output_dir, epochs, args.resume,
        monitor_enabled=monitor_enabled,
    )
    _write_json(output_dir / "training_summary.json", summary)
    return 0


def _evaluate_command(args, config, device, config_path) -> int:
    bridge_mode = args.bridge_mode or str(config.get("bridge_modes", ["full"])[-1])
    _, test_loaders = build_loaders(config, config_path, 1, args.limit_test)
    model = build_model(config, device, bridge_mode)
    load_training_checkpoint(args.checkpoint, model, map_location=device)
    results = _evaluate_sets(model, test_loaders, device)
    _write_json(_output_root(args, config, bridge_mode) / "evaluation.json", results)
    print(json.dumps(results, indent=2))
    return 0


def _attempt_real_forward(config, config_path, device) -> Dict[str, object]:
    smoke_config = dict(config.get("smoke", {}))
    if not smoke_config.get("attempt_real_forward", True):
        return {"status": "skipped"}
    try:
        model = build_model(config, device, "static", smoke=False)
        model.eval()
        size = tuple(config.get("input_size", (352, 352)))
        with torch.no_grad():
            outputs = model(torch.zeros(1, 3, *size, device=device))
        shapes = [list(output.shape) for output in outputs]
        return {"status": "passed", "output_shapes": shapes}
    except Exception as exc:
        return {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _smoke_command(args, config, device, config_path) -> int:
    smoke_config = dict(config.get("smoke", {}))
    train_limit = args.limit_train or int(smoke_config.get("train_samples", 2))
    test_limit = args.limit_test or int(smoke_config.get("test_samples", 1))
    train_loader, test_loaders = build_loaders(
        config, config_path, train_limit=train_limit, test_limit=test_limit
    )
    output_dir = _output_root(args, config, "smoke")
    model = build_model(config, device, "static", smoke=True)
    run_training(
        config, model, train_loader, test_loaders, device, output_dir, epochs=1,
        monitor_enabled=False,
    )

    restored = build_model(config, device, "static", smoke=True)
    load_training_checkpoint(output_dir / "latest.pt", restored, map_location=device)
    resumed_evaluation = _evaluate_sets(restored, test_loaders, device)
    report = {
        "workflow": {"status": "passed", "resumed_evaluation": resumed_evaluation},
        "real_forward": _attempt_real_forward(config, config_path, device),
    }
    _write_json(output_dir / "smoke_report.json", report)
    print(json.dumps(report, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    config_path = Path(str(config["_config_path"]))
    device = _device(args.device)
    _seed_everything(int(config.get("seed", 42)))
    if args.command == "train":
        return _train_command(args, config, device, config_path)
    if args.command == "evaluate":
        return _evaluate_command(args, config, device, config_path)
    return _smoke_command(args, config, device, config_path)


if __name__ == "__main__":
    raise SystemExit(main())
