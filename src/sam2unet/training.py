"""Training, evaluation, metrics, and checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_iou_loss(
    logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    dimensions = tuple(range(1, probabilities.ndim))
    intersection = (probabilities * target).sum(dim=dimensions)
    union = (probabilities + target - probabilities * target).sum(dim=dimensions)
    return (1.0 - (intersection + smooth) / (union + smooth)).mean()


def deep_supervision_loss(
    outputs: Sequence[torch.Tensor], target: torch.Tensor
) -> torch.Tensor:
    if not outputs:
        raise ValueError("deep supervision requires at least one output")
    return sum(
        F.binary_cross_entropy_with_logits(logits, target)
        + soft_iou_loss(logits, target)
        for logits in outputs
    )


class SegmentationMetrics:
    def __init__(self, threshold: float = 0.5, epsilon: float = 1e-7) -> None:
        self.threshold = threshold
        self.epsilon = epsilon
        self.reset()

    def reset(self) -> None:
        self.dice_sum = 0.0
        self.iou_sum = 0.0
        self.mae_sum = 0.0
        self.count = 0

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        probabilities = torch.sigmoid(logits.detach())
        predictions = probabilities >= self.threshold
        target_binary = target >= 0.5
        dimensions = tuple(range(1, predictions.ndim))
        intersection = (predictions & target_binary).sum(dim=dimensions).float()
        predicted_sum = predictions.sum(dim=dimensions).float()
        target_sum = target_binary.sum(dim=dimensions).float()
        union = (predictions | target_binary).sum(dim=dimensions).float()
        dice = (2 * intersection + self.epsilon) / (
            predicted_sum + target_sum + self.epsilon
        )
        iou = (intersection + self.epsilon) / (union + self.epsilon)
        mae = (probabilities - target).abs().mean(dim=dimensions)
        self.dice_sum += dice.sum().item()
        self.iou_sum += iou.sum().item()
        self.mae_sum += mae.sum().item()
        self.count += target.shape[0]

    def compute(self) -> Dict[str, float]:
        if not self.count:
            raise ValueError("cannot compute metrics without samples")
        return {
            "dice": self.dice_sum / self.count,
            "iou": self.iou_sum / self.count,
            "mae": self.mae_sum / self.count,
        }


def _unpack_batch(batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, Mapping):
        images, masks = batch["image"], batch["mask"]
    else:
        images, masks = batch[0], batch[1]
    return images.to(device, non_blocking=True), masks.to(device, non_blocking=True)


def _primary_outputs(model_output) -> Sequence[torch.Tensor]:
    if (
        isinstance(model_output, tuple)
        and len(model_output) == 2
        and isinstance(model_output[1], Mapping)
    ):
        model_output = model_output[0]
    if not isinstance(model_output, (tuple, list)):
        return (model_output,)
    return tuple(model_output)


def train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accumulation_steps: int = 1,
    max_grad_norm: float | None = 1.0,
    scaler=None,
) -> float:
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    batches = 0
    use_amp = device.type == "cuda"
    for batch_index, batch in enumerate(loader, start=1):
        images, masks = _unpack_batch(batch, device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = _primary_outputs(model(images))
            loss = deep_supervision_loss(outputs, masks)
            scaled_loss = loss / accumulation_steps
        if scaler is not None and use_amp:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        should_step = batch_index % accumulation_steps == 0
        if should_step:
            if scaler is not None and use_amp:
                scaler.unscale_(optimizer)
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if scaler is not None and use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        loss_sum += loss.detach().item()
        batches += 1

    if not batches:
        raise ValueError("training loader is empty")
    if batches % accumulation_steps:
        if scaler is not None and use_amp:
            scaler.unscale_(optimizer)
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if scaler is not None and use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return loss_sum / batches


@torch.no_grad()
def evaluate_model(
    model: nn.Module, loader: Iterable, device: torch.device
) -> Dict[str, float]:
    model.eval()
    metrics = SegmentationMetrics()
    for batch in loader:
        images, masks = _unpack_batch(batch, device)
        outputs = _primary_outputs(model(images))
        metrics.update(outputs[0], masks)
    return metrics.compute()


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_score: float,
    history: Sequence[Mapping[str, object]],
    model_config: Mapping[str, object],
    scaler=None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_score": best_score,
        "history": list(history),
        "model_config": dict(model_config),
    }
    torch.save(payload, path)


def _load_checkpoint(path: Path, map_location="cpu") -> MutableMapping[str, object]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, MutableMapping):
        raise ValueError("training checkpoint must contain a mapping")
    return payload


def load_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    scaler=None,
    map_location="cpu",
) -> MutableMapping[str, object]:
    payload = _load_checkpoint(Path(path), map_location=map_location)
    if "model_state_dict" not in payload:
        raise ValueError("training checkpoint is missing model_state_dict")
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    return payload


__all__ = [
    "SegmentationMetrics",
    "deep_supervision_loss",
    "evaluate_model",
    "load_training_checkpoint",
    "save_training_checkpoint",
    "soft_iou_loss",
    "train_one_epoch",
]
