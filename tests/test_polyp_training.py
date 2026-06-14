import csv
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from sam2unet.polyp_dataset import PolypManifestDataset
from sam2unet.training import (
    SegmentationMetrics,
    deep_supervision_loss,
    evaluate_model,
    load_training_checkpoint,
    save_training_checkpoint,
    soft_iou_loss,
    train_one_epoch,
)


def _write_manifest_sample(root: Path) -> Path:
    image_path = root / "images" / "sample.png"
    mask_path = root / "masks" / "sample.png"
    image_path.parent.mkdir(parents=True)
    mask_path.parent.mkdir(parents=True)
    image = Image.new("RGB", (4, 2), "black")
    image.putpixel((0, 0), (255, 0, 0))
    image.save(image_path)
    mask = Image.new("L", (4, 2), 0)
    mask.putpixel((0, 0), 255)
    mask.save(mask_path)
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=("dataset", "split", "image", "mask")
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "test",
                "split": "train",
                "image": image_path.relative_to(root).as_posix(),
                "mask": mask_path.relative_to(root).as_posix(),
            }
        )
    return manifest


def test_manifest_dataset_loads_resized_rgb_image_and_binary_mask(tmp_path: Path):
    manifest = _write_manifest_sample(tmp_path)
    dataset = PolypManifestDataset(manifest, image_size=(8, 8))

    sample = dataset[0]

    assert sample["image"].shape == (3, 8, 8)
    assert sample["mask"].shape == (1, 8, 8)
    assert set(torch.unique(sample["mask"]).tolist()) <= {0.0, 1.0}
    assert sample["dataset"] == "test"


def test_manifest_dataset_applies_imagenet_normalization(tmp_path: Path):
    manifest = _write_manifest_sample(tmp_path)
    dataset = PolypManifestDataset(manifest, image_size=(2, 4))

    image = dataset[0]["image"]

    assert image[0, 0, 0].item() == pytest.approx((1.0 - 0.485) / 0.229)
    assert image[1, 0, 0].item() == pytest.approx((0.0 - 0.456) / 0.224)


def test_manifest_dataset_applies_synchronized_forced_flips(tmp_path: Path):
    manifest = _write_manifest_sample(tmp_path)
    dataset = PolypManifestDataset(
        manifest,
        image_size=(2, 4),
        training=True,
        horizontal_flip_probability=1.0,
        vertical_flip_probability=1.0,
    )

    sample = dataset[0]

    assert sample["mask"][0, -1, -1] == 1
    assert sample["image"][0, -1, -1] > 0.9


def test_manifest_dataset_resolves_paths_relative_to_manifest_parent_parent(
    tmp_path: Path,
):
    prepared = tmp_path / "prepared"
    manifest = _write_manifest_sample(prepared)
    manifests = prepared / "manifests"
    manifests.mkdir()
    nested_manifest = manifests / "train.csv"
    manifest.replace(nested_manifest)

    dataset = PolypManifestDataset(nested_manifest, image_size=(8, 8))

    assert dataset[0]["image"].shape == (3, 8, 8)


def test_soft_iou_and_deep_supervision_losses_are_finite_and_differentiable():
    target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    primary = torch.zeros_like(target, requires_grad=True)
    outputs = (primary, primary + 0.1, primary - 0.1)

    iou_loss = soft_iou_loss(primary, target)
    total = deep_supervision_loss(outputs, target)
    total.backward()

    assert torch.isfinite(iou_loss)
    assert torch.isfinite(total)
    assert primary.grad is not None
    assert total > iou_loss


def test_segmentation_metrics_report_perfect_predictions():
    metrics = SegmentationMetrics()
    logits = torch.tensor([[[[10.0, -10.0], [-10.0, 10.0]]]])
    target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])

    metrics.update(logits, target)
    result = metrics.compute()

    assert result["dice"] == 1.0
    assert result["iou"] == 1.0
    assert result["mae"] < 0.001


class TinySegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        output = self.conv(x)
        return output, output, output


def _tiny_loader():
    images = torch.rand(2, 3, 8, 8)
    masks = (images[:, :1] > 0.5).float()
    return DataLoader(TensorDataset(images, masks), batch_size=1)


def test_train_evaluate_and_checkpoint_round_trip(tmp_path: Path):
    model = TinySegmentationModel()
    initial = model.conv.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)

    loss = train_one_epoch(
        model,
        _tiny_loader(),
        optimizer,
        device=torch.device("cpu"),
        accumulation_steps=2,
        max_grad_norm=1.0,
    )
    metrics = evaluate_model(model, _tiny_loader(), device=torch.device("cpu"))
    checkpoint = tmp_path / "training.pt"
    save_training_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        epoch=3,
        best_score=0.75,
        history=[{"epoch": 3}],
        model_config={"bridge_mode": "smoke"},
    )
    restored = TinySegmentationModel()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        restored_optimizer, T_max=2
    )
    state = load_training_checkpoint(
        checkpoint, restored, restored_optimizer, restored_scheduler
    )

    assert loss > 0
    assert not torch.equal(initial, model.conv.weight)
    assert metrics.keys() == {"dice", "iou", "mae"}
    assert state["epoch"] == 3
    assert state["best_score"] == 0.75
    torch.testing.assert_close(restored.conv.weight, model.conv.weight)
