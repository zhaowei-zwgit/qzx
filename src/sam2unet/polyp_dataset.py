"""Manifest-backed polyp segmentation datasets."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


class PolypManifestDataset(Dataset):
    """Load paired polyp images and masks from a normalized CSV manifest."""

    def __init__(
        self,
        manifest_path: Path,
        image_size: Tuple[int, int] = (352, 352),
        training: bool = False,
        horizontal_flip_probability: float = 0.5,
        vertical_flip_probability: float = 0.5,
        project_root: Path | None = None,
        image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.image_size = tuple(image_size)
        self.training = training
        self.horizontal_flip_probability = horizontal_flip_probability
        self.vertical_flip_probability = vertical_flip_probability
        self.image_mean = image_mean
        self.image_std = image_std
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[2]
        )
        with self.manifest_path.open(newline="", encoding="utf-8") as file:
            self.rows = list(csv.DictReader(file))
        if not self.rows:
            raise ValueError(f"manifest is empty: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, path_text: str) -> Path:
        path = Path(path_text)
        if path.is_absolute():
            return path
        project_path = self.project_root / path
        if project_path.is_file():
            return project_path
        manifest_path = self.manifest_path.parent / path
        if manifest_path.is_file():
            return manifest_path
        dataset_root_path = self.manifest_path.parent.parent / path
        if dataset_root_path.is_file():
            return dataset_root_path
        raise FileNotFoundError(f"manifest path does not exist: {path_text}")

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        with Image.open(self._resolve(row["image"])) as source_image:
            image = source_image.convert("RGB")
        with Image.open(self._resolve(row["mask"])) as source_mask:
            mask = source_mask.convert("L")

        if self.training:
            if random.random() < self.horizontal_flip_probability:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() < self.vertical_flip_probability:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        image = TF.resize(
            image, self.image_size, interpolation=InterpolationMode.BILINEAR
        )
        mask = TF.resize(mask, self.image_size, interpolation=InterpolationMode.NEAREST)
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, self.image_mean, self.image_std)
        mask_tensor = (TF.to_tensor(mask) >= 0.5).to(torch.float32)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "dataset": row.get("dataset", ""),
            "split": row.get("split", ""),
            "image_path": row["image"],
        }


__all__ = ["PolypManifestDataset"]
