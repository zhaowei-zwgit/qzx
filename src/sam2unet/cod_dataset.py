"""Directory-backed COD segmentation datasets."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


SUPPORTED_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


class CODDirectoryDataset(Dataset):
    """Load paired COD images and masks from a directory tree."""

    def __init__(
        self,
        root: Path,
        dataset_name: str,
        split: str,
        image_size: Tuple[int, int] = (352, 352),
        training: bool = False,
        horizontal_flip_probability: float = 0.5,
        vertical_flip_probability: float = 0.5,
        image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.split = split
        self.image_size = tuple(image_size)
        self.training = training
        self.horizontal_flip_probability = horizontal_flip_probability
        self.vertical_flip_probability = vertical_flip_probability
        self.image_mean = image_mean
        self.image_std = image_std

        self.image_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"{self.dataset_name}: missing images directory at {self.image_dir}"
            )
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(
                f"{self.dataset_name}: missing masks directory at {self.mask_dir}"
            )

        self._pairs = self._index_pairs()
        if not self._pairs:
            raise ValueError(f"{self.dataset_name}: no image-mask pairs found")

    @staticmethod
    def _format_stem_list(stems: list[str], limit: int = 8) -> str:
        preview = stems[:limit]
        rendered = ", ".join(repr(stem) for stem in preview)
        if len(stems) > limit:
            rendered = f"{rendered}, ... (+{len(stems) - limit} more)"
        return f"[{rendered}]"

    def _collect_files(self, directory: Path) -> Dict[str, Path]:
        indexed: Dict[str, Path] = {}
        for path in sorted(directory.iterdir(), key=lambda candidate: candidate.name.casefold()):
            if not path.is_file():
                continue
            if path.suffix.casefold() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            stem = path.stem.casefold()
            if stem in indexed:
                raise ValueError(
                    f"{self.dataset_name}: duplicate case-insensitive stem in {directory.name}: {path.stem}"
                )
            indexed[stem] = path
        return indexed

    def _index_pairs(self) -> list[tuple[str, Path, Path]]:
        image_files = self._collect_files(self.image_dir)
        mask_files = self._collect_files(self.mask_dir)

        if not image_files and not mask_files:
            return []

        paired_stems = sorted(image_files.keys() & mask_files.keys())
        if len(paired_stems) != len(image_files) or len(paired_stems) != len(mask_files):
            image_only = sorted(set(image_files) - set(mask_files))
            mask_only = sorted(set(mask_files) - set(image_files))
            raise ValueError(
                f"{self.dataset_name}: unmatched image and mask files "
                f"(image-only stems={self._format_stem_list(image_only)}, "
                f"mask-only stems={self._format_stem_list(mask_only)})"
            )

        return [
            (stem, image_files[stem], mask_files[stem])
            for stem in paired_stems
        ]

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> Dict[str, object]:
        _, image_path, mask_path = self._pairs[index]
        with Image.open(image_path) as source_image, Image.open(mask_path) as source_mask:
            image = source_image.convert("RGB")
            mask = source_mask.convert("L")

        if self.training:
            if random.random() < self.horizontal_flip_probability:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() < self.vertical_flip_probability:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        # Official COD benchmarks include a few source pairs with different image/mask sizes,
        # so both are normalized independently to the configured input size here.
        image = TF.resize(image, self.image_size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.image_size, interpolation=InterpolationMode.NEAREST)
        image_tensor = TF.normalize(TF.to_tensor(image), self.image_mean, self.image_std)
        mask_tensor = (TF.to_tensor(mask) >= 0.5).to(torch.float32)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "dataset": self.dataset_name,
            "split": self.split,
            "image_path": str(image_path),
        }


__all__ = ["CODDirectoryDataset"]
