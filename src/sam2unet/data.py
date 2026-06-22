"""Prepare and validate PraNet-format polyp segmentation datasets."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from PIL import Image


SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
TEST_DATASETS = {
    "Kvasir": "Kvasir",
    "CVC-ClinicDB": "CVC-ClinicDB",
    "CVC-300": "CVC-300",
    "CVC-ColonDB": "CVC-ColonDB",
    "ETIS-LaribPolypDB": "ETIS",
}
Pair = Tuple[Path, Path]


def _indexed_files(directory: Path) -> Dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"missing directory: {directory}")
    indexed: Dict[str, Path] = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            key = path.stem.casefold()
            if key in indexed:
                raise ValueError(f"duplicate sample stem in {directory}: {path.stem}")
            indexed[key] = path
    return indexed


def _verify_image(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        raise ValueError(f"unreadable image: {path}") from exc


def discover_pairs(image_dir: Path, mask_dir: Path) -> list[Pair]:
    """Return sorted, readable image-mask pairs matched by filename stem."""
    images = _indexed_files(Path(image_dir))
    masks = _indexed_files(Path(mask_dir))
    if images.keys() != masks.keys():
        image_only = sorted(images.keys() - masks.keys())
        mask_only = sorted(masks.keys() - images.keys())
        raise ValueError(
            f"unmatched image-mask files; images_without_masks={image_only[:5]}, "
            f"masks_without_images={mask_only[:5]}"
        )
    pairs = [(images[key], masks[key]) for key in sorted(images)]
    if not pairs:
        raise ValueError(f"no image-mask pairs found in {image_dir} and {mask_dir}")
    for image_path, mask_path in pairs:
        _verify_image(image_path)
        _verify_image(mask_path)
    return pairs


def _find_directory(root: Path, candidates: Sequence[str]) -> Path:
    for candidate in candidates:
        path = root / candidate
        if path.is_dir():
            return path
    raise FileNotFoundError(
        f"none of the expected directories exist under {root}: {list(candidates)}"
    )


def _discover_pairs_from_root(
    root: Path,
    image_candidates: Sequence[str] = ("images", "image", "Original"),
    mask_candidates: Sequence[str] = ("masks", "mask", "Ground Truth"),
) -> list[Pair]:
    root = Path(root)
    return discover_pairs(
        _find_directory(root, image_candidates),
        _find_directory(root, mask_candidates),
    )


def _require_empty_output(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"output directory must be empty: {output_root}")


def _write_manifest(
    output_root: Path,
    name: str,
    pairs: Iterable[Pair],
    dataset: str | Mapping[str, str],
    split: str,
) -> None:
    manifest_path = output_root / "manifests" / f"{name}.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=("dataset", "split", "image", "mask")
        )
        writer.writeheader()
        for image_path, mask_path in pairs:
            source_dataset = (
                dataset[image_path.stem.casefold()]
                if isinstance(dataset, Mapping)
                else dataset
            )
            writer.writerow(
                {
                    "dataset": source_dataset,
                    "split": split,
                    "image": image_path.relative_to(output_root).as_posix(),
                    "mask": mask_path.relative_to(output_root).as_posix(),
                }
            )


def _copy_pairs(pairs: Iterable[Pair], image_dir: Path, mask_dir: Path) -> list[Pair]:
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Pair] = []
    for source_image, source_mask in pairs:
        target_image = image_dir / source_image.name
        target_mask = mask_dir / source_mask.name
        shutil.copy2(source_image, target_image)
        shutil.copy2(source_mask, target_mask)
        copied.append((target_image, target_mask))
    return copied


def prepare_pranet_polyp_tests(
    test_source: Path,
    output_root: Path,
    datasets: Sequence[str] | None = None,
) -> Dict[str, int]:
    """Add selected PraNet test sets without replacing prepared training data."""
    test_source = Path(test_source)
    output_root = Path(output_root)
    source_by_output = {
        output_dataset: source_dataset
        for source_dataset, output_dataset in TEST_DATASETS.items()
    }
    requested = tuple(datasets) if datasets is not None else tuple(source_by_output)
    unknown = sorted(set(requested) - set(source_by_output))
    if unknown:
        raise ValueError(f"unsupported PraNet test datasets: {unknown}")

    summary: Dict[str, int] = {}
    for output_dataset in requested:
        source_dataset = source_by_output[output_dataset]
        target_root = output_root / "test" / output_dataset
        _require_empty_output(target_root)
        pairs = discover_pairs(
            test_source / source_dataset / "images",
            test_source / source_dataset / "masks",
        )
        copied = _copy_pairs(
            pairs,
            target_root / "images",
            target_root / "masks",
        )
        _write_manifest(
            output_root,
            f"test-{output_dataset}",
            copied,
            dataset=output_dataset,
            split="test",
        )
        summary[f"test/{output_dataset}"] = len(copied)
    return summary


def prepare_pranet_polyp_data(
    train_source: Path,
    test_source: Path,
    output_root: Path,
) -> Dict[str, int]:
    """Normalize PraNet's official polyp training and selected test datasets."""
    train_source = Path(train_source)
    test_source = Path(test_source)
    output_root = Path(output_root)
    _require_empty_output(output_root)

    train_pairs = _discover_pairs_from_root(train_source)
    copied_train = _copy_pairs(
        train_pairs,
        output_root / "train" / "images",
        output_root / "train" / "masks",
    )
    _write_manifest(
        output_root,
        "train",
        copied_train,
        dataset="Kvasir-SEG+CVC-ClinicDB",
        split="train",
    )

    summary = {"train": len(copied_train)}
    summary.update(prepare_pranet_polyp_tests(test_source, output_root))
    return summary


def prepare_polyp_data_from_full_sources(
    official_train_source: Path,
    kvasir_source: Path,
    clinic_source: Path,
    output_root: Path,
) -> Dict[str, int]:
    """Build PraNet's exact test split as the complement of its official train set."""
    output_root = Path(output_root)
    _require_empty_output(output_root)
    official_train_pairs = _discover_pairs_from_root(Path(official_train_source))
    full_sources = {
        "Kvasir": _discover_pairs_from_root(Path(kvasir_source)),
        "CVC-ClinicDB": _discover_pairs_from_root(Path(clinic_source)),
    }

    train_stems = {image_path.stem.casefold() for image_path, _ in official_train_pairs}
    full_stems = {
        dataset: {image_path.stem.casefold() for image_path, _ in pairs}
        for dataset, pairs in full_sources.items()
    }
    dataset_by_stem = {
        stem: dataset for dataset, stems in full_stems.items() for stem in stems
    }
    unknown_train_stems = train_stems - set().union(*full_stems.values())
    if unknown_train_stems:
        raise ValueError(
            f"official training samples are absent from full datasets: "
            f"{sorted(unknown_train_stems)[:5]}"
        )

    copied_train = _copy_pairs(
        official_train_pairs,
        output_root / "train" / "images",
        output_root / "train" / "masks",
    )
    _write_manifest(
        output_root,
        "train",
        copied_train,
        dataset=dataset_by_stem,
        split="train",
    )
    summary = {"train": len(copied_train)}

    for dataset, pairs in full_sources.items():
        test_pairs = [
            pair for pair in pairs if pair[0].stem.casefold() not in train_stems
        ]
        copied_test = _copy_pairs(
            test_pairs,
            output_root / "test" / dataset / "images",
            output_root / "test" / dataset / "masks",
        )
        _write_manifest(
            output_root,
            f"test-{dataset}",
            copied_test,
            dataset=dataset,
            split="test",
        )
        summary[f"test/{dataset}"] = len(copied_test)
    return summary


def validate_prepared_polyp_data(
    output_root: Path,
    expected_counts: Mapping[str, int] | None = None,
) -> Dict[str, int]:
    """Validate all normalized image-mask pairs and optional expected counts."""
    output_root = Path(output_root)
    expected_counts = expected_counts or {}
    locations = {"train": output_root / "train"}
    for dataset in TEST_DATASETS.values():
        name = f"test/{dataset}"
        path = output_root / "test" / dataset
        if (
            dataset in {"Kvasir", "CVC-ClinicDB"}
            or path.is_dir()
            or name in expected_counts
        ):
            locations[name] = path
    summary = {
        name: len(discover_pairs(path / "images", path / "masks"))
        for name, path in locations.items()
    }
    for name, expected in expected_counts.items():
        actual = summary.get(name)
        if actual != expected:
            raise ValueError(f"expected {expected} pairs for {name}, found {actual}")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("train_source", type=Path)
    prepare.add_argument("test_source", type=Path)
    prepare.add_argument("output_root", type=Path)
    prepare_full = subparsers.add_parser("prepare-from-full")
    prepare_full.add_argument("official_train_source", type=Path)
    prepare_full.add_argument("kvasir_source", type=Path)
    prepare_full.add_argument("clinic_source", type=Path)
    prepare_full.add_argument("output_root", type=Path)
    validate = subparsers.add_parser("validate")
    validate.add_argument("output_root", type=Path)
    validate.add_argument("--train-count", type=int, default=1450)
    validate.add_argument("--kvasir-count", type=int, default=100)
    validate.add_argument("--clinic-count", type=int, default=62)
    validate.add_argument("--cvc-300-count", type=int, default=60)
    validate.add_argument("--colon-count", type=int, default=380)
    validate.add_argument("--etis-count", type=int, default=196)
    validate.add_argument(
        "--basic-only",
        action="store_true",
        help="require only the train, Kvasir, and CVC-ClinicDB splits",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        summary = prepare_pranet_polyp_data(
            args.train_source, args.test_source, args.output_root
        )
    elif args.command == "prepare-from-full":
        summary = prepare_polyp_data_from_full_sources(
            args.official_train_source,
            args.kvasir_source,
            args.clinic_source,
            args.output_root,
        )
    else:
        expected_counts = {
            "train": args.train_count,
            "test/Kvasir": args.kvasir_count,
            "test/CVC-ClinicDB": args.clinic_count,
        }
        if not args.basic_only:
            expected_counts.update(
                {
                    "test/CVC-300": args.cvc_300_count,
                    "test/CVC-ColonDB": args.colon_count,
                    "test/ETIS": args.etis_count,
                }
            )
        summary = validate_prepared_polyp_data(
            args.output_root,
            expected_counts=expected_counts,
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
