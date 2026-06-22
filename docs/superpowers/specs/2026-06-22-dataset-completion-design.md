# Dataset Completion Design

## Goal

Complete every dataset named in the root README while preserving the existing
PraNet polyp training split and making the additional polyp test sets available
to the current evaluation workflow.

## Scope

- Keep the verified 1,450-pair polyp training set unchanged.
- Add CVC-300, CVC-ColonDB, and ETIS-LaribPolypDB from PraNet's official test
  archive to `data/polyps/prepared/test/`.
- Generate one CSV manifest per additional polyp test set and register each
  manifest in `configs/polyp_train.json`.
- Download CHAMELEON, CAMO, COD10K, and NC4K from dataset-author or official
  paper repositories into `data/cod/`.
- Preserve source archives only when disk capacity permits. Extracted,
  validated data takes priority over duplicate archive copies.
- Record datasets that cannot be downloaded because of authentication,
  authorization, quota, dead links, or insufficient disk space.

## Dataset Classification

CHAMELEON is treated as a camouflaged-object benchmark even though the current
README lists it under polyp segmentation. It belongs under `data/cod/` and is
not added to the medical polyp evaluation configuration.

## Polyp Data Flow

`sam2unet.data.prepare_pranet_polyp_data` will normalize all five PraNet test
directories: Kvasir, CVC-ClinicDB, CVC-300, CVC-ColonDB, and ETIS. Existing
training behavior and manifests remain unchanged. Validation discovers all
supported test sets, verifies image-mask readability and stem pairing, and
checks optional expected counts.

The existing prepared directory will be augmented from the official PraNet
test archive rather than rebuilt, avoiding unnecessary duplication of the
verified training set.

## COD Data Flow

COD downloads are stored under `data/cod/archives/` while in transit and
extracted under `data/cod/prepared/`. The final layout preserves the official
train/test split and normalizes dataset folder names where necessary. A local
inventory records image and mask counts and detects unmatched stems.

The repository does not currently contain `cod_dataset.py` or
`configs/cod_train.json`; implementing a COD training pipeline is outside this
dataset-completion task. No placeholder training code will be added.

## Sources

- PraNet official repository and its published TrainDataset/TestDataset links.
- CAMO author project page for CAMO.
- SINet/COD10K official repository for COD10K and CHAMELEON benchmark data.
- NC4K official project repository for NC4K.

Mirrors are used only if an official link is inaccessible, and any mirror use
is reported explicitly.

## Validation

- Existing archive SHA-256 values must continue matching
  `configs/polyp_sources.json`.
- Polyp pair counts must be 1,450 train, 100 Kvasir, 62 CVC-ClinicDB,
  60 CVC-300, 380 CVC-ColonDB, and 196 ETIS.
- Every generated manifest row must resolve to readable image and mask files.
- COD datasets must have matching image/mask stems and README-scale counts for
  every downloaded split.
- The focused data tests and complete pytest suite must pass.

## Failure Reporting

Each unavailable dataset is reported with its attempted official URL, observed
failure class, and the user action needed to resolve it. Partial archives are
not presented as completed datasets.

