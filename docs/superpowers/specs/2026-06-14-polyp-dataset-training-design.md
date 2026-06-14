# Polyp Dataset Training Design

## Goal

Prepare Kvasir-SEG and CVC-ClinicDB for fast, reproducible SAM2-UNet polyp
segmentation experiments.

## Scope

- Download the complete Kvasir-SEG and CVC-ClinicDB datasets.
- Preserve the source archives and record their origins.
- Build the standard PraNet/SAM2-UNet split:
  - Kvasir-SEG: 900 train, 100 test.
  - CVC-ClinicDB: 550 train, 62 test.
- Expose one combined 1,450-image training set and two named test sets.
- Validate image-mask pairing, image readability, mask readability, and counts.
- Add a machine-readable experiment configuration for 352x352 binary
  segmentation.

## Directory Layout

```text
data/polyps/
  archives/
  raw/
    Kvasir-SEG/
    CVC-ClinicDB/
  train/
    images/
    masks/
  test/
    Kvasir/
      images/
      masks/
    CVC-ClinicDB/
      images/
      masks/
  manifests/
```

Dataset names prefix files in the combined training set to prevent collisions.
Manifests record source dataset, split, image path, and mask path.

## Data Preparation

The preparation command accepts downloaded archives or extracted source
directories. It discovers image and mask folders, matches samples by stem,
sorts them deterministically, and reserves the final official-sized subset for
testing. The remaining samples form the training set.

Files are copied into the normalized layout so training code does not depend on
the packaging conventions of a specific mirror.

## Experiment Configuration

The configuration uses:

- Input size: 352x352.
- Task: binary segmentation.
- Training datasets: Kvasir-SEG and CVC-ClinicDB.
- Test datasets: Kvasir-SEG and CVC-ClinicDB.
- Training epochs: 20.
- Metrics: Dice, IoU, and MAE.
- Model comparison modes: rfb, static, and full.

## Validation

Preparation fails when:

- An image or mask is unreadable.
- An image has no corresponding mask.
- A mask has no corresponding image.
- Dataset counts are lower than the required official split.

Automated tests use small synthetic datasets to verify deterministic splitting,
collision-safe naming, manifests, and validation errors. A final validation
command checks the real prepared data and reports counts.

## Repository Safety

The `data/` directory is ignored by Git. Scripts, configuration, manifests,
and source records remain reproducible project files where appropriate, while
downloaded archives and images remain local.
