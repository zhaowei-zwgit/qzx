# COD Training Integration Design

## Goal

Make the prepared camouflaged-object datasets fully runnable through the existing
`main.py` smoke, training, and evaluation workflows without changing the model,
loss, checkpoint, or metric contracts used by polyp segmentation.

## Scope

The implementation will add a directory-backed COD dataset loader, a COD training
configuration, and configuration-driven loader dispatch in the shared experiment
pipeline. It will update documentation and add automated and real-data smoke tests.

The task remains binary segmentation. NC4K instance masks are retained on disk but
are not consumed by the current loss or metrics.

## Architecture

### COD Directory Dataset

Add `CODDirectoryDataset` in `src/sam2unet/cod_dataset.py`. Each instance receives
one dataset root containing `images/` and `masks/`, discovers supported image files,
and matches image-mask pairs by case-insensitive filename stem.

The loader rejects missing directories, empty datasets, duplicate stems, and
unmatched files with errors that include the dataset name and relevant path. Each
sample returns the same mapping contract as `PolypManifestDataset`: `image`, `mask`,
`dataset`, `split`, and `image_path`.

Images are converted to RGB, resized with bilinear interpolation, normalized with
ImageNet statistics, and returned as `(3, H, W)` tensors. Masks are converted to
grayscale, resized with nearest-neighbor interpolation, binarized at 0.5, and
returned as `(1, H, W)` float tensors. Training applies synchronized horizontal and
vertical flips; evaluation applies no augmentation.

### Experiment Loader Dispatch

Keep the existing manifest workflow as the default. `build_loaders()` will inspect
`dataset_type`:

- `manifest` or an omitted value uses `PolypManifestDataset` and the existing
  `train_manifest`/`test_sets` schema.
- `cod_directory` uses `CODDirectoryDataset`, combines configured training datasets
  with `ConcatDataset`, and creates one evaluation loader per configured test set.

Model construction, optimizer setup, deep-supervision loss, metrics, checkpoint
handling, monitoring, and root CLI forwarding remain shared.

## Configuration

Create `configs/cod_train.json` with:

- `dataset_type`: `cod_directory`
- `train_sets`: CAMO and COD10K directory roots
- `test_sets`: CAMO, COD10K, NC4K, and CHAMELEON directory roots
- `output_root`: `runs/cod`
- expected train and test counts matching the validated local inventory
- the same SAM2 checkpoint, model configuration, binary metrics, optimizer options,
  bridge modes, and smoke controls used by the current experiment pipeline

The training `ConcatDataset` preserves each source dataset label in returned samples.
NC4K uses its binary `masks/` directory; `instances/` remains out of scope.

## Error Handling

Dataset construction fails immediately when a configured path is absent, contains
no supported files, contains duplicate case-insensitive stems, or has unmatched
images and masks. Unsupported `dataset_type` values raise a clear configuration
error rather than falling back silently.

Expected counts in the COD configuration are verified by a final real-data
integration check so accidental truncation is detected before a full training run.
The portable unit-test suite uses temporary fixtures and does not require ignored
dataset files to be present after a fresh clone.

## Testing And Acceptance

Automated tests will cover:

- image-mask pairing, tensor shapes, binary masks, labels, and synchronized flips
- missing masks, duplicate stems, and empty-directory errors
- COD loader dispatch, combined training length, and named test loaders
- compatibility with the existing manifest-backed Polyp workflow
- JSON configuration structure and temporary directory fixtures

Final acceptance requires a separate count check against the prepared real data,
the complete portable test suite to pass, and a CPU COD smoke run to train on
limited CAMO/COD10K samples, evaluate all four COD test datasets, save a checkpoint,
restore it, and write the normal smoke report under `runs/cod/smoke/`.

README status will then identify the COD data loader and training configuration as
implemented while preserving the distinction between binary NC4K masks and unused
instance annotations.
