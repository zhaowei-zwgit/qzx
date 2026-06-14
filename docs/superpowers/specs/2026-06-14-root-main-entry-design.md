# Root Main Entry Design

## Goal

Add a root-level `main.py` that provides one convenient command-line entry point
for the existing polyp segmentation smoke test, formal training, and evaluation
workflows.

## Architecture

`main.py` is a thin adapter over `sam2unet.experiment.main`. It parses a
user-friendly `--mode` option, translates the selected options into the
existing experiment CLI arguments, prints the selected run and elapsed time,
and returns the experiment exit code. Training, evaluation, checkpointing,
metrics, data loading, and random seeding remain owned by
`src/sam2unet/experiment.py`.

The default mode is `smoke`, so running `python main.py` does not accidentally
start a long formal training run. The default configuration is
`configs/polyp_train.json`.

## Command-Line Interface

- `--mode {smoke,train,evaluate}` selects the workflow and defaults to `smoke`.
- `--config`, `--device`, `--bridge-mode`, `--output-root`, `--limit-train`,
  and `--limit-test` are forwarded for all modes.
- `--epochs` and `--resume` are forwarded only for `train`.
- `--checkpoint` is required for `evaluate` and rejected for the other modes.

## Error Handling

Argument validation happens before importing and running the experiment.
Evaluate mode without `--checkpoint` produces a standard argparse error.
Exceptions from the underlying workflow are not hidden; timing output is still
printed from a `finally` block.

## Testing

Unit tests verify default smoke argument translation, train-specific argument
translation, evaluate checkpoint validation, and experiment exit-code
propagation. A command-line help check and the existing lightweight smoke
workflow provide end-to-end verification.
