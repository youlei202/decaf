# DECAF

DECAF is an endpoint-anchored decomposition of a counterfactual response
trajectory into evidence (E), contradiction (C), and endpoint-null fragility
(F). This repository is the clean, reproducible implementation accompanying
the DECAF experiments.

The public experiment interface has four families:

- controlled synthetic mechanisms;
- ImageNet-9 paired-background mechanisms;
- attribution benchmarks on FunnyBirds, ImageNet-1k, and PartImageNet;
- the Covertype tabular benchmark.

## Quick start

Install Python 3.11 or newer and
[uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync --all-extras
bash scripts/reproduce/verify.sh --mode unit
```

Reference-run replay uses four environment variables:

```bash
export DECAF_DATA_ROOT=/path/to/datasets
export DECAF_CACHE_ROOT=/path/to/cache
export DECAF_RESULTS_ROOT=/path/to/results
export DECAF_REFERENCE_RUNS_ROOT=/path/to/sealed/reference/runs
```

No private filesystem location is assumed by the code. See
`docs/reproduction.md` for analysis-only replay, smoke runs, full experiment
plans, and paper-asset generation.

## Reproduce the paper

```bash
bash scripts/reproduce/controlled.sh --stage analyze --profile paper --output runs/controlled/replay
bash scripts/reproduce/imagenet9.sh --stage analyze --profile paper --output runs/imagenet9/replay
bash scripts/reproduce/attribution.sh --stage analyze --profile paper --output runs/attribution/replay
bash scripts/reproduce/covertype.sh --stage analyze --profile paper --output runs/covertype/replay
bash scripts/reproduce/paper.sh --reference-runs "$DECAF_REFERENCE_RUNS_ROOT" --output paper/generated
bash scripts/reproduce/verify.sh --mode analysis-replay
```

The stored analysis replay does not perform model inference. Full vision
training and inference require the datasets and checkpoints documented in
`docs/datasets.md` and `docs/checkpoints.md`, together with multi-GPU
hardware. The CPU verification suite runs the common core, a real Covertype
shard, schema checks, and static audits of every full experiment plan.

## Repository contract

Each run follows the schema in `docs/result_schema.md`. Generated PDF files,
raw datasets, restricted checkpoints, private paths, and development incident
documents are deliberately excluded from releases.
