# Reproduction guide

This repository separates four activities that are easy to confuse:

- **Analysis replay** reads frozen, lightweight reference runs. It does not train
  a model or rerun GPU inference.
- **Smoke execution** runs a deliberately small end-to-end workload and checks
  plumbing, not paper-scale numerical agreement.
- **Full paper execution** regenerates a paper-profile run from source data and
  checkpoints. It can require substantial accelerator time.
- **Paper rendering** consumes completed or reference runs and writes standalone
  TeX plus machine-readable panel data. It does not create PDFs.

## Environment

Use Python 3.11 and install the locked project environment:

```bash
uv sync --all-extras
```

Keep data, caches, outputs, and optional reference archives outside the source
tree. The commands below use these public interfaces:

```bash
export DECAF_DATA_ROOT=/path/to/datasets
export DECAF_CACHE_ROOT=/path/to/cache
export DECAF_RESULTS_ROOT=/path/to/results
export DECAF_REFERENCE_RUNS_ROOT=/path/to/reference-runs
```

Dataset and checkpoint placement is defined in
`manifests/datasets/` and `manifests/checkpoints/`. A restricted asset is
never fetched implicitly; supply it only if its terms allow your use.

## Analysis replay

These are the canonical replay commands, one per experiment family:

```bash
bash scripts/reproduce/controlled.sh --stage analyze --profile paper --output "$DECAF_RESULTS_ROOT/controlled/replay"
bash scripts/reproduce/imagenet9.sh --stage analyze --profile paper --output "$DECAF_RESULTS_ROOT/imagenet9/replay"
bash scripts/reproduce/attribution.sh --stage analyze --profile paper --output "$DECAF_RESULTS_ROOT/attribution/replay"
bash scripts/reproduce/covertype.sh --stage analyze --profile paper --output "$DECAF_RESULTS_ROOT/covertype/replay"
```

Replay expects the matching frozen run beneath
`$DECAF_REFERENCE_RUNS_ROOT`. It verifies the reference manifest before
analysis. Endpoint M is part of the ordinary attribution analysis path; it is
not a separate reproduction mode.

## Smoke execution

These commands exercise preparation, computation, analysis, and family-local
paper-data generation on reduced plans:

```bash
bash scripts/reproduce/controlled.sh --stage all --profile smoke --output "$DECAF_RESULTS_ROOT/controlled/smoke"
bash scripts/reproduce/imagenet9.sh --stage all --profile smoke --output "$DECAF_RESULTS_ROOT/imagenet9/smoke"
bash scripts/reproduce/attribution.sh --stage all --profile smoke --output "$DECAF_RESULTS_ROOT/attribution/smoke"
bash scripts/reproduce/covertype.sh --stage all --profile smoke --output "$DECAF_RESULTS_ROOT/covertype/smoke"
```

Smoke outputs prove that the local stack works. They are not substitutes for
paper-profile estimates and must not be compared to headline tolerances.

## Full paper execution

The paper profile expands the sealed plan. Use `--resume` so completed members
with valid receipts are retained after interruption:

```bash
bash scripts/reproduce/controlled.sh --stage all --profile paper --output "$DECAF_RESULTS_ROOT/controlled/paper" --resume
bash scripts/reproduce/imagenet9.sh --stage all --profile paper --output "$DECAF_RESULTS_ROOT/imagenet9/paper" --resume
bash scripts/reproduce/attribution.sh --stage all --profile paper --output "$DECAF_RESULTS_ROOT/attribution/paper" --resume
bash scripts/reproduce/covertype.sh --stage all --profile paper --output "$DECAF_RESULTS_ROOT/covertype/paper" --resume
```

Before allocating hardware, inspect the exact static plan without executing it:

```bash
bash scripts/reproduce/verify.sh --mode full-plan
```

The paper plans contain 30 controlled base models, 72 ImageNet-9 models, the
three aligned attribution architectures plus optional large-model and boundary
profiles, and 135 Covertype models. See `docs/hardware.md` before starting.

## Paper rendering

Rendering is deliberately separate from family computation:

```bash
bash scripts/reproduce/paper.sh --reference-runs "$DECAF_REFERENCE_RUNS_ROOT" --output paper/generated
```

The renderer emits standalone `.tex` fragments and machine-readable panel
data. A TeX installation is optional because PDF compilation is outside this
repository's contract.

## Stages and recovery

Family wrappers accept
`--stage prepare|compute|analyze|paper|all`,
`--profile smoke|paper`, `--output`, `--resume`, and `--plan-only`.
The family-local `paper` stage prepares results for the global renderer; it
does not compile a manuscript. Resume is receipt-driven: a member is reused only
when its terminal receipt and declared artifact hashes validate. See
`docs/result_schema.md`.
