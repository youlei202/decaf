# DECAF supplementary experiments

Each reviewer question has its own directory, executable entry point, protocol
description, tests, and run outputs.

| Directory | Question | Measurement |
| --- | --- | --- |
| [01_stage_resolution](01_stage_resolution/README.md) | Does refining the reveal grid change E/C/F? | The same measured 65-stage trajectories are subsampled to 3/5/9/17/33/65 stages. Component errors, composition, missed contradiction, and model rankings are compared. |
| [02_cub_sub](02_cub_sub/README.md) | What happens on the requested fine-grained attribute-substitution benchmark? | Official SUB images and labels, an official-style attribute-recognition baseline, and a paired-response audit with the available pairing limitations stated explicitly. |
| [03_component_specificity](03_component_specificity/README.md) | Do other components also correlate with the controlled behavior? | All components and magnitude/endpoint comparators on the same historical observations, with paired cluster bootstrap and architecture/task breakdowns. |

## Completed measurements

All three runs completed on 2026-11-09. The GPU inference processes have exited.
The complete supplementary test suite passes 23 tests, and Ruff reports no errors.

| Experiment | Completed support | Main observation |
| --- | --- | --- |
| [Grid results](01_stage_resolution/result_summary.json) | 12 models, 450 source images, two background contrasts, two paths; 2,808,000 input evaluations | Nine-point model rankings agree strongly with 65 points (rho >= 0.986), but C normalized mean absolute error is about 16–18%. Stable rankings do not imply converged component values. |
| [SUB results](02_cub_sub/result_summary.json) | 3,072 original PNG images, all 768 bird–attribute cells, 2,104 predefined pairs; 500 bird-cluster bootstrap replicates | For recognition success, E AUROC is 0.8449 versus endpoint M 0.8688. F equals Abs on null pairs. A separately labeled post-hoc negative-M control also prevents claiming a reversal-detection advantage for C. |
| [Cross-component results](03_component_specificity/result_summary.json) | 52 evidence checkpoints, 18 fragility models, 30 contradiction models; 2,000 cluster-bootstrap replicates | On the same evidence checkpoints, E/C/F margins correlate 0.9361/0.6899/-0.8933 with vulnerability; M correlates 0.9594. Other components also carry behavioral information. |

The SUB release contains synthetic birds, and the response audit pairs different
generated instances of the same class/attribute group. These measurements do not
establish same-photograph, single-attribute causal effects. The cross-component
run reanalyzes existing recorded measurements; it does not train a new common
model population. Each linked README gives the full protocol and uncertainty.

## Environment

The GPU environment was verified on NVIDIA B200 with PyTorch 2.9.1/CUDA 12.8
and torchvision 0.24.1. The exact installed package versions are recorded in
[requirements.lock.txt](requirements.lock.txt). Create an isolated environment:

```bash
uv venv --python python3.12 /tmp/decaf-supplemental-venv
uv pip install --python /tmp/decaf-supplemental-venv/bin/python \
  -r supplemental_experiments/requirements.lock.txt
```

Run commands from the repository root. Each script imports the repository core
where needed. Generated runs are kept inside each experiment's `runs/` directory;
the repository already ignores these large run artifacts. Input data and weights
live in a separate asset directory. Small result summaries are retained beside
the code after completed analyses.

## GPU inventory

The four supplied SSH ports plus the current machine expose eight distinct B200
GPUs: local (one), 2813 (three), 2550 (two), 2702 (one), and 2064 (one). Initially
only the three GPUs on port 2813 were empty. The first two are used for the grid
experiment and the third for SUB. Existing workloads on other GPUs are retained.
GPU occupancy is transient; inspect it before another launch:

```bash
python supplemental_experiments/probe_gpus.py \
  --identity /path/to/ssh_private_key \
  --output supplemental_experiments/runs/gpu_inventory.json
```

The probe is read-only and counts unique GPU UUIDs, including when endpoints
refer to the same physical GPU. It does not reserve devices or stop processes.

## Public ImageNet-9 assets

Historical ImageNet-9 images and author fine-tunes were unavailable in the current
mount. The grid experiment therefore uses the official Backgrounds Challenge test
release and the twelve public torchvision models listed in the paper's model zoo.
It is a new measurement on a documented subset, not a reproduction of all 72 models.

```bash
python supplemental_experiments/prepare_imagenet9_public.py \
  --root /path/to/imagenet9_assets --download-data --source-pairs 450 \
  --config-output supplemental_experiments/01_stage_resolution/runs/public/config.json
```

This verifies all 4,050 foreground identities across the image variants, the
paper's official class mapping, and official pretrained weight digests. It creates
a new manifest and a class-balanced, score-independent selection configuration.
Skip `--download-data` when the release is already extracted. See the experiment
README for planning, worker sharding, cache validation, and analysis commands.

## Verification

```bash
python -m pytest -q --import-mode=importlib supplemental_experiments
```

Small artificial arrays used by numerical tests are test fixtures only. Scientific
outputs come from actual public data/model inference or the recorded historical
C1/C2 measurements. Missing files, incomplete caches, undefined correlations, and
unavailable historical assets are reported explicitly rather than filled in.
