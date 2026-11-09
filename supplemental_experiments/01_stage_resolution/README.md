# Stage-resolution sensitivity of E/C/F

This experiment addresses reviewer vCLF's request to measure E/C/F sensitivity
to finer reveal grids. It fixes each model, paired endpoints, reveal path,
patch order, preprocessing, and epsilon=0.02. Inference measures only the
finest grid; all coarser grids are exact subsets of that same score cache.
Every grid uses its own trapezoidal weights.

**No synthetic result fallback exists.** Analytic arrays in `tests/` are unit
tests of the analysis, not empirical evidence. The finest grid is a numerical
reference, not ground truth or a proof of continuous-path convergence.
Conservation is reported separately from convergence diagnostics.

## Inputs and dependencies

Run from the repository root. Use Python 3.11+ with the repository dependencies;
inference additionally requires compatible PyTorch, torchvision, and Pillow.
`analyze` needs no model, image data, PyTorch, or GPU once the complete cache and
manifest have been copied. All checkpoints must already exist locally;
the runner never downloads assets.

Copy `config.example.json` and set real paths. Path strings are resolved relative
to the config file, so absolute paths are recommended. A new support drawn from
the official ImageNet-9 release is accepted and fingerprinted independently; it
is not claimed to be the original paper's 72-model/768-image support.

Accepted CSV/Parquet manifests use the repository's ImageNet-9 schemas:

- Wide: `pair_id,true_in9_class,mixed_same_path,mixed_rand_path,mixed_next_path`.
  Each source row expands into both Same-Rand and Same-Next.
- Typed: `pair_id,pair_type,original_path,counterfactual_path,class_id`, optionally
  `source_pair_id`. Class IDs are 0–8 and image paths are dataset-relative.

Optional `split` filters a manifest's split column. `max_source_pairs` performs
a reproducible class-balanced selection keyed by `selection_seed`; null uses
all selected source pairs. The manifest records exact image-byte hashes, source
manifest hash, checkpoint hashes, mapping hash, configuration hash, and source
code hashes. Use at least three models for a defined across-model Spearman
correlation; more models provide a more informative ranking check.

`pretrained` checkpoints are strict torchvision state dicts or a `state_dict`
wrapper; any registered torchvision classification architecture is supported.
`fine_tuned` uses the repository's existing strict loader, including its
`model_spec` validation. 1000-way logits receive one softmax followed by the
official 1000-to-9 probability-mass mapping, without renormalization. Direct
9-way heads receive one softmax. Inputs use the repository's ImageNet evaluation
geometry, [0,1] pixel reveal, and model-side ImageNet normalization.

## Execute

```bash
python supplemental_experiments/01_stage_resolution/run.py plan \
  --config /absolute/path/to/stages.json --output /absolute/path/to/stage-run
python supplemental_experiments/01_stage_resolution/run.py infer \
  --config /absolute/path/to/stages.json --output /absolute/path/to/stage-run \
  --device cuda:0
python supplemental_experiments/01_stage_resolution/run.py analyze \
  --output /absolute/path/to/stage-run
python supplemental_experiments/01_stage_resolution/report.py \
  --output /absolute/path/to/stage-run
```

Inference shards by model, reveal path, and fixed blocks of typed pairs. Each
NPZ contains `grid`, `q_plus`, `q_minus`, and `pair_ids`; it is accompanied by a
SHA-256 receipt. Re-running `infer` skips verified shards. An output directory
with a changed configuration, model, data, or implementation is rejected.
Interrupted writes without a receipt are recomputed. Corrupt completed shards
are rejected, rather than silently reused.

To distribute the same plan over two independent processes/devices, pass
`--workers 2 --worker-index 0 --device cuda:0` and
`--workers 2 --worker-index 1 --device cuda:1`. Use distinct worker indices;
the processes share the same output directory. `--model-id NAME` can restrict
work to one registered model. `analyze` requires every planned shard, records
missing task IDs in `analysis_status.json`, and refuses to present partial
support as a completed result.

## Outputs and interpretation

- `manifest.json`: complete input, support, configuration, code identities and
  task plan.
- `cache/*.npz` and `cache/*.json`: reusable finest-grid score queries and receipts.
- `analysis/sample_metrics.parquet`: E/C/F/Abs per pair/grid; absolute and
  relative error against the finest grid; component shares and share errors;
  endpoint activity; missed contradiction; conservation residual.
- `analysis/model_metrics.csv`: model/path/pair-type means, normalized mean
  absolute error, relative mean bias, pair-rank Spearman, aggregate shares, and
  share errors. `normalized_mae` divides sample MAE by the finest-grid component
  mean; no arbitrary epsilon is added to stabilize a zero denominator.
- `analysis/model_rank_stability.csv`: model-rank Spearman for component means
  and shares, using identical pair support for all models.
- `analysis_status.json`: completeness, query counts, and interpretation limits.
- `analysis/grid_summary.csv`: median and worst errors across the model/pair-type
  groups, and minimum across-model rank correlation for each path/grid/component.
- `analysis/endpoint_checks.json`: an independent cache-hash and cross-path
  endpoint check. The report checks that the two paths retained identical
  endpoint scores and endpoint gates for each model/pair.

Zero-reference relative errors and correlations of constant vectors remain
undefined (`NaN`), with absolute errors still reported. The reference grid has
zero error by construction; useful evidence comes from whether changes shrink
as the preceding grids approach it, especially 9→17→33→65.

The patch path uses 64 patches and floor-based reveal counts. At 65 stages every
patch addition is sampled. This is a discrete protocol with step transitions;
do not describe its trapezoidal summary as an exact integral of a smooth path.
Blend paths can have unsampled changes even between the finest stages.

## Validation

```bash
python -m pytest supplemental_experiments/01_stage_resolution/tests -q
```

Tests include a narrow reversal that coarse grids miss despite exact
conservation, analytically integrable active/null paths, the threshold boundary,
non-nested grids, and inconsistent cross-model support.

## Completed public-data experiment

`result_summary.json` records the completed run with 12 public pretrained
torchvision classifiers, 450 deterministically selected class-balanced source
images, both Same-Rand and Same-Next (900 typed pairs), and blend/patch_A paths.
The 696 cache shards contain 2,808,000 model-input evaluations. Two B200 workers
took 660.10 seconds from process start to the last inference receipt; CPU
analysis is excluded from that time. The full manifest, per-pair results,
checkpoint/data hashes, logs, and environment records are under
`runs/public_12models_450pairs/` (large run files are ignored by Git).

At nine stages, pooled sample MAE divided by the finest-grid component mean is:

| Path | E | C | F |
| --- | ---: | ---: | ---: |
| Blend | 2.94–3.07% | 16.11–16.42% | 11.28–12.50% |
| Patch A | 10.28–10.47% | 17.02–18.22% | 11.96–13.24% |

Ranges span Same-Rand and Same-Next. Across-model ranking correlations are
0.986–1.000 at nine stages. Component magnitudes still change appreciably with
resolution: nine stages miss positive reference contradiction on 23.23–33.73%
of reference-positive model/pair cases. Those misses account for 0.925–3.291%
of total reference contradiction mass. At 33 stages, normalized C errors fall
to 1.56–4.12%; all detailed values, including adverse and undefined cases, are
retained in the JSON and CSV outputs. These results support stable model
rankings on this support and show measurable resolution sensitivity in
nine-stage component values. No application-specific acceptance tolerance was
imposed.

All 10,800 cross-path endpoint comparisons have exactly equal endpoint scores
and gates; conservation residuals are zero. These checks establish the matched
protocol and arithmetic identities, separately from the resolution findings.

The execution hashes in `result_summary.json` refer to the source preserved in
`runs/public_12models_450pairs/source_snapshot/`. After completion, the current
CLI received a typed-manifest split-filter fix; the measured run used a wide
manifest and no split, so its selection and numbers are unaffected. The
analysis module is unchanged. Current `run.py analyze` and `report.py` reproduce
the cached analysis; new inference should use a new output directory. To verify
or resume the original execution contract, use its snapshot entrypoint with
the repository's unchanged dependency package on `PYTHONPATH`:

```bash
PYTHONPATH="$PWD/src" python \
  supplemental_experiments/01_stage_resolution/runs/public_12models_450pairs/source_snapshot/supplemental_experiments/01_stage_resolution/run.py \
  plan \
  --config supplemental_experiments/01_stage_resolution/runs/public_12models_450pairs/config.json \
  --output supplemental_experiments/01_stage_resolution/runs/public_12models_450pairs
```

Replace `plan` with `infer --device cuda:0` to resume verified missing shards
using that snapshot. Existing complete shards are skipped. Regenerate the
tracked summary with:

```bash
python supplemental_experiments/01_stage_resolution/report.py \
  --output supplemental_experiments/01_stage_resolution/runs/public_12models_450pairs \
  --summary supplemental_experiments/01_stage_resolution/result_summary.json
```
