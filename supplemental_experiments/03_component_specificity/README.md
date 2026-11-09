# Three-component specificity on the existing controlled experiments

This supplement directly addresses pFed's question: on the **same evidence
checkpoints and behavioral target**, do C and F also correlate with the behavior
attributed to E? It reanalyzes measured historical caches; it does not train models,
construct synthetic scientific results, or combine distinct modules into one pool.

## Run

From the repository root, using an environment with NumPy, pandas, SciPy and PyArrow:

```bash
python supplemental_experiments/03_component_specificity/analyze.py \
  --reference-root verification/replay/paper_data \
  --output supplemental_experiments/03_component_specificity/runs/my_run \
  --bootstrap 2000 --confidence 0.95 --seed 20261109
```

The completed run is `runs/full_reference_final/`. Its input files are the registered
C1 `aggregate_summary.parquet` and C2 `epsilon_curves.csv` / `behavior_rates.csv`.
`result_summary.json` contains selected results and their run provenance for review.
`runs/full_reference/` is the earlier completed version before adding the endpoint
M comparator. `runs/full_reference_v2/` precedes a formatting-only cleanup;
the final run recomputes all statistics with the current script, with unchanged results.

Individual input paths can instead be supplied using `--c1-aggregate`,
`--c2-scores`, and `--c2-behavior`. C2 paths must be supplied together. A C1 input
contains both the evidence and fragility modules. `--geometry` selects a single
alternative reveal geometry; the default uses the registered primary geometry.
Nonempty output directories are rejected. `--allow-partial` explicitly labels a
run with other model counts as partial; it never fills absent measurements.

## Matched comparisons and uncertainty

- **C1 evidence:** 52 checkpoints from eight training trajectories. Compare
  `E_wall-E_shape`, `C_wall-C_shape`, `F_wall-F_shape`, `Abs_wall-Abs_shape`, and
  `M_wall-M_shape` against the same `V_rev`; also report the individual wall and
  shape scores. All predictors share one complete-case support. Bootstrap whole
  training trajectories within architecture × training-correlation strata, retaining
  every selected checkpoint together. Report architecture, training-correlation,
  and individual-trajectory results separately.
- **C1 fragility:** 18 models, measuring floor-color E/C/F/Abs against the same
  held-out `null_prediction_change_rate`. Report both architectures and each
  intervention variant separately. Bootstrap architecture–seed blocks, keeping
  the three trained variants together, stratified by architecture. Clean accuracy
  and endpoint-null fraction remain in the exported observations: some ResNet
  models do not realize the intended endpoint-null, high-accuracy regime.
- **C2 contradiction:** 30 models, 21 mismatch probabilities and two wall maps
  (1,260 rows). Join independently recorded behavior one-to-one on all measurement
  keys; compare every component with preservation, swap and collapse. Resample
  architecture–seed blocks, keeping all tasks, mismatch levels and maps together,
  within architecture strata. Task, architecture, and task × architecture results
  are also reported. The historical C2 column `epsilon` denotes **context mismatch
  probability**, not DECAF's endpoint-null threshold.

All component comparisons use **the same bootstrap draws**. We report both
signed correlation differences and differences in absolute correlation strength;
a large negative correlation is not evidence of no association. Constant scores
or outcomes give missing correlations, with explicit status. A subgroup with fewer
than two independent clusters in any stratum has point estimates only, rather
than an interval manufactured from repeated checkpoint or mismatch observations.

Intervals are 95% percentile cluster-bootstrap intervals with 2,000 replicates.
They are conditional on the registered tasks, architecture and training-correlation
strata. They are marginal intervals, not simultaneous multiple-comparison bounds;
there are only eight evidence trajectory clusters and only three or five seed
blocks per architecture in the other experiments. They quantify uncertainty in
these retained runs, not all possible trained models. They deliberately differ
from the paper's original 90% evidence interval.

## What the measured results say

On the same 52 evidence checkpoints:

| Score versus shortcut reversal vulnerability | Spearman rho | 95% cluster-bootstrap interval |
| --- | ---: | --- |
| E margin | 0.9361 | [0.9220, 0.9626] |
| C margin | 0.6899 | [0.6548, 0.7393] |
| F margin | -0.8933 | [-0.9360, -0.8643] |
| Abs margin | 0.8965 | [0.8392, 0.9551] |
| Endpoint M margin | 0.9594 | [0.9536, 0.9647] |

E's correlation is stronger than C's in the pooled comparison (difference
0.2462, interval [0.1841, 0.3034]). However, **F is strongly negatively correlated**;
the absolute-strength gap between E and F is only 0.0427 [0.0030, 0.0786]. Across
the six trajectories with variable vulnerability, the mean within-trajectory
correlation is +0.8364 for E, -0.8364 for F, and +0.8364 for M. The other two
trajectories have constant outcomes and are marked missing. Within ResNet alone,
E and C are both strongly correlated (0.8603 and 0.8556). These results do **not**
establish that only E contains information about shortcut vulnerability. The
stronger endpoint comparator must also be acknowledged.

For fragility, on the nine Small-ViT models, both F and Abs have rho = 1.0 with
held-out null prediction change (E = 0.6167, C = 0.7333). Across all 18 models,
F = 0.4716 while Abs = 0.9670; the ResNet subgroup has F = -0.0167 and contains
models that become endpoint-active or lose clean accuracy. This is a reason to
state the operating regime, not to claim uniform cross-architecture superiority.

For C2, C tracks label swap at rho = 0.9607 while Abs = -0.0359 and E = -0.6814.
F is identically zero and its correlation is **undefined**, not zero. Within
Invert alone, E and C are almost perfectly opposite correlates of swap
(-0.9989 and +0.9989). The direct task's swap rate is constant, so that task's
swap correlations are undefined. The results support interpreting aligned versus
opposed responses; they do not establish statistical independence of components.

Some pooled cross-component correlations already existed in the historical
C1 tables T03 and T05 but were not reported in the supplied paper/rebuttal. This
supplement makes those comparisons explicit, adds matched component-difference
intervals and stratified analyses, and preserves unfavorable as well as favorable
results. It is **not** a new experiment varying all three mechanisms independently
in a single trained model population.

## Outputs and checks

- `*_observations.csv`: exact measured rows used by each analysis, with cluster IDs.
- `correlations.csv`: point correlations, absolute strengths, intervals, and missing statuses.
- `paired_correlation_differences.csv`: paired signed and absolute-strength differences.
- `support.csv`: common-support model/row/cluster counts and excluded rows.
- `receipt.json`: input SHA-256 hashes, script hash, settings, and completion scope.

```bash
python -m pytest -q supplemental_experiments/03_component_specificity/test_analysis.py
```

The six tests cover constant-variable handling, preserving entire clusters,
paired-bootstrap differences, common support, insufficient independent clusters,
and rejection of incomplete factor/behavior joins. Their small fixtures are only
statistical software tests and are never written as experiment results.
