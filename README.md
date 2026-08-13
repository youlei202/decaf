<div align="center">

# DECAF

### Decomposition of Evidence, Contradiction, and Fragility in Perturbation Responses

**Response magnitude tells you how much a model reacts. DECAF tells you what kind of response produced it.**

[Paper](https://cspaper.org/openprint/20260813.0001v1) · [Quick start](#quick-start) · [Reproduce the paper](#reproduce-the-paper) · [Documentation](#documentation)

</div>

<!-- Add an online-demo link here when a public demo is available. -->
<!-- Add a PyPI installation badge and command after the first package release. -->

## Why DECAF?

Perturbation and counterfactual methods often reduce a model response to one number: **how much the prediction changed**. Equal response magnitudes, however, can describe very different behavior.

Given a factual endpoint $x^+$, a counterfactual endpoint $x^-$, and matched reveal trajectories $x^+(t)$ and $x^-(t)$, DECAF observes

$$
r(t) = q(x^+(t)) - q(x^-(t)), \qquad d = r(1),
$$

where $q$ is any scalar model score. The clean endpoint determines whether the final contrast is active and, when active, which direction counts as support:

$$
a = \mathbf{1}\{|d| \geq \varepsilon\}, \qquad s = \mathrm{sign}(d).
$$

DECAF then routes every stage response into three nonnegative components:

$$
e(t) = a\,[s r(t)]_+, \qquad
c(t) = a\,[-s r(t)]_+, \qquad
f(t) = (1-a)|r(t)|.
$$

| Component | Meaning |
|---|---|
| **Evidence — $E$** | Response aligned with the final factual–counterfactual effect |
| **Contradiction — $C$** | Response opposed to the final effect |
| **Fragility — $F$** | Response along the path when the final effect is negligible |

The routing is lossless:

$$
|r(t)| = e(t) + c(t) + f(t), \qquad \mathrm{Abs} = E + C + F.
$$

DECAF therefore preserves ordinary response magnitude while revealing which kind of response produced it.

## What DECAF requires

DECAF begins after a paired intervention has been specified. You provide:

1. a scalar score $q$ to explain;
2. a meaningful factual–counterfactual pair $(x^+, x^-)$;
3. matched reveal trajectories from a common uninformative state to those endpoints;
4. a threshold $\varepsilon$ expressed on the same scale as the score.

DECAF does **not** choose the counterfactual or reveal path for you. Its semantics are relative to the score, pair, path, stage measure, and threshold that you specify.

Once the paired scores are available, the decomposition:

- requires no gradients, parameters, or internal activations;
- adds no model queries beyond the paired trajectory;
- works with neural networks, tree ensembles, simulators, and remote score APIs;
- supports batching across examples, stages, factors, paths, and models;
- can be accumulated online with constant additional memory per active batch.

## Quick start

### 1. Install from source

A PyPI release is not available yet. The current repository can be used directly with Python 3.11 or newer and [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/youlei202/decaf.git
cd decaf
uv sync
```

### 2. Decompose paired score trajectories

The example below contains three paired trajectories: an evidence-dominant response, a response with substantial contradiction, and an endpoint-null fragile response.

```python
import numpy as np

from decaf.core import trajectory_scores

# Five matched reveal stages from the common start (t=0) to clean endpoints (t=1).
t = np.array([0.00, 0.25, 0.50, 0.75, 1.00])

# Rows are paired examples; columns are reveal stages.
q_plus = np.array(
    [
        [0.50, 0.55, 0.65, 0.75, 0.900],
        [0.50, 0.40, 0.35, 0.60, 0.800],
        [0.50, 0.70, 0.40, 0.65, 0.505],
    ]
)
q_minus = np.array(
    [
        [0.50, 0.45, 0.40, 0.35, 0.400],
        [0.50, 0.60, 0.65, 0.50, 0.400],
        [0.50, 0.40, 0.60, 0.40, 0.495],
    ]
)

response = q_plus - q_minus
result = trajectory_scores(
    grid=t,
    response=response,
    epsilon=0.02,
)

summary = {
    name: np.round(result[name], 4).tolist()
    for name in ("M", "E", "C", "F", "Abs")
}
print(summary)
assert result["numeric_audit"]["passed"]
```

Expected output:

```text
{
  'M':   [0.5, 0.4, 0.01],
  'E':   [0.25, 0.075, 0.0],
  'C':   [0.0, 0.125, 0.0],
  'F':   [0.0, 0.0, 0.1887],
  'Abs': [0.25, 0.2, 0.1887]
}
```

Run the example with:

```bash
uv run python example.py
```

### 3. Connect DECAF to your own model

The model-facing part is deliberately small. Evaluate the same scalar score on the two matched branches at every reveal stage, then pass their difference to DECAF:

```python
plus_scores = []
minus_scores = []

for x_plus_t, x_minus_t in paired_path:
    plus_scores.append(score(model(x_plus_t)))
    minus_scores.append(score(model(x_minus_t)))

q_plus = np.stack(plus_scores, axis=-1)
q_minus = np.stack(minus_scores, axis=-1)

result = trajectory_scores(
    grid=stage_positions,
    response=q_plus - q_minus,
    epsilon=epsilon,
)
```

For stochastic models or stochastic reveal protocols, use shared randomness across the factual and counterfactual branches whenever possible.

## Output fields

`trajectory_scores(...)` returns per-example endpoint and trajectory summaries:

| Field | Description |
|---|---|
| `M` | Clean endpoint magnitude $|d|$ |
| `E` | Integrated endpoint-aligned evidence |
| `C` | Integrated endpoint-opposed contradiction |
| `F` | Integrated endpoint-null fragility |
| `Abs` | Integrated ordinary magnitude; exactly $E+C+F$ up to numerical tolerance |
| `Net` | Signed oriented mass, $E-C$ |
| `signed_E` | Evidence restored to the endpoint score direction |
| `endpoint_delta` | Signed clean endpoint contrast $d$ |
| `endpoint_active` | Whether $|d| \geq \varepsilon$ |
| `pointwise_components` | Stage-wise decomposition and endpoint metadata |
| `numeric_audit` | Pointwise and integrated conservation checks |

The core package also exposes:

- `decompose(...)` for pointwise routing;
- `integrate_components(...)` for finite-grid integration;
- `StreamingDECAFAccumulator` for stage-by-stage accumulation;
- bootstrap, metric, manifest, and run-receipt utilities used by the experiments.

## Choosing the endpoint threshold

The threshold $\varepsilon$ is not universal. It must be chosen on the scale of the score being analyzed.

The paper uses $\varepsilon=0.02$ for true-class probabilities in the main ImageNet-9 audit. Logits, margins, regression values, rewards, and other scores generally require a different threshold. We recommend reporting:

- the score definition and direction;
- the chosen threshold;
- the endpoint-active fraction;
- a small threshold-sensitivity sweep when conclusions depend on branch membership.

Under positive score rescaling, rescale $\varepsilon$ by the same factor to preserve gate membership.

## What the paper finds

The experiments test DECAF across controlled vision, natural images, tabular models, external attribution benchmarks, and a large vision transformer.

| Question | Main finding |
|---|---|
| Can equal magnitudes hide different behavior? | In a 72-model ImageNet-9 audit with nearly matched ordinary magnitudes, response-role agreement rises from **0.350** for magnitude alone to **0.964** with DECAF. |
| What changes when only the reveal path changes? | A patch reveal increases total response by about **1.8×**; evidence changes little, while fragility grows by more than **4×**. |
| Does the forward-only design scale? | On a 1B-scale DINOv2 model, DECAF-5 reaches nearly the same attribution quality as IG-32 with **4.75× lower wall time** and **2.36× lower peak memory**. |

Read the full methods, assumptions, experiments, and boundary cases in the [paper](https://cspaper.org/openprint/20260813.0001v1).

## Reproduce the paper

The repository contains four experiment families:

- controlled 3D Shapes mechanisms;
- ImageNet-9 paired-background audits;
- FunnyBirds, ImageNet-1k, PartImageNet, and DINOv2 attribution experiments;
- the Covertype tabular benchmark.

Install the development dependencies and run the CPU verification suite:

```bash
uv sync --all-extras
bash scripts/reproduce/verify.sh --mode unit
```

Reference-run replay uses four explicit roots:

```bash
export DECAF_DATA_ROOT=/path/to/datasets
export DECAF_CACHE_ROOT=/path/to/cache
export DECAF_RESULTS_ROOT=/path/to/results
export DECAF_REFERENCE_RUNS_ROOT=/path/to/sealed/reference/runs
```

Analysis replay and paper-asset generation:

```bash
bash scripts/reproduce/controlled.sh \
  --stage analyze --profile paper --output runs/controlled/replay

bash scripts/reproduce/imagenet9.sh \
  --stage analyze --profile paper --output runs/imagenet9/replay

bash scripts/reproduce/attribution.sh \
  --stage analyze --profile paper --output runs/attribution/replay

bash scripts/reproduce/covertype.sh \
  --stage analyze --profile paper --output runs/covertype/replay

bash scripts/reproduce/paper.sh \
  --reference-runs "$DECAF_REFERENCE_RUNS_ROOT" \
  --output paper/generated

bash scripts/reproduce/verify.sh --mode analysis-replay
```

The stored analysis replay does not perform model inference. Full vision training and inference require the datasets and checkpoints described in the documentation, together with suitable GPU hardware.

The CPU verification suite exercises the common core, a real Covertype shard, result schemas, and static audits of the full experiment plans. The representative B200 verification records real execution shards and runtime contracts; it does not claim a complete paper-scale rerun. Sealed historical outputs remain the source of the regenerated paper values.

## Documentation

- [Reproduction guide](docs/reproduction.md)
- [Datasets](docs/datasets.md)
- [Checkpoints](docs/checkpoints.md)
- [Hardware](docs/hardware.md)
- [Verification scope](docs/verification.md)
- [Result schema](docs/result_schema.md)

## Repository layout

```text
src/decaf/core/          Reusable DECAF decomposition and numerical primitives
src/decaf/experiments/   Controlled, ImageNet-9, attribution, and Covertype pipelines
src/decaf/paper/         Reference replay and paper-asset generation
configs/                  Smoke, integration, and paper experiment profiles
manifests/                Dataset, checkpoint, and reference-run contracts
scripts/reproduce/        Stable verification and reproduction entry points
docs/                     Data, hardware, schema, and reproduction documentation
tests/                    Unit, integration, and regression tests
paper/                    Visual manifests and generated-asset targets
```

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check .
```

The implementation promotes response arrays to `float64` before endpoint classification and identity checks. Tests cover conservation, endpoint-swap invariance, score scaling, finite-grid/streaming agreement, validation failures, schemas, experiment contracts, and paper replay.

## Citation

```bibtex
@article{you2026decaf,
  title   = {DECAF: Decomposition of Evidence, Contradiction, and Fragility in Perturbation Responses},
  author  = {You, Lei},
  year    = {2026},
  note    = {Preprint},
  url     = {https://cspaper.org/openprint/20260813.0001v1}
}
```

## License

Released under the [MIT License](LICENSE).
