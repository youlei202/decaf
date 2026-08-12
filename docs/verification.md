# Verification

The verification wrapper provides named lanes with different evidence levels:

| Mode | Purpose | Executes model compute |
|---|---|---|
| `unit` | Deterministic unit tests | No |
| `integration-cpu` | Small real Covertype integration and CPU pipeline checks | Yes, CPU only |
| `analysis-replay` | Frozen run validation, analyses, figures, and headline assertions | No |
| `full-plan` | Static expansion and contract audit of every paper plan | No |
| `repository-audit` | Public-file, schema, English-only, and private-path checks | No |
| `checkpoint-fingerprint` | Exact offline checkpoint loads and fixed real CUDA forwards | Yes, one CUDA GPU |
| `all-cpu` | All CPU-safe lanes, analysis replay, paper regeneration, and audits | CPU only |

Run the broad CPU-safe lane with:

```bash
bash scripts/reproduce/verify.sh --mode all-cpu
```

Run an individual lane by replacing `all-cpu` with its mode name. Analysis
replay requires `$DECAF_REFERENCE_RUNS_ROOT`; integration requires the
dataset roots described in `manifests/datasets/`.

The CPU integration lane specifically requires the pinned real Covertype cache:

```bash
export DECAF_DATA_ROOT=/path/to/covertype-cache
bash scripts/reproduce/verify.sh --mode integration-cpu
```

That directory must directly contain
`covertype_balanced_240000_split7601.npz` and its adjacent
`covertype_balanced_240000_split7601.manifest.json`. The lane verifies both byte
digests and the logical split fingerprint, then runs a fixed 1,200/400/400-row
train/validation/test shard with one model family, one seed, one direction
regime, one fragility regime, DECAF, a permutation baseline, analysis,
paper-data generation, and receipt-based resume. An unset root, missing cache,
or digest mismatch is a failure; synthetic substitution is forbidden.

## What is checked

1. Reference archives and selected assets are matched by SHA-256 before use.
2. Real-shard input receipts bind the cache archive, companion manifest, logical
   split, deterministic shard selection, and selected-shard fingerprint.
3. Run receipts are terminal and each declared artifact exists with its recorded
   byte count and digest.
4. Analysis replay reproduces canonical tables and paper panel data from frozen
   raw results.
5. Headline assertions compare named metrics against frozen targets with
   assertion-specific absolute or relative tolerances. A rounded value shown in
   a paper is never used as the tolerance source.
6. Paper rendering is rerun from the validated analysis outputs.
7. The full-plan audit checks expected member cardinalities and dependencies
   without scheduling GPU work.
8. The repository audit rejects private absolute paths, non-public run
   references, and non-English prose in public files.

Expected static paper-plan counts include:

- controlled: 30 base models, 180 factor-model units, 44 C1 factory jobs covering
  52 evidence, 18 causal, and 18 fragility checkpoints, 316 C1 measurements,
  and 30 context-swap models with 30 dependent evaluations;
- ImageNet-9: 72 model descriptors, 32 deep models, and 768 sealed deep pairs;
- attribution: three aligned architectures, with DINOv2-g and PartImageNet
  boundary work in explicitly requested profiles;
- Covertype: 135 models, comprising 90 causal-direction and 45 fragility models.

## Reading the result

A verification report records the lane, validated input digests, individual
checks, elapsed time, and final status. Its top-level status is `passed` only
when every required gate succeeds; `failed` is not a passing result. Individual
experiment run receipts use the separate lifecycle statuses documented in
`docs/result_schema.md`.

The all-CPU lane is intentionally honest about its boundary. Full real-shard GPU
verification is a separate, explicitly activated workflow. The v2 verification
release records representative real single-B200 shards for controlled,
ImageNet-9, attribution, DINOv2-g, and PartImageNet, plus a real Covertype CPU
shard. Those shards validate the refactored compute paths; they do not claim a
full paper-scale recomputation or real multi-GPU execution.

## Single-B200 verification mode

Set `DECAF_B200_VERIFY=1` only after binding the offline datasets and exact
checkpoint files documented in the manifests. In this mode, the three vision
wrappers use the persistent GPU Python selected by `DECAF_GPU_PYTHON`, require
exactly device 0, reject downloads and fallbacks, and write hash-bound atomic
member receipts. Run the checkpoint gate first:

```bash
DECAF_B200_VERIFY=1 bash scripts/reproduce/verify.sh \
  --mode checkpoint-fingerprint --devices 0 --output /path/to/verification
```

Fingerprint coverage is fixed at two controlled models, three ImageNet-9
models, and seven attribution models. Each record binds checkpoint bytes,
sample IDs, the canonical preprocessed tensor hash, target, logits,
probabilities, precision, library versions, and GPU identity. The controlled
and ImageNet-9 `smoke` profiles and attribution `smoke`, `large-model-smoke`,
`boundary-smoke`, and `smoke-resume` profiles then execute the reduced real
shards. Default smoke behavior is unchanged when the gate is absent.
