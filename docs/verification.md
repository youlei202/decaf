# Verification

The verification wrapper provides named lanes with different evidence levels:

| Mode | Purpose | Executes model compute |
|---|---|---|
| `unit` | Deterministic unit tests | No |
| `integration-cpu` | Small real Covertype integration and CPU pipeline checks | Yes, CPU only |
| `analysis-replay` | Frozen run validation, analyses, figures, and headline assertions | No |
| `full-plan` | Static expansion and contract audit of every paper plan | No |
| `repository-audit` | Public-file, schema, English-only, and private-path checks | No |
| `all-cpu` | All CPU-safe lanes, analysis replay, paper regeneration, and audits | CPU only |

Run the broad CPU-safe lane with:

```bash
bash scripts/reproduce/verify.sh --mode all-cpu
```

Run an individual lane by replacing `all-cpu` with its mode name. Analysis
replay requires `$DECAF_REFERENCE_RUNS_ROOT`; integration requires the
dataset roots described in `manifests/datasets/`.

## What is checked

1. Reference archives and selected assets are matched by SHA-256 before use.
2. Run receipts are terminal and each declared artifact exists with its recorded
   byte count and digest.
3. Analysis replay reproduces canonical tables and paper panel data from frozen
   raw results.
4. Headline assertions compare named metrics against frozen targets with
   assertion-specific absolute or relative tolerances. A rounded value shown in
   a paper is never used as the tolerance source.
5. Paper rendering is rerun from the validated analysis outputs.
6. The full-plan audit checks expected member cardinalities and dependencies
   without scheduling GPU work.
7. The repository audit rejects private absolute paths, non-public run
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
checks, elapsed time, and final status. `completed` means every required check
passed. `completed_with_optional_failures` means required checks passed and
only declared optional checks failed. `partial` and `failed` are not passing
results.

The all-CPU lane is intentionally honest about its boundary. Full real-shard GPU
verification for controlled, ImageNet-9, and attribution remains pending. Until
those runs complete, describe the repository as CPU-verified with frozen
analysis replay, not as fully reproduced on GPUs.
