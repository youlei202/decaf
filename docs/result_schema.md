# Result and receipt schema

Every invocation writes one self-contained run directory:

```text
runs/<experiment>/<run_id>/
|-- run.json
|-- config.yaml
|-- environment.json
|-- manifests/
|   |-- data.json
|   |-- checkpoints.json
|   `-- jobs.jsonl
|-- raw/
|-- metrics/
|-- paper_data/
|-- receipts/
`-- logs/
```

Paths recorded inside public artifacts are relative to the run root. Hostnames,
usernames, private mount points, access tokens, and environment-variable values
must not be serialized.

## Global receipt

`run.json` is the authoritative global receipt. Required fields are:

| Field | Meaning |
|---|---|
| `schema_version` | Integer schema version |
| `run_id` | Unique, filesystem-safe run identifier |
| `experiment` | `controlled`, `imagenet9`, `attribution`, or `covertype` |
| `stage` and `profile` | Requested stage and `smoke` or `paper` profile |
| `status` | One of the five statuses defined below |
| `started_at`, `completed_at` | UTC RFC 3339 timestamps; completion is null only while running |
| `config_sha256` | Digest of the frozen resolved configuration |
| `data_manifest_sha256` | Digest of `manifests/data.json` |
| `checkpoint_manifest_sha256` | Digest of `manifests/checkpoints.json` |
| `jobs_sha256` | Digest of `manifests/jobs.jsonl` |
| `required_members` | Deterministically sorted required member identifiers |
| `optional_failures` | Deterministically sorted optional failure summaries |
| `error` | Structured terminal error, or null |

Allowed global statuses are exactly `running`, `partial`, `failed`,
`completed`, and `completed_with_optional_failures`. Once all child
processes exit, the global receipt must be terminal; it must never remain
`running`.

`environment.json` records portable facts needed for interpretation: Python
and package versions, platform and accelerator types, deterministic settings,
source revision, and sanitized scheduler metadata. It must not expose private
paths or secrets.

## Job plan

Each line of `manifests/jobs.jsonl` is one deterministic member with
`member_id`, `phase`, `required`, `resource`, `seed`,
`dependencies`, and a run-relative `output`. The ordered JSONL file is the
schedule contract. Reordering or changing any member changes its digest.

## Atomic member receipts

Each planned member writes `receipts/<member_id>.json` with:

- `schema_version`, `member_id`, `attempt`, `required`, and `status`;
- UTC `started_at` and `completed_at`;
- resolved config, dataset, checkpoint, and dependency digests;
- an `artifacts` list containing run-relative `path`, `bytes`, and
  `sha256`;
- timing and sanitized resource metadata; and
- a structured `error` containing type, message, and retryability, or null.

Member receipts use `running`, `failed`, or `completed`. Writers create a
temporary file in the receipt directory, flush and synchronize it, and atomically
replace the destination. A receipt is reusable only when it says `completed`,
its input digests still match, and every artifact matches both byte count and
SHA-256. A non-terminal or failed receipt may be retried. A completed receipt
whose identity, input digest, byte count, or artifact SHA-256 no longer matches
is rejected fail-closed; use a clean run directory (or explicitly remove the
stale member and receipt) before recomputing it.

Global status is reduced from member receipts: any failed required member makes
the run `failed`; missing required terminal receipts make it `partial`; all
required members plus failed optional members yield
`completed_with_optional_failures`; otherwise the result is `completed`.

## ImageNet-9 external GPU members

The paper-profile prepare stage writes manifests/jobs.jsonl and repeats the
worker contract in manifests/plan.json. The external worker writes only to each
job's declared run-relative output and receipt paths. Scan output is Parquet
with pair_id, pair_type, model_id, reveal_path, stage_index, alpha, and response.
Baseline output is Parquet with pair_id, pair_type, model_id, method_id, and
score.

Every receipt contains schema_version, job_id, the canonical job_sha256,
terminal status=completed, output, output_sha256, row_count, and the exact
ordered dependency_artifacts list. Compute rejects missing or extra pair
support, a changed alpha/stage grid, wrong model or method identity, row-count
drift, dependency-output drift, and receipt/output hash mismatches. The public
compute stage validates and aggregates these materialized files; it does not
execute the external GPU worker.

## Attribution GPU adapter

The formal adapter is loaded lazily from
execution.adapter=module:function and has signature
(job: Mapping[str, Any], context: RunContext) -> pandas.DataFrame. Image-member
frames cover the exact ordered image_index interval, contain unique image_id,
and repeat the planned scope, dataset, model, and method. Target jobs add finite
target_effects; quality jobs add finite patch_scores, decaf_M, spearman, and
finite_complete. Jobs without target dependencies also provide their endpoint
or quality-target vectors. Timing jobs return exactly one row with the planned
repeat and four nonnegative timing measurements.

The framework attaches plan, data-manifest, checkpoint-byte, input-contract, and
dependency-output hashes; validates the resulting frame; writes Parquet
atomically; and creates the terminal receipt. Resume revalidates every one of
those bindings before skipping a member. The adapter must not forge lineage
columns itself.

## Single-B200 verification receipts

The gated real-shard runners use the same global lifecycle and add one atomic
receipt per GPU member. A completed receipt binds the canonical job/config hash,
dataset-manifest hash, checkpoint SHA-256, run-relative output, output SHA-256,
attempt, device, precision, elapsed time, and peak CUDA memory where applicable.
The single-device scheduler records queue order, refill events, unique outputs,
unique receipts, failure isolation, and finalization. A normal SIGTERM makes the
global receipt terminal (`partial` or `failed`); completed members remain
reusable and an incomplete member is retried on `--resume`.

Checkpoint fingerprint JSON is separate from experiment output. It contains
exactly 12 cases and records one or more checkpoint byte identities, fixed
sample IDs, a little-endian C-contiguous tensor fingerprint, target class,
rank-two logits and probabilities, float precision, package versions, and the
single visible B200 identity. The public release includes the portable summary,
not private asset paths or checkpoint bytes.

## Portable tables

Canonical raw response tables include the identifiers needed to reconstruct a
cell: `experiment`, `run_id`, `member_id`, `model_id`, `sample_id`,
`protocol`, `stage_index`, `stage_value`, and `seed`, followed by
family-specific response columns. Canonical model-metric tables include
`experiment`, `model_id`, `profile`, `metric`, `value`,
`n_samples`, and the relevant uncertainty fields.

Every global canonical paper-data table under
`paper_data/canonical/{figures,tables}` includes `artifact_id`, `panel_id`,
`series`, `x`, `y`, `estimate`, `ci_low`, `ci_high`, `n`, `source_sha256`, and
`record_json`. The final field is a portable canonical JSON object preserving
the asset-specific semantic identity fields. Family-local `paper_data` files
remain validated source/intermediate schemas and are not claimed to use this
global renderer schema. Column order, null policy, panel cardinality, source
lineage, and semantic-contract hashes are frozen in `canonical_receipt.json`.
