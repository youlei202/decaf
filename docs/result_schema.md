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
SHA-256. Otherwise resume reruns that member.

Global status is reduced from member receipts: any failed required member makes
the run `failed`; missing required terminal receipts make it `partial`; all
required members plus failed optional members yield
`completed_with_optional_failures`; otherwise the result is `completed`.

## Portable tables

Canonical raw response tables include the identifiers needed to reconstruct a
cell: `experiment`, `run_id`, `member_id`, `model_id`, `sample_id`,
`protocol`, `stage_index`, `stage_value`, and `seed`, followed by
family-specific response columns. Canonical model-metric tables include
`experiment`, `model_id`, `profile`, `metric`, `value`,
`n_samples`, and the relevant uncertainty fields.

Every `paper_data` table includes `artifact_id`, `panel_id`, `series`,
`x`, `y`, `estimate`, `ci_low`, `ci_high`, `n`, and
`source_sha256` where applicable. Tables may add documented family-specific
columns, but must not silently drop identity, sample count, or provenance
columns. Column order, dtypes, null policy, and sort keys are frozen in the
resolved schema used by the run.
