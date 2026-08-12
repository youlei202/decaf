# Hardware expectations

Resource needs differ sharply by mode. Values below are planning guidance, not
guaranteed performance targets. Storage estimates exclude backups and source
archives that a dataset license may require you to obtain separately.

| Workload | Practical minimum | Recommended paper-scale host | Accelerator status |
|---|---|---|---|
| Unit tests and repository audit | 2 CPU cores, 8 GB RAM | 8 CPU cores, 16 GB RAM | None |
| Frozen analysis replay and TeX generation | 4 CPU cores, 16 GB RAM | 16-32 CPU cores, 32-64 GB RAM | None |
| Controlled smoke | 8 CPU cores, 32 GB RAM, one CUDA GPU | 16 CPU cores, 64 GB RAM | One modern GPU |
| Controlled full plan | 16 CPU cores, 64 GB RAM | 32 CPU cores and at least four high-memory CUDA GPUs | Representative single-B200 shard verified; full plan not rerun |
| ImageNet-9 smoke | 8 CPU cores, 32 GB RAM, one CUDA GPU | 16 CPU cores, 64 GB RAM | One modern GPU |
| ImageNet-9 full plan | 16 CPU cores, 128 GB RAM | 32 or more CPU cores, 256 GB RAM, eight high-memory CUDA GPUs | Representative single-B200 shard verified; full plan not rerun |
| Attribution smoke | 8 CPU cores, 32 GB RAM, one CUDA GPU | 16 CPU cores, 64 GB RAM | One modern GPU |
| Attribution full plan | 16 CPU cores, 128 GB RAM | 32 or more CPU cores, 256 GB RAM, eight high-memory CUDA GPUs | Main, DINOv2-g, and PartImageNet single-B200 paths verified; full plan not rerun |
| Covertype smoke | 4 CPU cores, 16 GB RAM | 16 CPU cores, 32 GB RAM | None |
| Covertype full plan | 16 CPU cores, 64 GB RAM | 32 CPU cores, 64-128 GB RAM | CPU-only; 135 models |

Allow at least 300 GB for ImageNet-1k, ImageNet-9, PartImageNet, FunnyBirds,
derived manifests, checkpoints, and result caches together. Exact use depends
on whether source archives and extracted copies are retained. The controlled
dataset itself is small, while attribution perturbation caches can grow
substantially.

## Scheduling guidance

- Treat the 32-CPU recommendations as aggregate worker capacity. Cap data-loader
  workers so the sum across concurrent jobs does not oversubscribe the host.
- Schedule DINOv2-g and other large attribution jobs exclusively. A checkpoint
  being only a few gigabytes does not predict activation-memory demand.
- Use local scratch for random-access image data, then copy only receipt-declared
  artifacts to durable storage.
- Use `--resume` for paper runs. Completed members are hash-validated before
  reuse, so a preemption does not require repeating the whole plan.
- Run `--plan-only` before dispatch to see member counts and resource labels.

## Verification boundary

The all-CPU verification lane validates tests, frozen analysis replay, paper
regeneration, a real small Covertype integration, static paper plans, and public
repository hygiene. The separate v2 single-B200 workflow validates
representative real CUDA shards, checkpoint fingerprints, receipt-driven resume,
and a heterogeneous one-device queue. It does **not** validate real multi-GPU
scheduling or rerun the paper-scale plans. CPU success alone must not be reported
as GPU verification, and single-B200 success must not be reported as a full
paper-scale recomputation.
