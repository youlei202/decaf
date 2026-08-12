# Dataset inventory

No dataset is redistributed by this repository. Put licensed or downloaded data
under `$DECAF_DATA_ROOT` according to the static manifests in
`manifests/datasets/`. Every frozen fingerprint is a SHA-256 digest unless a
different algorithm is named explicitly.

| Dataset | Canonical local root | Official identity | License or access condition |
|---|---|---|---|
| 3D Shapes | `$DECAF_DATA_ROOT/3d_shapes` | [DeepMind 3D Shapes](https://github.com/google-deepmind/3d-shapes) | Apache-2.0 |
| ImageNet-9 / Backgrounds Challenge | `$DECAF_DATA_ROOT/imagenet9` | [MadryLab Backgrounds Challenge](https://github.com/MadryLab/backgrounds_challenge) | No license was recorded in the pinned source; images derive from ImageNet and are not redistributed |
| FunnyBirds | `$DECAF_DATA_ROOT/funnybirds` | [FunnyBirds](https://github.com/visinf/funnybirds) | Apache-2.0 |
| ImageNet-1k for IDSDS | `$DECAF_DATA_ROOT/imagenet1k` | [ImageNet](https://www.image-net.org/) | ImageNet access and usage terms; registration may be required |
| PartImageNet | `$DECAF_DATA_ROOT/partimagenet` | [PartImageNet](https://github.com/TACJu/PartImageNet) | No machine-readable license was recorded in the pinned source; underlying ImageNet terms apply |
| Covertype | `$DECAF_DATA_ROOT/covertype` | [UCI Covertype](https://archive.ics.uci.edu/dataset/31/covertype) | CC BY 4.0 |

## Frozen contracts

### 3D Shapes

The source is the complete 480,000-row Cartesian grid over six generative
factors. The official `3dshapes.h5` download is 267,573,662 bytes with SHA-256
`0a0f6ed98baff276a50f3a081a7434d788da63cb135a98189b2a5b5769be1785`
and MD5 `099a2078d58cec4daad0702c55d06868`. The paper pipeline uses a
32-by-32 processed view and frozen split, factor-index, covariance-geometry, and
state-support contracts. Their digests are recorded in
`manifests/datasets/3d_shapes.yaml`.

### ImageNet-9 / Backgrounds Challenge

The experiment uses the nine-class Backgrounds Challenge organization and a
frozen paired-variant manifest of 4,050 rows, 450 per class. The sealed score,
deep, and outcome split manifests contain 1,644, 768, and 1,586 rows
respectively. The 820-row preselection count found in the paired-variant
manifest precedes sealing; 768 is the actual paper deep-pair plan. The
class-mapping and class-name files are fingerprinted in the YAML manifest.
Image files remain subject to their upstream terms.

For the gated single-B200 shard, `DECAF_DATA_ROOT` points to the prepared
ImageNet-9 data root containing `manifests/paired_variants.parquet`; the official
1,000-to-9 mapping may be selected explicitly with `DECAF_IMAGENET9_MAPPING`.
Controlled binds the two processed arrays directly through
`DECAF_CONTROLLED_IMAGES_32_UINT8` and
`DECAF_CONTROLLED_FACTOR_INDICES`; both complete arrays are hashed and their
shape, dtype, and row-major factor identity are validated.

### FunnyBirds

The source repository is pinned to revision
`91b4b4628ffa962148144ee6bb5af5f022cac2f8`; the evaluation framework is
pinned to `1350bbe812ff18df25b2708df861245b9a63b9c9`. The official archive
is 1,549,208,915 bytes with SHA-256
`5b0dc030484ff4b1cd993e4084927bf9709c037193fa9fe4818d70a3e9e89f51`.
The frozen study manifest identifies the 1,001 used members; verification does
not require hashing every unused source member on each replay.

### ImageNet-1k for IDSDS

IDSDS uses ImageNet-1k validation images with frozen 10,000-image evaluation,
50,000-image full-validation, 1,024-image timing, and strict common-support
manifests. A pinned complete, unresampled parquet repack may be used as a
transport mechanism, but it does not change ImageNet licensing or grant
redistribution rights. Users must acquire authorized access themselves.

### PartImageNet

The semantic-segmentation archive is pinned to the official PartImageNet source
revision `f4bf3df88b126d3a2d5e8671a8c2ea90de39638e`. Its expected size is
3,124,435,169 bytes and SHA-256 is
`9719778db7a7f589af94de4d7e4a025b832835502df370154f7c0a8b35466090`.
The paper path validates 72,240 used archive members through the frozen subset
and part-group manifests.

Attribution B200 profiles require explicit dataset manifests through
`DECAF_FUNNYBIRDS_MANIFEST`, `DECAF_IDSDS_MANIFEST`, and
`DECAF_PARTIMAGENET_MANIFEST`, together with their respective source-root
variables. The manifests select fixed real common support and semantic masks;
directory discovery or synthetic substitution is not accepted.

### Covertype

The original UCI table has 581,012 rows and 54 features. The frozen paper split
selects forest cover types 1 and 2, maps them to -1 and +1, balances 240,000
rows, and uses seed 7601 for 144,000 training, 48,000 validation, and 48,000 test
rows. The first ten continuous features are standardized from training
statistics with population variance; the remaining 44 binary features are
preserved. The logical split fingerprint is
`07aa4349a338b765e0c143407ecc0acd4ccdf35ed0e13c8014519fdc013ade9c`.

The required CPU integration uses the pinned transport archive
`$DECAF_DATA_ROOT/covertype_balanced_240000_split7601.npz` and its adjacent
`.manifest.json`. The archive SHA-256 is
`681f893d49757e4d588115430b072980df2f4c281acedb1183b53ef5b4e443de`; the
manifest SHA-256 is
`ce6790aed36ce39051c2bbcb2672689a51cf3a504a893b6122a9d7ffa4a219ed`. Neither
file is redistributed. The integration profile selects fixed balanced rows from
each already-frozen split and records both source digests plus the selected-shard
fingerprint in its data manifest. Missing or changed bytes stop the run; this path
never falls back to the synthetic smoke fixture.

## Fingerprint policy

`verified` in a YAML entry means the fingerprint came from a frozen public
configuration, manifest, or archived lightweight reference package. It does not
mean that the current machine already contains the asset. `user-provided` or
`unverified` means no redistributable, immutable byte source was frozen; the
pipeline must stop with an actionable message unless the user supplies an
authorized copy or regenerates it. Verification never silently substitutes a
similarly named dataset.
