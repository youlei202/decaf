# SUB attribute-substitution audit

This experiment addresses reviewer pFed's explicit request for a CUB/SUB
fine-grained attribute evaluation. It uses the **official SUB images and labels**,
the official attribute prompts, and frozen public OpenCLIP weights. It does not
generate replacement images or substitute a patch-deletion experiment for SUB.

## What the public data supports

[SUB, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Bader_SUB_Benchmarking_CBM_Generalization_via_Synthetic_Attribute_Substitutions_ICCV_2025_paper.html)
contains photorealistic **FLUX-generated** birds with CUB species and substituted
attributes. It is not a collection of edits paired with original CUB photographs.
The paper's [Sections 3–4](https://arxiv.org/html/2507.23784v1) explain that TDG's
guidance image is discarded. Its paired generation should not be confused with
two released images sharing a source and differing in one attribute.

Pinned public sources:

| Resource | Pinned version / protocol |
| --- | --- |
| [Official dataset](https://huggingface.co/datasets/Jessica-bader/SUB/tree/6687681ee20f37b375b9ea8354e5841409fd5d7e) | `6687681ee20f37b375b9ea8354e5841409fd5d7e`; CC-BY-NC-4.0; 38,400 rows, 23 Parquet shards, about 12.1 GB |
| [Official evaluation code and prompts](https://github.com/ExplainableML/sub/tree/953a7559878a29e2c7c4a9c79db5e72eee046a85) | `953a7559878a29e2c7c4a9c79db5e72eee046a85` |
| [Official CLIP example](https://github.com/ExplainableML/sub/blob/953a7559878a29e2c7c4a9c79db5e72eee046a85/CBM_testing/test_clip_example.py) | `ViT-B-32`, `laion2b_s34b_b79k`, normalized image/text cosine scores |
| [Official generation demo](https://github.com/ExplainableML/sub/blob/953a7559878a29e2c7c4a9c79db5e72eee046a85/tied_diffusion_guidance_demo.py) | No published cross-attribute seed mapping (`generator=None`) |

The pinned release exposes `image`, `bird_label`, and `attr_label`. Its actual
label dictionary has 33 bird classes and 33 target attribute labels, comprising
768 class–attribute cells with 50 images each. The article discusses initial
candidate concepts and a filtered label set with different counts; this code
uses the actual pinned release, without inventing absent labels. Parquet
`image.path` contains repeated filenames such as `7.png`, not source-pair IDs.
Equal filenames therefore do **not** establish matched generation seeds.

There are two outputs with different evidential scopes:

1. **Single-image substitution recognition.** Following the official CLIP
   evaluation, choose the highest cosine-scoring attribute within the labeled
   attribute group, using all alternatives in the official CUB prompt file.
   Compare it with the released substituted-attribute label. Report clean and
   held-out-view accuracy, by attribute group and with bird-class cluster
   bootstrap intervals. All released attributes are included, so this is not an
   exact reproduction of paper tables that filter concepts for particular CBMs.
2. **DECAF response audit on predefined matched pairs.** Pair images of the same
   bird class with different values of the same attribute group. These are
   **different generated instances**, with potentially different pose,
   background, and other attributes. This tests DECAF on semantic contrasts in
   the requested benchmark; it does not establish an isolated causal effect of
   changing one pixel-level factor in one original photograph.

## Fixed evaluation protocol

Defaults are chosen before model inference. Select four images from every
class–attribute cell by SHA256 ordering with seed `20261109`: 3,072 images.
For each class/group, enumerate every unordered pair of available target values
and zip the independently hash-ordered images. This gives 2,104 pairs. Matching
does not use embeddings, predictions, correctness, or outcome measurements.
The lower attribute ID determines the plus branch and queried attribute; no
orientation is selected to improve results. One image can participate in
multiple contrasts, so uncertainty resamples **bird classes**, not individual
observations as if independent.

The scalar score is the frozen CLIP model's softmax probability for the plus
attribute, among **all official prompt alternatives in its group**, using the
checkpoint's logit scale. This explicit normalization gives the threshold a
probability-unit interpretation. The official single-image metric still uses
cosine argmax (equivalent ranking), rather than pretending that the paper uses
this probability calibration.

Both branches start from a shared Gaussian-blurred midpoint (sigma 12 model-input
pixels) and linearly reveal their endpoints at nine equally spaced stages.
Trapezoidal aggregation produces E, C, F, Abs, endpoint magnitude M, signed
endpoint, and signed net. The threshold is `epsilon=0.02`; equality is active.
The code verifies common start, conservation, endpoint consistency, and exact
`F=Abs` on endpoint-null pairs.

Independent endpoint queries use four fixed views, none of which are trajectory
samples: horizontal flip, Gaussian blur (radius 1.5 original-image pixels), JPEG
quality 40, and downsample-to-128 then bilinear resize. The views are intended to
preserve attribute identity while testing recognition robustness; severe
visibility loss remains a limitation, especially for small parts.

| Behavioral outcome | Definition, using endpoint views only |
| --- | --- |
| Recognition success | Both endpoints' group-wise argmax matches their respective supplied attribute labels in at least 3 of 4 views |
| Off-path reversal | On clean endpoint-active pairs, at least one view reverses the clean score-contrast sign with magnitude at least epsilon |
| Off-path instability | At least one branch changes its group-wise predicted attribute, or its queried-attribute probability changes by at least 0.20 in any view |

Report every scalar baseline and component on **identical full, active, or null
populations**. Reversal is only evaluated on active pairs because near-zero
clean contrast does not provide a reliable orientation. The null gate is shared
with the representation and is explicitly disclosed. On null pairs F and Abs
must have identical AUROC; the script does not award F a different gate from
its comparator. Degenerate outcomes yield missing AUROC, never a fabricated 0.5.
Paired AUROC differences against Abs and M use the same bootstrap resamples.
No classifier, calibration, threshold optimization, or prompt selection is fit
to SUB. These are exploratory robustness diagnostics, without an assumed winner.

After the initial results were inspected, the analysis additionally reports
`negative_M` and `negative_Abs` **for every outcome and population**, marked
`posthoc_direction_control` in the CSVs and summary. Low endpoint margins can
predict reversal, so a comparison against positive magnitude alone would be
incomplete. These controls do not alter the originally specified rows, do not
take the better direction silently, and are not represented as a prospectively
chosen or independently fitted predictor. The original analysis script and
tables are retained in `runs/clip_b32_laion2b/analysis_prespecified/`.

There is no random image-level train/test split because the public model is
frozen and all SUB data are evaluation-only. Existing CLIP pretraining may
contain CUB concepts; we do not assert a fully decontaminated pretraining corpus.
Any future fitted analysis should split by bird class or source group, not
randomly divide pairs that reuse images.

## Run

Python 3.11+ dependencies:

```bash
python -m pip install numpy scipy scikit-learn pyarrow pyyaml Pillow pytest
python -m pip install torch torchvision open_clip_torch
```

From the repository root:

```bash
python supplemental_experiments/02_cub_sub/prepare.py \
  --root /work/Users/leiyo/decaf_supplemental_assets/cub_sub \
  --per-cell 4 --workers 4 --delivery parquet

CUDA_VISIBLE_DEVICES=2 python supplemental_experiments/02_cub_sub/run.py \
  --manifest /work/Users/leiyo/decaf_supplemental_assets/cub_sub/manifest_per_cell_4_seed_20261109.json \
  --output supplemental_experiments/02_cub_sub/runs/clip_b32_laion2b \
  --model ViT-B-32 --pretrained laion2b_s34b_b79k --device cuda:0

python supplemental_experiments/02_cub_sub/analyze.py \
  --run supplemental_experiments/02_cub_sub/runs/clip_b32_laion2b \
  --epsilon 0.02 --bootstrap 500

python -m pytest supplemental_experiments/02_cub_sub/test_protocol.py -q
```

The default downloader downloads and SHA256-verifies all official Parquet shards
(about 12.1 GB), then extracts selected images to PNG. This is the formal-run
delivery mode. Metadata-only inspection uses efficient byte-range requests.

The optional `--delivery viewer` mode retrieves only selected 512×512 images
from the **official Hugging Face viewer**. These may be JPEG derivatives of the
original PNGs; the manifest
records delivery type and hashes the actual images evaluated. It checks the
viewer revision and row labels against the pinned source and refuses resized
images. It requires no credentials. A future release change causes a clear
failure rather than silently mixing revisions.

If the viewer is unavailable or slow, use the default Parquet mode. Both modes retain
the same label-based selection and pair construction. `--per-cell 0` evaluates
all 38,400 images; the default subset covers all 768 cells. The official
repository does not distribute a trained CBM checkpoint; its README instructs
users to train one. This experiment therefore uses its supported frozen CLIP
route, rather than mislabeling an ImageNet head as a CUB classifier.

`run.py --limit-images N --limit-pairs N` is available for debugging, and sets
`incomplete_smoke=true` in all receipts. It is not a formal result. Use separate
output directories for different runs. The model state, prompt file, source
manifest, and images are hashed; incompatible cache reuse is rejected. Pair
inference is resumable. Changing epsilon requires only rerunning analysis,
not any model queries; use another copy of the run directory to preserve
multiple analysis results.

## Artifacts

- Assets directory: pinned source metadata, complete label index, official
  prompts, downloaded images, selected-image hashes, and explicit pair manifest.
- Run directory: configuration, environment and actual model-state hash,
  single-image scores, pair-level trajectory cache, held-out predictions,
  inference completion receipt, and complete evaluated image/pair lists.
- Analysis: `summary.json`, `single_image_predictions.csv`,
  `single_image_accuracy_by_group.csv`, `pair_scores_and_targets.csv`,
  `paired_metrics.csv`, and `paired_auc_differences.csv`.

Keep conclusions scoped to the measured outcomes. The official substitution
recognition task answers whether the frozen model recognizes the modified
attribute. The matched-pair audit answers whether DECAF's measured response
composition predicts specified external behavior under this protocol. It does
not turn the unpaired release into same-image counterfactual ground truth, nor
does it by itself prove that DECAF outperforms endpoint magnitude.

## Completed run and validation

The completed run is `runs/clip_b32_laion2b/`, summarized in the tracked
[`result_summary.json`](result_summary.json). It used the official original PNG
bytes, all 768 class–attribute cells (four images each), 2,104 predefined pairs,
and frozen OpenCLIP ViT-B-32 LAION-2B weights on one NVIDIA B200. There was no
training. The complete inference involved 53,232 image queries and took 136.5
seconds after model loading; dataset download, checkpoint loading, and CPU
bootstrap analysis are excluded from that timing.

- Modified-attribute recognition accuracy: **50.88%**, with bird-class cluster
  95% interval **46.66–55.23%**.
- For held-out recognition success, E's AUROC is **0.8449**, Abs is **0.8447**,
  and endpoint M is **0.8688**. E does not improve on Abs here and loses to M
  (paired difference −0.0239, 95% interval −0.0346 to −0.0146).
- C's reversal AUROC is **0.8593** on 1,775 active pairs, but only **14** have a
  reversal. The disclosed direction check gives **0.8827** for −M. C's high
  AUROC therefore does not establish information beyond a suitably oriented
  endpoint-margin score (C−(−M) AUROC difference −0.0234, paired 95% interval
  −0.1858 to +0.1027).
- On 329 null pairs with 92 instability positives, **F = Abs** and both have
  AUROC **0.6075**. M has **0.5610**; the F−M confidence interval includes zero.

These results supply the requested SUB experiment, including a VLM scalar-score
application, but do not establish uniform attribution gains or same-image causal
attribute isolation. The paired analyses retain the cross-instance limitation
described above.

All six protocol/cache tests pass, including changed source/preprocessing and
corrupted cache/order regressions; Ruff passes. On **all 2,104 actual response
trajectories**, E, C, F, and Abs match trapezoidal aggregation of the repository's
authoritative `decaf.core.decomposition` output with **zero discrepancy**.
Common-start responses are exactly zero; the maximum conservation discrepancy
is 1.11e−16, and null F−Abs is exactly zero. The full audit is stored as
`canonical_implementation_check.json` in the run directory. Cached scores,
immutable implementation identities, and source snapshots remain available for
reanalysis. The post-hoc direction audit also verifies that every originally
specified metric row is unchanged and all 14 available negative-magnitude
comparisons satisfy `AUROC(-score) = 1 - AUROC(score)` to numerical precision.
