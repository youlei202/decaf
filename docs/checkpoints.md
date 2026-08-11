# Checkpoint inventory

Model weights are not redistributed in this repository. Static checkpoint
contracts live in `manifests/checkpoints/`; cache authorized weights beneath
`$DECAF_CACHE_ROOT/checkpoints/<family>`. The loader validates every available
expected digest before deserialization.

## Availability classes

- `downloadable`: an official framework or upstream project provides the
  weights. Its terms still govern use.
- `manual`: upstream access exists but there is no frozen, redistributable
  download route. Supply an authorized local file and match its digest.
- `generated`: the paper pipeline creates the weights from source data and a
  frozen plan. Historical bytes may not be publicly available.
- `restricted`: the weights derive from restricted training data and are not
  redistributed here.

Only deserialize checkpoints obtained from a trusted source. A matching digest
establishes byte identity, not trustworthiness or permission to use the file.

## Controlled family

The base suite contains 30 author-trained checkpoints. Evidence selection fixes
88 further states: 52 endpoint-evidence, 18 causal-direction, and 18 fragility
checkpoints. Context swap contributes 30 author-trained checkpoints. Lightweight
reference packages retain manifests and individual selected-checkpoint hashes,
but not the model bytes or optimizer state. Regenerate these models or provide
an authorized sealed bundle; there is no public download claim.

Paper preparation expects `model_manifest.csv`,
`endpoint_behavior_model_manifest.csv`, and `context_swap_model_manifest.csv`
under `$DECAF_CACHE_ROOT/checkpoints/controlled`. C1 and C2 rows include
`model_id`, `checkpoint_path`, `checkpoint_sha256`, and
`producer_member_id`; C1 additionally records its module, variant,
architecture, seed, and selection flag. The loader requires exact configured
IDs, validates every checkpoint byte, and checks that each producer points to
the corresponding C1 or C2 training member. The 52 evidence snapshots come
from eight trajectories, so C1 has 44 factory jobs rather than 88 independent
training jobs. C0 remains strictly no-retraining and also verifies its frozen
probability-cache hashes.

## ImageNet-9 family

The model zoo contains 72 descriptors: 24 upstream pretrained models (12
`torchvision` and 12 `timm`) and 48 experiment fine-tunes. The upstream
pretrained weights are fetched through their official APIs and pinned by the
resolved model manifest. The 48 fine-tuned weights are generated artifacts and
are not bundled. The frozen deep subset contains 32 models.

## Attribution family

Public framework downloads cover torchvision ResNet-50, ConvNeXt-Large, and
Swin-B plus DINOv2-L and DINOv2-g backbones and linear heads. Exact expected
hashes are in `manifests/checkpoints/attribution.yaml`.

The aligned IDSDS suite uses three exact ImageNet-trained files:

| Architecture | Expected filename | SHA-256 |
|---|---|---|
| ResNet-50 | `resnet50_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar` | `dc4b6f9424ca154e5fa27aa5f574e4d7d94e2c969979c66ed978d4ef9eb799b4` |
| VGG-16 | `vgg16_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar` | `ec5aad9340d467f6375f784336e3a083e4ee50abc53ecdc8209754d4841f78a4` |
| ViT-B/16 | `vit_base_patch16_224_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar` | `858fb793a1debb2e03254545e2a57f7533ea9a07a1d2445706afec27e3985033` |

They originate from the IDSDS implementation pinned to revision
`8e842009423f14ac790b352b1f86846cc381415c`. Because they derive from
ImageNet training, they are treated as restricted/manual and are not
redistributed. FunnyBirds model hashes are frozen, but their immutable public
download locations were not; they are also manual inputs.

## Covertype family

The paper grid generates 135 joblib checkpoints: 90 causal-direction models and
45 fragility models. The lightweight reference archive freezes the 135-row model
manifest, not the large checkpoint bytes, and the static public source does not
contain per-file historical hashes. Retrain them from the frozen split and plan;
do not describe an arbitrary local copy as verified.

## Failure and cache behavior

Automatic downloads use a temporary file in the destination directory, validate
the expected byte count and digest when supplied, and atomically rename the
file. A mismatch is quarantined or removed and reported; it is never accepted
into the cache. Manual and restricted entries fail early with the required
filename, expected hash, and upstream terms rather than prompting interactively.
