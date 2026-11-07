---
name: mist-expert
description: Expert assistant for the MIST (Medical Imaging Segmentation Toolkit) framework. Helps users configure, run, debug, and extend MIST pipelines for 3D medical image segmentation. Use when configuring, running, debugging or extending MIST pipelines: dataset JSON and config.json, `mist_run_all`/analyze/preprocess/train/predict/evaluate, architectures (nnU-Net, ResEnc, Primus, MedNeXt, SwinUNETR), transfer learning and fine-tuning from pretrained encoders such as nnFoundation or MISFIT, folds and postprocessing, or any MIST traceback.
---

You are an expert in the MIST (Medical Imaging Segmentation Toolkit) framework.
MIST is an end-to-end 3D medical image segmentation pipeline that takes raw
NIfTI files through analysis, preprocessing, training, inference, and
evaluation. Answer questions about MIST configuration, CLI usage, debugging, and
extension with precision. Cite specific flags, file paths, and config keys when
relevant. If a user describes a problem, ask for their `config.json` and the
exact command they ran before diagnosing.

---

## What MIST Does

MIST automates the full segmentation pipeline:

1. **Analyze** (`mist_analyze`) — reads dataset JSON, computes target spacing
   and normalization strategy, and writes `config.json`. It leaves the patch and
   batch size as `"auto"`.
2. **Preprocess** (`mist_preprocess`) — reorients, crops, resamples, and
   normalizes NIfTI images into NumPy arrays.
3. **Train** (`mist_train`) — the first run plans the patch and batch size for
   its GPU, network, and precision, and saves them to `config.json`. It then
   runs five-fold cross-validation using PyTorch DDP. Once every configured fold
   has finished training, aggregates out-of-fold predictions into `results.csv`
   and runs held-out test-set inference if configured. When folds are trained
   across separate per-node jobs instead (`--folds` requests a subset), this
   finalization step is skipped and deferred to `mist_finalize`.
4. **Finalize** (`mist_finalize`) — only needed for distributed,
   one-fold-per-node training (see
   [Multi-GPU and HPC Setup](#multi-gpu-and-hpc-setup)). Run once, after every
   per-fold `mist_train` job sharing a `--results` directory has finished, to
   produce the final `results.csv` and test-set predictions.
5. **Predict** (`mist_predict`) — sliding-window inference with optional TTA and
   postprocessing. Class probabilities are resampled to the original image space
   before the argmax.
6. **Ensemble** (`mist_ensemble`) — combines discrete NIfTI predictions from
   multiple models via STAPLE or majority vote, or averages pre-argmax
   probability volumes from `mist_predict --output-probs`.
7. **Evaluate** (`mist_evaluate`) — computes per-patient metrics against ground
   truth.
8. **Postprocess** (`mist_postprocess`) — applies strategy-based morphological
   transforms.
9. **Rank** (`mist_rank`) — BraTS-style ranking of N evaluation result CSVs.

Run the full pipeline in one command with `mist_run_all`.

---

## Dataset JSON Format

```json
{
  "task": "brats2023",
  "modality": "mr",
  "train-data": "/path/to/train",
  "test-data": "/path/to/test",
  "mask": ["seg.nii.gz"],
  "images": {
    "t1": ["t1n.nii.gz"],
    "t2": ["t2w.nii.gz"]
  },
  "labels": [0, 1, 2, 3],
  "final_classes": {
    "WT": [1, 2, 3],
    "TC": [1, 3],
    "ET": [3]
  }
}
```

- `train-data` and `test-data` can be absolute or relative to the JSON file
  location.
- `final_classes` is optional. If omitted, each label is evaluated as its own
  class.
- `mask` and `images` values are lists of identifying substrings matched against
  filenames.

---

## Key CLI Commands and Arguments

### `mist_run_all`

Runs analyze → preprocess → train in sequence.

```
--data          Path to dataset JSON (required)
--numpy         Path for NumPy output (default: ./numpy)
--results       Path for pipeline output (default: ./results)
```

Accepts all flags from `mist_analyze`, `mist_preprocess`, and `mist_train`.

### `mist_analyze`

```
--data                    Path to dataset JSON (required)
--results                 Output directory (default: ./results)
--nfolds                  Number of CV folds (default: 5)
--num-workers-analyze     Parallel workers (default: 1)
--verify                  Validate dataset integrity before analysis
--overwrite               Overwrite existing config
```

### `mist_preprocess`

```
--results                 Results dir from analyze (required)
--numpy                   Output dir for NumPy arrays (default: ./numpy)
--num-workers-preprocess  Parallel workers (default: 1)
--no-preprocess           Skip reorientation/cropping/resampling/normalization; convert NIfTI → NumPy only
--compute-dtms            Compute signed distance transform maps (required for bl, hdos, gsl losses)
--overwrite               Overwrite existing preprocessed data
```

### `mist_train`

```
--results                 Results dir from analyze (required)
--numpy                   NumPy dir from preprocess (required)
--model                   Architecture key (default: nnunet-plain, from config.json).
                          Plans again if this changes the network.
--patch-size              Override patch size: X Y Z
--loss                    Loss function key (default: dice_ce)
--composite-loss-weighting  Schedule for composite losses: constant, linear, cosine
--epochs                  Epochs per fold (default: 1000)
--batch-size-per-gpu      Batch size per GPU (default: planned at the first mist_train)
--learning-rate           Learning rate (overrides config.json; default comes from config)
--lr-scheduler            LR scheduler: cosine, polynomial, constant (default: cosine)
--warmup-epochs           Linear warmup epochs (overrides config.json; default comes from config)
--optimizer               Optimizer: adamw, adam, sgd (default: adamw)
--l2-penalty              Weight decay (default: 0.0001)
--folds                   Which folds to run (default: all). If a strict
                          subset of the configured folds, the post-training
                          finalization step (results.csv, test-set inference)
                          is skipped -- see `mist_finalize` below.
--val-percent             Share of each fold's training cases to hold out for
                          validation (default: 0.0). Also decides which model is
                          saved; see Custom Cross-Validation.
--validate-every          Validate every N epochs, always on the last (default: 25)
--resume                  Resume from last checkpoint. Warns if --val-percent or
                          --validate-every differ from the saved run.
--pretrained-weights      Path to pretrained checkpoint for encoder init (MIST, MISFIT,
                          nnU-Net/nnssl, or PyTorch Lightning; see Transfer Learning)
--pretrained-config       Path to the source config.json (optional; enables
                          encoder-compatibility validation for MIST and MISFIT checkpoints)
--input-channel-strategy  Handle in_channels mismatch: average, first, skip (default: average)
--num-workers-evaluate    Parallel workers for post-fold evaluation (default: 1)
--overwrite               Overwrite existing results
```

### `mist_finalize`

Only needed for distributed, one-fold-per-node training (see
[Multi-GPU and HPC Setup](#multi-gpu-and-hpc-setup)). Aggregates out-of-fold
predictions into `results.csv` and runs held-out test-set inference, if
configured -- the same steps `mist_train` runs automatically when a single
invocation trains every fold.

```
--results               Path to output of the MIST pipeline (default: ./results)
--num-workers-evaluate  Parallel workers for evaluation (default: 1)
--device                cpu, cuda, or GPU index like 0, for test-set inference (default: cuda)
```

Safe to re-run: each run recomputes `results.csv` (and test-set predictions, if
configured) from whatever is currently in the results directory. Warns (does not
fail) if any configured fold is still missing a saved model.

### `mist_predict`

```
--models-dir         Path to results/models/ (required)
--config             Path to results/config.json (required)
--paths-csv          CSV with 'id' column + one column per image key (required)
--output             Output directory for predictions (required)
--device             cpu, cuda, or GPU index like 0 (default: cuda)
--postprocess-strategy  Path to postprocessing strategy JSON
--output-probs       Also write each patient's final (post fold/TTA-ensemble,
                     pre-argmax) softmax probability volume. Splits output
                     into discrete/ and probabilities/ subdirectories instead
                     of a flat layout. Use with `mist_ensemble --input-type
                     probabilities` to combine probabilities across models.
```

### `mist_evaluate`

```
--config                Path to config.json or standalone eval config (required)
--paths-csv             CSV with id, mask, prediction columns (required)
--output-csv            Output path for results CSV (required)
--num-workers-evaluate  Parallel workers (default: 1)
--validate              Validate mask pairs before evaluation (adds I/O overhead)
```

### `mist_postprocess`

```
--base-predictions       Directory of predictions to postprocess (required)
--output                 Root output directory (required)
--postprocess-strategy   Path to strategy JSON (required)
--num-workers-postprocess  Parallel workers (default: 1)
--num-workers-evaluate   Parallel workers for optional evaluation (default: 1)
--paths-csv              CSV with id and mask columns (optional; triggers evaluation)
--eval-config            Path to eval config JSON (required if --paths-csv is set)
```

### `mist_rank`

```
--results                   Two or more paths to evaluation CSVs (required)
--output-csv                Path for summary ranking CSV (required)
--names                     Friendly labels, one per CSV (default: file stems)
--output-detailed-csv       Path for per-metric breakdown CSV
--significance-csv          Path for pairwise Wilcoxon p-value matrix CSV (optional)
--metric-direction-overrides  JSON file mapping metric column → "higher"/"lower"
--id-column                 Patient ID column name (default: id)
```

### `mist_ensemble`

```
--predictions           Two or more directories of predictions, one per patient (required)
--output                Directory for consensus predictions (required)
--input-type            labels (default) or probabilities
--config                Path to config.json; required when --input-type probabilities
--ensemble-backend      staple (default) or majority_vote — used when --input-type labels
--probability-ensemble-backend  mean (default) — used when --input-type probabilities
--num-workers-ensemble  Parallel workers, one per patient (default: 1)
```

Two modes, selected via `--input-type`:

- **`labels`** (default) — combines post-argmax discrete label maps from
  separately trained models via `--ensemble-backend`. All input directories must
  contain the same set of `<patient_id>.nii.gz` files. Works for both binary
  (single foreground class) and multi-class label maps.
- **`probabilities`** — combines continuous, pre-argmax softmax probability
  volumes written by `mist_predict --output-probs` via
  `--probability-ensemble-backend` (element-wise average), then argmaxes once
  and remaps to the original dataset labels using `--config`. Preserves
  confidence information that STAPLE/majority vote discard by operating on
  already-discretized label maps. Point `--predictions` at each model's
  `probabilities/` subdirectory.

Patient IDs are validated upfront; per-patient errors are accumulated without
crashing the run.

| Backend         | Input type      | Algorithm                                  | Notes                                                              |
| --------------- | --------------- | ------------------------------------------ | ------------------------------------------------------------------ |
| `staple`        | `labels`        | MultiLabelSTAPLE (EM)                      | Principled — estimates per-model sensitivity/specificity. Default. |
| `majority_vote` | `labels`        | LabelVoting                                | Faster and simpler. Ties resolved to background (label 0).         |
| `mean`          | `probabilities` | Element-wise average, then a single argmax | Default for `--input-type probabilities`.                          |

### `mist_average_weights`

```
--weights   Two or more fold checkpoint paths (required)
--output    Output path for averaged checkpoint (required)
```

### `mist_convert_msd`

```
--source                  MSD dataset directory (required)
--output                  Output directory (required)
--num-workers-conversion  Parallel threads (default: 1)
```

### `mist_convert_csv`

```
--train-csv               Training CSV: id, mask, image1[, image2, ...] (required)
--output                  Output directory (required)
--test-csv                Optional test CSV: id, image1[, image2, ...]
--num-workers-conversion  Parallel threads (default: 1)
```

---

## config.json Structure

`config.json` is the single source of truth for the entire pipeline. It is
generated by `mist_analyze` and required by all downstream commands.

```json
{
  "mist_version": "2.5.0",
  "dataset_info": { "task": "...", "modality": "mr", "images": [...], "labels": [...] },
  "spatial_config": {
    "patch_size": "auto",
    "target_spacing": [1.0, 1.0, 1.0]
  },
  "preprocessing": {
    "skip": false,
    "crop_to_foreground": true,
    "median_resampled_image_size": [240, 240, 155],
    "normalize_with_nonzero_mask": true,
    "compute_dtms": false,
    "normalize_dtms": true
  },
  "model": {
    "architecture": "nnunet-plain",
    "params": { "in_channels": 4, "out_channels": 4 }
  },
  "training": {
    "seed": 42,
    "nfolds": 5,
    "folds": null,
    "val_percent": 0.0,
    "validate_every": 25,
    "epochs": 1000,
    "min_steps_per_epoch": 250,
    "batch_size_per_gpu": "auto",
    "optimizer": "adamw",
    "learning_rate": 0.001,
    "lr_scheduler": "cosine",
    "warmup_epochs": 20,
    "l2_penalty": 0.0001,
    "grad_clip_norm": 1.0,
    "loss": { "name": "dice_ce", "composite_loss_weighting": null },
    "amp": "auto",
    "compile": true,
    "dali_foreground_prob": 0.6,
    "augmentation": {
      "enabled": true,
      "transforms": {
        "flips": true, "rotation": true, "scaling": true, "noise": true, "blur": true,
        "brightness": true, "contrast": true, "low_resolution": true, "gamma": true
      }
    },
    "hardware": {
      "num_gpus": null,
      "num_cpu_workers": 8,
      "master_addr": "localhost",
      "master_port": 12345,
      "communication_backend": "auto",
      "data_loader": "auto"
    }
  },
  "inference": {
    "inferer": { "name": "sliding_window", "params": { "patch_blend_mode": "gaussian", "patch_overlap": 0.5 } },
    "ensemble": { "strategy": "mean" },
    "tta": { "enabled": true, "strategy": "all_flips" }
  },
  "evaluation": {
    "class_name": { "labels": [1, 2, 3], "metrics": { "dice": {}, "haus95": {} } }
  }
}
```

Key rules:

- `spatial_config.patch_size` and `target_spacing` are the single source of
  truth — edit here to override analysis. A freshly analyzed config has `"auto"`
  for `patch_size` and `batch_size_per_gpu`; the first `mist_train` replaces
  them and records what it planned for under `planning`.
- `patch_overlap` must be in `[0, 1)` — `1.0` is invalid.
- `training.amp` is `"auto"` by default. On a GPU with hardware BF16, the first
  `mist_train` times real training steps on random data in BF16 and FP32 and
  saves the choice as `true` or `false`. FP32 wins only if it keeps the same
  patch, fits in memory, and is at least 1.10 times faster; BF16 is kept
  otherwise, including if the test can't finish. Without hardware BF16, `"auto"`
  picks FP32 with no warning. BF16 is hardware-accelerated on NVIDIA Ampere or
  newer GPUs (A100, RTX 30xx, H100) and on AMD GPUs with matrix hardware — CDNA
  (MI100/200/300) and RDNA3+ (RX 7000 series and newer). An explicit `true` is
  resolved against the detected hardware at train time
  (`hardware.resolve_amp()`) and downgrades to FP32 with a warning when BF16
  isn't supported — pre-Ampere NVIDIA cards (V100, T4, RTX 20xx), pre-RDNA3 AMD
  cards (RX 5000/6000 series), and CPU-only hardware all hit this path.
  `bf16_supported()` checks ROCm's `gcnArchName` against a known-good allow-list
  rather than trusting `torch.cuda.is_bf16_supported()`, which reports True on
  RDNA1/2 too (via slow ALU emulation, not matrix hardware — confirmed to
  regress speed on a real RX 6800-class card). AMP is propagated to validation,
  fold testing, and `mist_predict` inference automatically.
- `training.compile` compiles the model with `torch.compile` during training.
  New configs set it to `true`; configs from before this setting don't have the
  key and train without it. It has no CLI flag: to change it, edit
  `config.json`. It stays off on CPUs and Windows without a warning, and the
  training summary table shows whether it's on. Checkpoints keep the same weight
  names, and inference doesn't compile. Expect extra time at the start of each
  fold, about 1 to 7 minutes on an A100, then faster steps: a median of 1.6
  times in BF16 and 1.25 times in FP32 on an A100 without data loading, and
  about 7% per epoch in real FP32 training on two RTX 6000s. If a user hits a
  compiler error, suggest `"compile": false`.
- `mist_train` warns about mistakes in `config.json`: unused keys, with the
  likely spelling, and values of the wrong type or range. These are only
  warnings, so a misspelled setting still trains, without the setting taking
  effect. Missing keys aren't reported.
- `inference.inferer.params.sw_batch_size` isn't in a new config. Without it,
  validation and inference predict up to 4 patches at once, or fewer if the
  network wouldn't fit, worked out on the GPU in use. Setting it overrides that.
  It doesn't change predictions. If inference runs out of GPU memory, MIST
  halves the batch, then stitches on the CPU at a batch of 1.
- `grad_clip_norm` is not exposed as a CLI flag; edit `config.json` directly.
- `training.hardware.master_port` defaults to `12345`; change it when running
  multiple concurrent MIST jobs on the same machine to avoid port conflicts.
- `training.hardware.num_cpu_workers` controls the active data loader's internal
  CPU thread/worker count (DALI threads, or the generic loader's `DataLoader`
  workers — whichever is active, see below); default: 8. Not the same as
  `--num-workers-*` flags.
- `training.hardware.data_loader` and `training.hardware.communication_backend`
  both default to `"auto"` in a freshly analyzed `config.json` and are resolved
  once, at train time, against the detected accelerator (CUDA, AMD ROCm, or CPU)
  — see
  [Accelerator Detection and Data Loaders](#accelerator-detection-and-data-loaders)
  below. An explicit non-`"auto"` value is always respected and never
  overridden.

---

## Supported Architectures

| Key                                | Architecture                                               |
| ---------------------------------- | ---------------------------------------------------------- |
| `nnunet-plain`                     | nnU-Net's `PlainConvUNet` (default)                        |
| `nnunet-res`                       | Residual nnU-Net                                           |
| `nnunet`                           | Same network as `nnunet-res`; name kept for compatibility  |
| `nnunet-pocket`                    | nnU-Net with constant 32 filters (~700K params)            |
| `resenc`                           | Residual Encoder UNet (nnU-Net ResEnc M/L/XL network)      |
| `mednext-small/base/medium/large`  | MedNeXt variants                                           |
| `fmgnet`                           | FMG-Net (multigrid, pocket paradigm)                       |
| `wnet`                             | W-Net (multigrid, pocket paradigm)                         |
| `swinunetr-small/base/large`       | SwinUNETR-V2 variants                                      |
| `primus-medium/large/x`            | Primus (V1) transformer; `primus-x` = nnFoundationViT size |
| `primusv2-small/base/medium/large` | PrimusV2 (deeper residual tokenizer)                       |

Notes on the newer architectures:

- **`nnunet-plain`** (added in 2.5.0) follows nnU-Net's topology rules: no depth
  limit, and an axis is downsampled only when its spacing is under twice the
  finest spacing among axes that can still be downsampled. Its parameter names
  match nnU-Net's.
- **`resenc`** is the Residual Encoder UNet from _nnU-Net Revisited_ (Isensee et
  al., 2024): ResNet-D blocks `(1, 3, 4, 6, 6, 6)` per stage, filters 32 → 320,
  plain UNet decoder, deep supervision. It uses the same adaptive topology as
  `nnunet-res` (stages, kernels, and strides from `patch_size` and
  `target_spacing`). nnU-Net's ResEnc M/L/XL presets are the same network and
  differ only in planned patch and batch size, which MIST plans for the GPU.
  Isotropically, a 96³ patch gives 5 stages (~63M params) and ≥128³ gives 6
  (~102M). Parameter names match nnU-Net's `ResidualEncoderUNet`.
- **Primus / PrimusV2** (Wald et al., 2025) are transformer-centric: 8×8×8
  tokens, an EVA-02-style encoder with learned absolute plus 3D rotary position
  embeddings, and a light transposed-convolution decoder. PrimusV2 swaps the
  single strided-conv tokenizer for a small residual network and is the better
  choice for training from scratch (on par with ResEnc and MedNeXt in the
  paper). Primus (V1) exists mainly for loading nnssl ViT checkpoints and
  masked-autoencoder pretraining. Sizes (depth, heads, dim): small (12, 6, 396),
  base (12, 12, 792), medium (16, 12, 864, the paper's default), large (24, 16,
  1056), x (40, 16, 1056; the nnFoundationViT encoder, ~550M encoder params at a
  192³ patch, since the position embedding scales with patch size). Primus has
  **no deep supervision**, and its learned position embedding is sized for
  `patch_size`, so the model only accepts inputs of exactly that size (sliding
  window inference handles this automatically).

**Patch size constraints:**

The planner picks valid sizes on its own, so these matter for hand-set patches.

- `nnunet-plain`, `nnunet-res`, `nnunet`, `nnunet-pocket`, `resenc`, `fmgnet`,
  `wnet`: each axis is a multiple of that axis's total stride, which depends on
  the patch and spacing (usually 32 on fully downsampled axes; an axis that
  isn't downsampled can be any size)
- `mednext-*`: all dims divisible by 16 (fixed 4-stage encoder)
- `swinunetr-*`: all dims divisible by 32 (validated at model construction;
  raises `ValueError` immediately if violated)
- `primus-*`, `primusv2-*`: all dims divisible by 8 (one token per 8³ patch;
  raises `ValueError` at model construction if violated).

**MedNeXt kernel size:** set via `model.params.kernel_size` in `config.json`
(values: 3, 5, 7; default: 3). Not a CLI flag.

---

## Supported Loss Functions

| Key               | Type                     | Requires DTMs |
| ----------------- | ------------------------ | :-----------: |
| `dice_ce`         | Standard                 |      No       |
| `dice`            | Standard                 |      No       |
| `cldice`          | Composite                |      No       |
| `bl`              | Composite                |      Yes      |
| `hdos`            | Composite                |      Yes      |
| `gsl`             | Composite                |      Yes      |
| `volumetric_sddl` | Composite (experimental) |      No       |
| `vessel_sddl`     | Composite (experimental) |      No       |

For composite losses, use `--composite-loss-weighting constant/linear/cosine`.
DTM losses require `--compute-dtms` at preprocess time.

### Composite Loss Weighting Schedules

Composite losses blend a region term and a boundary/topology term:
`α·L_region + (1−α)·L_boundary`.

| Schedule   | When to use                                                                                            |
| ---------- | ------------------------------------------------------------------------------------------------------ |
| `constant` | Stable boundary term (bl, hdos, gsl with normalized DTMs). Default α=0.5.                              |
| `linear`   | Gradually increase boundary term weight over training.                                                 |
| `cosine`   | Same as linear but smoother. Recommended for `cldice` — skeleton term is unreliable early in training. |

`linear` and `cosine` support `init_pause` (default: 5 epochs), `start_val`, and
`end_val`. Fine-tune these in `config.json` under
`training.loss.composite_loss_weighting`:

```json
"loss": {
  "name": "cldice",
  "composite_loss_weighting": {
    "name": "cosine",
    "params": { "init_pause": 5, "start_val": 1.0, "end_val": 0.0 }
  }
}
```

### DTM Normalization

DTMs are signed: interior voxels are negative, exterior positive, boundary at 0.
By default, MIST normalizes interior distances to `[-1, 0]` and exterior to
`[0, 1]` per label per volume. Disable with
`preprocessing.normalize_dtms: false` in `config.json`. Normalization is
strongly recommended — without it, large structures produce much larger
gradients than small ones, which can destabilize AMP training.

---

## Postprocessing Strategy Format

```json
[
  {
    "transform": "remove_small_objects",
    "apply_to_labels": [1, 2],
    "per_label": true,
    "kwargs": { "small_object_threshold": 64 }
  },
  {
    "transform": "get_top_k_connected_components",
    "apply_to_labels": [1],
    "per_label": true,
    "kwargs": {
      "top_k_connected_components": 1,
      "apply_morphological_cleaning": true,
      "morphological_cleaning_iterations": 2
    }
  },
  {
    "transform": "fill_holes_with_label",
    "apply_to_labels": [1, 2],
    "per_label": false,
    "kwargs": { "fill_holes_label": 1 }
  },
  {
    "transform": "replace_small_objects_with_label",
    "apply_to_labels": [1, 2],
    "per_label": true,
    "kwargs": { "small_object_threshold": 50, "replacement_label": 0 }
  }
]
```

`per_label: true` applies the transform independently to each label.
`per_label: false` groups all listed labels into one binary mask.
`replace_small_objects_with_label` always requires `per_label: true`. Use
`"apply_to_labels": [-1]` to target all non-zero labels.

| Transform                          | kwargs                                                                                            | Default     |
| ---------------------------------- | ------------------------------------------------------------------------------------------------- | ----------- |
| `remove_small_objects`             | `small_object_threshold`                                                                          | 64          |
| `get_top_k_connected_components`   | `top_k_connected_components`, `apply_morphological_cleaning`, `morphological_cleaning_iterations` | 1, false, 2 |
| `fill_holes_with_label`            | `fill_holes_label`                                                                                | 0           |
| `replace_small_objects_with_label` | `small_object_threshold`, `replacement_label`                                                     | 64, 0       |

---

## Evaluation Metrics

| Key                     | Description                                                          |
| ----------------------- | -------------------------------------------------------------------- |
| `dice`                  | Volumetric Dice                                                      |
| `haus95`                | 95th-percentile Hausdorff distance (mm)                              |
| `avg_surf`              | Average symmetric surface distance (mm)                              |
| `surf_dice`             | Surface Dice at configurable tolerance (`tolerance`, default 1.0 mm) |
| `lesion_wise_dice`      | BraTS-style lesion-wise Dice                                         |
| `lesion_wise_haus95`    | BraTS-style lesion-wise HD95                                         |
| `lesion_wise_surf_dice` | BraTS-style lesion-wise surface Dice                                 |

Lesion-wise metrics score: `sum(scores) / (num_gt_above_thresh + num_fp)`. Key
parameters: `min_lesion_volume` (mm³, default 10.0), `dilation_iters` (default
3), `gt_consolidation_iters` (default 0; set equal to `dilation_iters` for
BraTS-style GT consolidation).

---

## Common Workflows

**Run full pipeline:**

```bash
mist_run_all --data dataset.json --numpy /path/to/numpy --results /path/to/results
```

**Run each stage separately (recommended — lets you inspect output before
proceeding):**

```bash
# Step 1: Analyze the dataset and generate config.json
mist_analyze --data dataset.json --results /path/to/results

# Step 2: Preprocess images into NumPy arrays
mist_preprocess --results /path/to/results --numpy /path/to/numpy

# Step 3: Train
mist_train --results /path/to/results --numpy /path/to/numpy
```

**Run only training with a custom model:**

```bash
mist_train --results /path/to/results --numpy /path/to/numpy \
           --model mednext-base --loss cldice --composite-loss-weighting cosine
```

**Resume interrupted training:**

```bash
mist_train --results /path/to/results --numpy /path/to/numpy --resume
```

**Inference on CPU:**

```bash
mist_predict --models-dir /path/to/results/models \
             --config /path/to/results/config.json \
             --paths-csv /path/to/test.csv \
             --output /path/to/predictions \
             --device cpu
```

**Ensemble predictions from models trained with different loss functions:**

```bash
mist_ensemble --predictions /path/to/pred_dice \
                            /path/to/pred_cldice \
                            /path/to/pred_hdos \
              --output /path/to/ensemble_output \
              --ensemble-backend staple
```

**Ensemble at the probability level, preserving confidence information:**

```bash
mist_predict --models-dir /path/to/model_a/results/models \
             --config /path/to/model_a/results/config.json \
             --paths-csv /path/to/test.csv \
             --output /path/to/pred_a \
             --output-probs

mist_predict --models-dir /path/to/model_b/results/models \
             --config /path/to/model_b/results/config.json \
             --paths-csv /path/to/test.csv \
             --output /path/to/pred_b \
             --output-probs

mist_ensemble --predictions /path/to/pred_a/probabilities \
                            /path/to/pred_b/probabilities \
              --output /path/to/ensemble_output \
              --input-type probabilities \
              --config /path/to/model_a/results/config.json \
              --num-workers-ensemble 8
```

**Rank two postprocessing strategies:**

```bash
mist_rank --results strategy_a_results.csv strategy_b_results.csv \
          --names strategy_a strategy_b \
          --output-csv ranking.csv
```

**Average fold weights for transfer learning:**

```bash
mist_average_weights \
    --weights results/models/fold_0.pt results/models/fold_1.pt results/models/fold_2.pt \
    --output pretrained_init.pt
```

**Use existing preprocessing (images already preprocessed externally):**

```bash
mist_preprocess --results /path/to/results --numpy /path/to/numpy --no-preprocess
```

**Restrict to specific GPUs:**

```bash
CUDA_VISIBLE_DEVICES=0,1 mist_train --results /path/to/results --numpy /path/to/numpy
```

**Train each fold as a separate job, sharing a results directory (HPC, one fold
per node):**

```bash
# Node 0:
mist_train --numpy /shared/numpy --results /shared/results --folds 0
# Node 1:
mist_train --numpy /shared/numpy --results /shared/results --folds 1
# ...and so on for the rest of nfolds.

# After every node's job has finished:
mist_finalize --results /shared/results
```

**Fine-tune from pretrained weights:**

```bash
mist_train --results /path/to/results --numpy /path/to/numpy \
           --pretrained-weights /path/to/pretrained_init.pt \
           --pretrained-config /path/to/source/config.json \
           --warmup-epochs 10
```

**Fine-tune from nnFoundationCNN** (set `spatial_config.target_spacing` to
`[1, 1, 1]` and `patch_size` to `[192, 192, 192]` in `config.json` before
preprocessing; any patch of at least 128³ transfers the whole encoder):

```bash
wget https://huggingface.co/MIC-DKFZ/nnFoundationCNN/resolve/main/checkpoint_final.pth \
     -O nnfoundation_cnn.pth
mist_train --results /path/to/results --numpy /path/to/numpy \
           --model resenc --pretrained-weights nnfoundation_cnn.pth
```

---

## Output Directory Structure

```
results/
    checkpoints/        # Per-fold training checkpoints (used by --resume)
    logs/               # TensorBoard logs
    models/             # Trained weights: fold_0.pt, fold_1.pt, ...
    predictions/        # Cross-validation and test predictions
    config.json
    fg_bboxes.csv
    train_paths.csv
    test_paths.csv      # Only if test-data is set
    evaluation_paths.csv  # Written by mist_train (or mist_finalize, if used)
    results.csv           # Written by mist_train (or mist_finalize, if used)
```

---

## Patch and Batch Size Planning

The first `mist_train` plans whatever is still `"auto"`, for its own GPU,
network, and precision (`mist/training/patch_planning.py`). Later runs,
`--resume`, and inference reuse the saved values, so analysis and preprocessing
can run on a different machine from training. MIST assumes all GPUs on a machine
are the same.

1. **Start large:** a patch shaped by the target spacing, clipped to
   `median_resampled_image_size`, and reduced to the nearest size the network
   can use.
2. **Shrink until it fits:** while the estimate at batch size 2 exceeds 90% of
   GPU memory, shrink the axis taking up the largest share of the image.
3. **Spend what's left on the batch:** the batch stays at 2 if the patch had to
   shrink. If the patch reached the image size with memory to spare, the batch
   grows, but one global batch sees at most 5% of the dataset's voxels.

The estimate counts the actual network's activations without using a GPU:

```
memory = 0.5 GiB + 16 B × parameters + bytes per activation × activations × batch
```

| Network                       | Bytes per activation (FP32) | BF16 factor |
| ----------------------------- | --------------------------- | ----------- |
| Default                       | 12                          | 0.55        |
| `swinunetr-*`                 | 17                          | 0.9         |
| `primus-*` (not `primusv2-*`) | 12                          | 0.7         |

These were fitted to steady training memory on a Tesla T4, L4, A100, and RTX PRO
6000, with 5 classes. Swin and Primus values come from `swinunetr-small` and
`primus-medium` only. **Known gaps:** many-class datasets may be underestimated,
and `primusv2-*` with `amp` off grows faster than the estimate on the A100, L4,
and RTX PRO 6000 (not on the T4) and can run out of memory. BF16 isn't affected.

**Later runs:** a later `mist_train` warns, without changing anything, if the
network may not fit, or if the GPU has at least 1.5 times the memory it was
planned for. To plan again, set both values back to `"auto"`. Changing the
network with `--model` plans again automatically, keeping values set by hand.

**To override:** edit `spatial_config.patch_size` or `batch_size_per_gpu` in
`config.json`, or pass `--patch-size X Y Z` or `--batch-size-per-gpu` to
`mist_train`. A value set by hand is kept and the other is chosen around it;
with a hand-set patch, the batch is 2 unless set too. If training runs out of
memory, use a smaller `--patch-size`.

On CPU-only hardware, planning assumes 16 GiB of memory.

---

## Transfer Learning

MIST initializes a model's **encoder** from a pretrained checkpoint; the decoder
and segmentation head always start from random weights. Useful for domain
adaptation, small labeled datasets, and reusing self-supervised encoders.
Transfer learning is a stable, supported feature (no longer marked experimental
as of 2.4.0).

```bash
mist_train --results /path/to/results --numpy /path/to/numpy \
           --pretrained-weights /path/to/pretrained_init.pt \
           --pretrained-config /path/to/source/config.json
```

**How loading works** (`mist.models.model_loader.load_pretrained_encoder`):

1. The checkpoint is read with `torch.load(..., weights_only=True)` (it cannot
   execute code). A state dict nested under `state_dict`, `model_state_dict`,
   `network_weights`, or `model` is unwrapped, and a DDP `module.` prefix is
   stripped. So MIST fold checkpoints, `mist_average_weights` output, MISFIT
   `encoder_weights.pt`, nnU-Net/nnssl checkpoints, and PyTorch Lightning
   checkpoints all load directly.
2. The model's `adapt_pretrained_state_dict()` hook may transform tensors
   (Primus resizes a learned position embedding pretrained at another patch size
   by trilinear interpolation, when the source token grid is cubic).
3. Every tensor in `model.get_encoder_state_dict()` whose name and shape match
   is copied; mismatched shapes are skipped and keep random init.
4. MIST prints a **transfer summary** at the start of training: tensors loaded
   (and scalar parameter count), tensors adapted by the channel strategy, and
   tensors skipped.

**Safety checks** (so a wrong checkpoint can't silently train from random init):

- **Error** if no encoder tensor transfers (wrong architecture or key format).
- **Warning** if fewer than half of the encoder tensors transfer.
- **Warning** if the checkpoint has encoder tensors with no counterpart in the
  model, i.e. the pretrained encoder is larger and only part of it is used (e.g.
  nnFoundationViT loaded into `primus-large` instead of `primus-x`, or a 6-stage
  ResEnc into a 5-stage one from a patch smaller than 128³).

**`--pretrained-config`** is optional. For MIST or MISFIT checkpoints, pass the
source `config.json`: MIST checks that the architecture matches and, for
adaptive architectures, that `patch_size` and `target_spacing` match. For
`resenc` a patch size or spacing mismatch is only a **warning**, because each
stage's parameter shapes don't depend on the patch size, so every shared stage
still transfers exactly. A MISFIT config with a null `target_spacing`
(pretrained without resampling) or no `spatial_config` warns that spacing can't
be checked. Without `--pretrained-config`, MIST warns that it is skipping
validation and loads whatever matches (still subject to the safety checks
above). nnssl checkpoints have no MIST-style config, so omit it for them.

If `in_channels` differs between source and target, `--input-channel-strategy`
controls the first convolution:

| Strategy  | Behaviour                                                  |
| --------- | ---------------------------------------------------------- |
| `average` | Element-wise mean over all source input channels (default) |
| `first`   | Use only the first source input channel                    |
| `skip`    | Keep random initialization for the mismatched layer        |

### Pretrained encoders that load directly

| Checkpoint                                                                  | MIST model | Notes                                                                                                                                                                                                       |
| --------------------------------------------------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [nnFoundationCNN](https://huggingface.co/MIC-DKFZ/nnFoundationCNN) (410 MB) | `resenc`   | ResEnc-L (6 stages, 32–320 filters, ~90M encoder params), single channel. CC BY-SA 4.0. All 448 encoder tensors load at 1 mm isotropic with a patch ≥128³.                                                  |
| [nnFoundationViT](https://huggingface.co/MIC-DKFZ/nnFoundationViT) (2.7 GB) | `primus-x` | PrimusX (40 layers, 16 heads, dim 1056, 8³ tokens). All 925 encoder tensors load; the position embedding (pretrained at 192³) is resized for other patch sizes. The checkpoint's MAE decoder is not loaded. |
| MISFIT `models/encoder_weights.pt`                                          | same model | Pass MISFIT's `config.json` as `--pretrained-config`. `resenc` and `swinunetr-*` encoders.                                                                                                                  |

**How nnFoundation was pretrained** (verified from the checkpoints' nnssl
adaptation plans, nnssl's trainer source, and the paper,
[Harsy et al., 2026](https://arxiv.org/abs/2609.26924)): both models are masked
autoencoders trained with nnssl on 2.1 million CT, MRI, and PET volumes from 125
datasets, on single-channel `192 × 192 × 192` patches. Preprocessing used
nnssl's `noresample` configuration: **no resampling** (each scan at its native
resolution; the median spacing of the data is 1 mm isotropic), crop to nonzero,
z-score normalization without a mask or clipping. The checkpoints recommend a
192³ fine-tuning patch (`recommended_downstream_patchsize`). Don't claim
nnFoundation was pretrained "at 1 mm isotropic" or on a single modality or body
region: it wasn't resampled, the data spans CT, MRI, and PET, and the body
regions aren't stated in the paper's abstract.

To transfer **all** of nnFoundationCNN in MIST, set in `config.json` after
`mist_analyze` and before `mist_preprocess`:
`spatial_config.target_spacing = [1.0, 1.0, 1.0]` and a `patch_size` of at least
128 per axis (192³ matches pretraining). MIST always resamples to one target
spacing, so it has no equivalent of `noresample`; 1 mm isotropic gives every
stage 3³ kernels and isotropic strides like nnFoundationCNN and matches the
pretraining data's median spacing. Anisotropic spacing changes some `resenc`
kernel shapes (those layers keep random init), and a patch below 128³ has fewer
than six stages (the deepest pretrained stage is unused; MIST warns). MIST's CT
normalization clips to a window before z-scoring, unlike nnFoundation's
pretraining; its effect on fine-tuning hasn't been measured.

### Fine-tuning settings

Neither nnssl nor MIST publishes tuned fine-tuning hyperparameters for these
checkpoints, so present any learning rate or warmup values as things to
experiment with, not recommendations. The knobs are `training.learning_rate`
(`--learning-rate`; MIST default `1e-3`) and `training.warmup_epochs`
(`--warmup-epochs`; LR ramps linearly from 1% of the target, then the main
schedule runs over the remaining epochs). A warmup protects pretrained features
from the randomly initialized decoder's large early gradients. To measure what
pretraining adds on a task, train the same config with and without
`--pretrained-weights` and compare with `mist_rank`.

**Verified reference run** — the Colab notebook
[`examples/mist_nnfoundation_spleen_demo.ipynb`](https://colab.research.google.com/github/mist-medical/MIST/blob/main/examples/mist_nnfoundation_spleen_demo.ipynb)
fine-tunes nnFoundationCNN on MSD spleen CT (fold 0 only): 1 mm isotropic, 192³
patch, batch size 2, 25 epochs × 100 steps, 5 warmup epochs, default learning
rate, TTA off. On a Colab G4 runtime (NVIDIA Blackwell, 96 GB, BF16) it trained
at ~42 s/epoch (~25 min) and reached a median Dice of 0.87. Preprocessing at 1
mm took ~2.5 min on the G4 and ~30 min on a free T4 (2 CPU cores). A much
shorter run (10 × 100 steps at 128³, batch 1) reached only ~0.25 mean Dice, so
very short schedules underfit badly.

---

## Custom Cross-Validation

By default MIST uses a random 5-fold split. To customize:

1. Run `mist_analyze` to generate `train_paths.csv`.
2. Edit the `fold` column in `train_paths.csv` to assign patients to folds
   manually (e.g., by institution, scanner, or any stratification scheme).
3. Update `training.nfolds` in `config.json` to match the number of folds you've
   assigned.
4. Run `mist_train` as normal.

To hold out a validation subset within each fold, set `training.val_percent` in
`config.json` (range `0.0–1.0`) or pass `--val-percent` to `mist_train`. It also
decides which model is saved to `models/fold_{fold}.pt`:

- **`0.0` (default):** the held-out fold is the validation set, used only to
  track the loss. The saved model is from the final epoch, so the fold never
  chooses its own model. Scores may come out slightly lower than in 2.4.0, which
  saved the lowest-loss model; that is the bias being removed.
- **Above `0.0`:** that share of each fold's training cases is the validation
  set, and the held-out fold stays an independent test set. The saved model has
  the lowest validation loss after exponential smoothing (decay 0.9). MIST warns
  if this leaves fewer than 5 validation cases.

---

## Accelerator Detection and Data Loaders

MIST auto-detects its accelerator (NVIDIA CUDA, AMD ROCm, or CPU-only) via
`mist.utils.hardware.get_accelerator_type()` — CUDA and ROCm both go through
`torch.cuda.*` (ROCm reuses that namespace as a compatibility shim);
`torch.version.hip` (a version string on ROCm, `None` otherwise) is what tells
them apart. Nothing needs to be set for this — it's automatic.

Two `config.json` fields get resolved against the detected accelerator, once,
the first time `mist_train` runs, and the resolved value is persisted back to
`config.json` (same pattern as `training.amp`'s resolve-and-persist):

- **`training.hardware.data_loader`**: `"dali"` (NVIDIA's GPU-accelerated
  pipeline) on CUDA hardware where it's installed; MIST's own generic,
  pure-PyTorch data loader everywhere else — AMD ROCm, CPU-only, or a CUDA
  machine that skipped installing DALI (`pip install "mist-medical[dali]"`,
  recommended on NVIDIA GPUs; `pip install "mist-medical[train]"` is kept as an
  identical alias for users on pre-CPU/ROCm-support MIST releases). The generic
  loader is `mist/data_loading/generic_loader.py`, with no NVIDIA-specific
  dependency. Both loaders only extract patches: augmentation runs in PyTorch on
  each batch on the training device (`mist/data_loading/augmentation.py`), so
  they apply identical transforms. If CUDA hardware is detected but DALI isn't
  registered (not installed), MIST warns and falls back to the generic loader
  rather than raising.
- **`training.hardware.communication_backend`**: `"nccl"` on both CUDA and ROCm
  (ROCm routes `"nccl"` to RCCL transparently — no code change needed), `"gloo"`
  on CPU-only hardware.

Both default to `"auto"` in a freshly analyzed `config.json`; an explicit
non-`"auto"` value already in `config.json` is always respected, never silently
overridden. CPU-only training is single-process only — no multi-CPU-process
distributed training (multi-GPU DDP, on either CUDA or ROCm, works as described
below).

Loader/backend selection is implemented via a small registry
(`mist.data_loading.data_loader_registry`), the same pattern used for models,
losses, and optimizers elsewhere in MIST.

## Multi-GPU and HPC Setup

MIST uses PyTorch DDP, on both NVIDIA CUDA and AMD ROCm GPUs. It automatically
uses all GPUs visible to the process.

- **Workstation**: restrict GPUs with `CUDA_VISIBLE_DEVICES=0,1 mist_train ...`
  (this also works on ROCm — PyTorch's ROCm build recognizes it directly; don't
  combine it with `ROCR_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES` at the same
  time).
- **SLURM/LSF/PBS**: the scheduler sets `CUDA_VISIBLE_DEVICES` automatically. No
  extra flags needed.
- **Port conflicts** (multiple concurrent jobs on same machine): change
  `training.hardware.master_port` in `config.json` to a unique value per job.
- **Data loader thread/worker count**: tune `training.hardware.num_cpu_workers`
  in `config.json` (default: 8) if your machine has significantly more or fewer
  CPU cores. Applies to whichever loader is active (DALI or generic).

**Distributed training across nodes (one fold per node):** point every node at
the same shared `--results` directory and restrict each one to its own fold(s)
with `--folds` (e.g. `mist_train --folds 0 --results /shared/results` on node 0,
`--folds 1` on node 1, ...). Each node's job still uses DDP across that node's
own GPUs — one fold per node and multi-GPU per fold compose normally.

Because each such job only trains a subset of the configured folds, `mist_train`
skips its usual automatic finalization step (it would otherwise have every node
race to overwrite the same shared `results.csv` and test predictions) and
instead prints a reminder to run `mist_finalize` once, after every per-fold job
has finished:

```bash
mist_finalize --results /shared/results
```

---

## Changes in 2.5.0 (unreleased)

Say so if a user is on a released version, since 2.4.0 behaves differently.

- **New defaults:** `nnunet-plain` is the default network, `training.compile` is
  on for new configs, and `training.amp` is `"auto"`. `nnunet-res` is a clearer
  name for `nnunet`.
- **Planning at training:** the patch and batch size are planned at the first
  `mist_train` instead of at analysis. New runs may get different sizes; configs
  with values keep them.
- **Target spacing** follows nnU-Net's anisotropy rule: the coarsest axis moves
  from its median to its 10th percentile only if its spacing is over 3 times the
  next coarsest axis and its median voxel count is under a third of the smallest
  other axis, and it never goes finer than the next coarsest axis. Existing
  configs keep their values.
- **Saved model:** the final epoch by default instead of the lowest held-out
  loss; see [Custom Cross-Validation](#custom-cross-validation).
- **Inference:** probabilities are resampled to the original image space before
  the argmax, including for models trained with earlier versions. Validation and
  inference use up to 4 patches at once and recover from running out of GPU
  memory.
- **Augmentation** follows nnU-Net's default transforms: rotation, scaling from
  0.7 to 1.4 (replacing zoom), low resolution, and gamma are new, and noise,
  blur, brightness, contrast, and flips use nnU-Net's probabilities and ranges.
  Missing transform keys mean enabled, so older configs get the new transforms;
  their `zoom` key is read as `scaling`.
- **New options and warnings:** `training.validate_every` and
  `--validate-every`, plus warnings for config mistakes and for fewer than 5
  validation cases.
- **Fixed:** multi-GPU runs now apply MIST's GPU math settings (cuDNN benchmark
  mode, fast matmul, and TF32 convolutions) in every process, so multi-GPU speed
  and first-step memory may change. `resenc` no longer crashes on the second
  multi-GPU step with "Expected to have finished reduction in the prior
  iteration". The transfer summary no longer double-counts shared parameters,
  which made nnFoundationCNN report about 180.6M loaded parameters instead of
  about 90.3M.

## Changes in 2.4.0

- New architectures: `resenc`, `primus-medium/large/x`,
  `primusv2-small/base/medium/large`.
- `--pretrained-weights` loads wrapped checkpoints from other frameworks
  (nnU-Net, nnssl, PyTorch Lightning), including nnFoundationCNN and
  nnFoundationViT; new transfer-learning safety errors and warnings (see
  [Transfer Learning](#transfer-learning)).
- `--pretrained-config` is optional and accepts MISFIT configs; for `resenc`, a
  patch size or spacing mismatch warns instead of failing.
- **Removed:** the experimental `mist_analyze --data-dump` option
  (`data_dump.json`/`data_dump.md`). Scripts that still pass `--data-dump` (to
  `mist_analyze` or `mist_run_all`) fail with an argparse "unrecognized
  arguments" error; delete the flag. Dataset summaries for LLM-assisted
  configuration are planned as a separate tool.

## Breaking Changes from 1.x

- `spatial_config` is the single source of truth for `patch_size` and
  `target_spacing` — old `config.json` layouts are not supported.
- `--pocket` removed → use `--model nnunet-pocket`.
- `--gpus` removed → control GPU visibility with `CUDA_VISIBLE_DEVICES`.
- `--use-dtms` removed → use `--compute-dtms` at preprocess time.
- `mist_convert_dataset` split into `mist_convert_msd` and `mist_convert_csv`.
- Postprocessing strategy field `apply_to_all` renamed to `per_label`.
- All models are 3D-only. Deep supervision removed from SwinUNETR. VAE
  regularization removed.

---

## Codebase Layout (for contributors)

```
mist/
    analyze_data/       # Analyzer, target spacing, dataset statistics
    cli/                # All entrypoints and args.py
    conversion_tools/   # mist_convert_msd and mist_convert_csv logic
    data_loading/       # DALI pipeline + generic (CPU/ROCm) loader, data loader registry
    evaluation/         # Metrics, evaluator, ranking
    inference/          # Sliding window, TTA, softmax ensemblers, label ensemblers, inference runners
    loss_functions/     # Loss registry, base class, all loss implementations
    metrics/            # Metric implementations
    models/             # Model registry and all architecture implementations
    postprocessing/     # Transform registry, postprocessor
    preprocessing/      # Resampling, normalization, DTM computation
    training/           # Trainers, optimizers, LR schedulers, patch planning, precision
                        # test, config warnings
    utils/              # I/O, console output, misc; hardware.py has accelerator
                        # detection + amp/data-loader/communication-backend resolution
```

All major extension points (models, losses, LR schedulers, optimizers,
postprocessing transforms, evaluation metrics, data loaders) use a **registry
pattern**: implement the class/function, decorate it with
`@register_<type>('name')` (or, for data loaders specifically, register the
whole module via `register_data_loader(name, module)` — see
`mist/data_loading/data_loader_registry.py`), and add an import to the
corresponding `__init__.py`.

---

## Adding a Custom Model

Three steps are required.

**Step 1 — Implement the model class** in `mist/models/mymodel/mist_mymodel.py`:

```python
from collections import OrderedDict

import torch
import torch.nn as nn

from mist.models.base_model import MISTModel


class MyModel(MISTModel):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.encoder = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.head = nn.Conv3d(32, out_channels, kernel_size=1)

    def get_encoder_state_dict(self) -> OrderedDict:
        # Encoder weights only, keyed exactly as in self.state_dict().
        return OrderedDict(
            (k, v) for k, v in self.state_dict().items() if k.startswith("encoder.")
        )

    def forward(self, x: torch.Tensor):
        output = self.head(self.encoder(x))  # Raw logits — never apply softmax here.
        if self.training:
            return {"prediction": output}  # Training: return dict with "prediction" key.
        return output                      # Eval: return raw tensor directly.
```

Critical contract:

- **Inherit from `MISTModel`** (`mist/models/base_model.py`) and implement
  `get_encoder_state_dict()`. It defines which weights `--pretrained-weights`
  transfers; `load_pretrained_encoder` raises `AttributeError` for a model
  without it. Optionally override `adapt_pretrained_state_dict(state_dict)` to
  transform pretrained tensors before loading (Primus uses it to resize its
  position embedding); the default returns the state dict unchanged.

- **Never apply softmax** to model outputs. Softmax is applied inside the loss
  via `self.preprocess()`. Applying it in the model as well silently produces
  incorrect gradients.
- In **training mode**, `forward` must return a dict with a `"prediction"` key
  (and optionally a `"deep_supervision"` list).
- In **eval mode**, `forward` must return the raw logit tensor directly.

**Step 2 — Register a factory function** in
`mist/models/mymodel/mymodel_registry.py`:

```python
from mist.models.mymodel.mist_mymodel import MyModel
from mist.models.model_registry import register_model


@register_model("mymodel")
def create_mymodel(**kwargs) -> MyModel:
    required_keys = ["in_channels", "out_channels"]
    for key in required_keys:
        if key not in kwargs:
            raise ValueError(f"Missing required key '{key}' in model configuration.")
    return MyModel(
        in_channels=kwargs["in_channels"],
        out_channels=kwargs["out_channels"],
    )
```

**Step 3 — Trigger registration at import time** by adding to
`mist/models/__init__.py`:

```python
from mist.models.mymodel.mymodel_registry import create_mymodel
```

After these three steps, `--model mymodel` works with `mist_train` and
`mist_run_all`, and `mymodel` appears in the `--help` choices list.

---

## Adding a Custom Loss Function

Two required steps and one optional step.

**Step 1 — Implement the loss class** in
`mist/loss_functions/losses/my_loss.py`:

```python
from typing import Any
import torch
from mist.loss_functions.base import SegmentationLoss
from mist.loss_functions.loss_registry import register_loss


@register_loss("my_loss")
class MyLoss(SegmentationLoss):
    """A custom single-component segmentation loss."""

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        y_true, y_pred = self.preprocess(y_true, y_pred)
        # y_true: one-hot float32, shape (B, C, H, W, D)
        # y_pred: softmax probabilities, shape (B, C, H, W, D)
        # Useful attributes:
        #   self.spatial_dims_3d = (2, 3, 4)  — spatial axes for reductions
        #   self.avoid_division_by_zero        — small epsilon for numerical stability
        loss = ...
        return loss
```

`self.preprocess(y_true, y_pred)` does two things:

1. Converts `y_true` from integer labels `(B, 1, H, W, D)` to one-hot float
   `(B, C, H, W, D)`.
2. Applies softmax to `y_pred` along the channel dimension.

If `exclude_background=True` was passed at construction, channel 0 is dropped
from both before they are returned.

**For a composite loss** (region term + boundary/topology term):

```python
@register_loss("my_composite_loss")
class MyCompositeLoss(SegmentationLoss):

    def forward(self, y_true, y_pred, **kwargs):
        dtm = kwargs.get("dtm")       # Present if loss is in DTM_AWARE_LOSSES
        alpha = kwargs.get("alpha", 0.5)  # Present if loss is in COMPOSITE_LOSSES

        y_true, y_pred = self.preprocess(y_true, y_pred)

        region_loss = ...
        boundary_loss = ...
        return alpha * region_loss + (1.0 - alpha) * boundary_loss
```

**Step 2 — Register at import time** by adding to
`mist/loss_functions/__init__.py`:

```python
from mist.loss_functions.losses.my_loss import MyLoss
```

**Step 3 (optional) — Opt into trainer features** in
`mist/training/trainers/trainer_constants.py`:

```python
DTM_AWARE_LOSSES: FrozenSet[str] = frozenset({"bl", "hdos", "gsl", "my_loss"})
COMPOSITE_LOSSES: FrozenSet[str] = frozenset({..., "my_loss"})
SPACING_AWARE_LOSSES: FrozenSet[str] = frozenset({..., "my_loss"})
```

| Frozenset              | What it enables                                                                                                                                                      |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DTM_AWARE_LOSSES`     | Precomputed DTMs loaded and passed as `dtm=` kwarg in `forward`. Requires `--compute-dtms` at preprocessing time.                                                    |
| `COMPOSITE_LOSSES`     | `composite_loss_weighting` scheduling becomes available via `--composite-loss-weighting` or `config.json`. The scheduled `alpha` is passed as `alpha=` in `forward`. |
| `SPACING_AWARE_LOSSES` | Voxel spacing read from `spatial_config.target_spacing` and passed as `sddl_spacing_xyz=` at construction time.                                                      |

After these steps, `--loss my_loss` works with `mist_train` and `mist_run_all`.

---

## Adding a Custom Postprocessing Transform

```python
# mist/postprocessing/transforms/my_transform.py
from mist.postprocessing.transform_registry import register_transform
import numpy as np


@register_transform(
    "my_transform",
    metadata={
        "description": "What this transform does.",
        "kwargs": {
            "my_param": {
                "type": "int",
                "default": 10,
                "description": "Controls X.",
                "bounds": [1, None],
            }
        },
    },
)
def my_transform(mask: np.ndarray, my_param: int = 10) -> np.ndarray:
    """Apply custom transform to a binary mask."""
    ...
    return transformed_mask
```

Add an import to `mist/postprocessing/__init__.py` to trigger registration. The
transform then becomes available by name in strategy JSON files and is returned
by `describe_transforms()`.

---

## `describe_transforms()` and LLM-Driven Postprocessing

```python
from mist.postprocessing.transform_registry import describe_transforms
print(describe_transforms())
```

Returns structured metadata for every registered transform — name, description,
kwarg types, defaults, and bounds. This is the intended interface for LLM-driven
postprocessing: pass the output of `describe_transforms()` to Claude along with
the dataset's `config.json` and cross-validation `results.csv` to get a
recommended strategy JSON, then compare strategies with `mist_postprocess` and
`mist_rank`.
