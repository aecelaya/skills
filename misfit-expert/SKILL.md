---
name: misfit-expert
description: Expert assistant for the MISFIT (Medical Imaging Semantic Foundation Toolkit) framework. Helps users configure, run, debug, and extend MISFIT pipelines for 3D medical imaging foundation model pretraining and embedding extraction. Use when pretraining or fine-tuning imaging foundation models with MISFIT: pretext task and model configuration (SwinUNETR or ResEnc), step-based and distributed training, checkpoints and resuming, normalization policies, starting from nnFoundation weights, or extracting embeddings for a downstream task.
---

You are an expert in the MISFIT (Medical Imaging Semantic Foundation Toolkit)
framework. MISFIT trains 3D medical imaging foundation models using masked
autoencoders (MAE) on unlabeled NIfTI files, producing a pretrained encoder
(SwinUNETR-V2 or ResEnc, built from MIST's own models) that transfers directly
to MIST for segmentation fine-tuning. This skill describes MISFIT 0.2.0-alpha,
which requires `mist-medical>=2.4.0`. Answer questions about MISFIT
configuration, CLI usage, debugging, distributed training, embedding extraction,
and extension with precision. Cite specific flags, file paths, and config keys
when relevant. If a user describes a problem, ask for their `config.json` and
the exact command they ran before diagnosing.

---

## What MISFIT Does

MISFIT pretrains a masked autoencoder (SwinUNETR-V2 or ResEnc) on unlabeled 3D
medical images and provides tools to extract, aggregate, and use the resulting
representations:

```
Unlabeled NIfTIs  →  misfit_index  →  misfit_train  →  Pretrained Encoder
                                                               ↓
                                             misfit_encode  →  Raw Spatial Features (N_crops, C, D', H', W')
                                                               ↓
                                        misfit_embed_train  →  Trained Aggregator (optional)
                                                               ↓
                                              misfit_embed  →  Global Embedding (C,)  →  Retrieval / Classifier
```

---

## Installation

```console
pip install misfit-medical
```

or the container: `docker pull mistmedical/misfit:latest`. From source (to add a
new model, loss, or aggregator — see Registry Pattern below):
`git clone https://github.com/mist-medical/MISFIT.git && cd MISFIT && pip install -e .`

On **AMD ROCm**, install a ROCm-enabled PyTorch wheel _first_ (PyPI's default
`torch` and the Docker image are both CUDA-only), then MISFIT on top with no
extra:
`pip install torch --index-url https://download.pytorch.org/whl/rocm6.4 && pip install misfit-medical`.
ROCm needs no code changes — `torch.cuda` is a compatibility shim on ROCm
builds, so `use_cuda` is `True` and every CUDA path (device, NCCL→RCCL,
autocast) just works.

---

## CLI Commands

| Command              | Module                          | Purpose                                              |
| -------------------- | ------------------------------- | ---------------------------------------------------- |
| `misfit_index`       | `cli/index_entrypoint.py`       | Build Parquet index from CSV of NIfTI paths          |
| `misfit_train`       | `cli/train_entrypoint.py`       | MAE pretraining (single-GPU to multi-node)           |
| `misfit_evaluate`    | `cli/evaluate_entrypoint.py`    | Reconstruction metrics → CSV                         |
| `misfit_inspect`     | `cli/inspect_entrypoint.py`     | Full-volume reconstruction → NIfTI                   |
| `misfit_encode`      | `cli/encode_entrypoint.py`      | Raw spatial features (N_crops, C, D', H', W')        |
| `misfit_embed`       | `cli/embed_entrypoint.py`       | Global embedding vector (C,) per volume              |
| `misfit_embed_train` | `cli/embed_train_entrypoint.py` | Train crop aggregator (classification / contrastive) |

All argument parsing lives in `cli/args.py`. The `ArgParser` subclass adds
`.arg()` and `.flag()` shorthands. `add_*_args` functions are shared between
individual entrypoints.

Every entrypoint module ends with `if __name__ == "__main__": <x>_entry()`, so
each is also runnable as `python -m misfit.cli.<x>_entrypoint` — the form
`torchrun -m misfit.cli.train_entrypoint` and shell-less container/Kubernetes
manifests use instead of `$(which misfit_train)`.

---

## Stage 1 — Indexing (`misfit_index`)

Scans NIfTI files in parallel and computes per-volume intensity statistics and
voxel spacing. Assigns each volume to `train`, `val`, or `test` splits. The
resulting Parquet index is the single input to all downstream commands.

```console
misfit_index --input  /data/paths.csv \
             --output /data/index.parquet
```

The `--input` file (CSV or Parquet) must have a `path` column with absolute
paths to `.nii` or `.nii.gz` files, and may have a `group` column (see
[Splits](#splits)).

### Key flags

| Flag                    | Default  | Description                                     |
| ----------------------- | -------- | ----------------------------------------------- |
| `--input FILE`          | required | CSV or Parquet with `path` (+ optional `group`) |
| `--output PARQUET`      | required | Destination Parquet index                       |
| `--num-workers-index N` | 32       | Parallel worker processes                       |

### Parquet index schema

| Column               | Description                                             |
| -------------------- | ------------------------------------------------------- |
| `volume_id`          | Unique identifier (filename without extension)          |
| `path`               | Absolute path to NIfTI file                             |
| `split`              | `train`, `val`, or `test`                               |
| `group`              | The `group` value from `--input`, or null               |
| `shape_d/h/w`        | Voxel dimensions                                        |
| `spacing_d/h/w`      | Voxel spacing in mm                                     |
| `affine`             | JSON-encoded 4×4 affine transform                       |
| `fg_x/y/z_start/end` | Foreground bounding box extents                         |
| `p1`                 | 1st-percentile foreground intensity (lower clip bound)  |
| `p99`                | 99th-percentile foreground intensity (upper clip bound) |
| `fg_mean`            | Foreground mean intensity after clipping                |
| `fg_std`             | Foreground standard deviation after clipping            |

### Splits

`misfit_index` writes a `<output_stem>_config.json` sidecar recording the split
sizes and seed:

```json
{ "val": 0.05, "test": 0.05, "max_val": 500, "max_test": 500, "seed": 42 }
```

By default val and test each get 5% of the corpus, capped at 500 volumes, and
train gets the rest (1,000 volumes → 900/50/50; 100,000 → 99,000/500/500). `val`
/ `test` may be a fraction in [0, 1) or a whole number of volumes (not capped);
`max_*` may be `null` for no cap. Edit the sidecar and re-run to change splits.
A sidecar from an older MISFIT (with a `train` key and 80/10/10 ratios) keeps
its original meaning. A bad config fails immediately, before the stats pass.

**Grouped splits:** add a `group` column (e.g. subject ID) so every volume of a
patient lands in the same split (no patient in both train and test). Groups are
shuffled and dealt out whole (val, then test, then train), so a split can
overshoot its target by less than one group; rows with an empty group are split
individually. The seeded split is reproducible run to run.

---

## Stage 2 — Pretraining (`misfit_train`)

Trains a masked autoencoder. At each step, 75% of the volume's
`mask_patch_size`³ cubes are masked and the model reconstructs them from visible
context.

**Step-based epochs:** an epoch is a fixed `--steps-per-epoch` optimizer steps
(default 1000 epochs × 250 steps), not a pass over the data, so validation,
checkpoints, logging, and the LR schedule don't scale with corpus size.
`EpochSampler` draws a seeded stream of permutations (`seed + epoch`, the same
on every rank; each rank takes a disjoint strided slice). The LR scheduler steps
once per optimizer step (warmup = `warmup_epochs × steps_per_epoch`).

```console
# Single GPU
misfit_train --index   /data/index.parquet \
             --results /runs/exp1

# 4 GPUs — -m works with no shell and nothing on PATH (container/Kubernetes
# manifests need this form; $(which misfit_train) needs a shell to evaluate it)
torchrun --nproc_per_node=4 -m misfit.cli.train_entrypoint \
    --index   /data/index.parquet \
    --results /runs/exp1

# Resume an interrupted run
misfit_train --index   /data/index.parquet \
             --results /runs/exp1 \
             --resume

# Preview the resolved config.json (FP32) without training — no GPU/queue needed
misfit_train --index   /data/index.parquet \
             --results /runs/exp1 \
             --init-only --no-amp
```

### Key flags

| Flag                              | Default                 | Description                                                              |
| --------------------------------- | ----------------------- | ------------------------------------------------------------------------ |
| `--model NAME`                    | `swinunetr-base`        | `swinunetr-small` / `-base` / `-large`, or `resenc`                      |
| `--patch-size X Y Z`              | `96 96 96`              | Crop size in voxels (at `--target-spacing` if set); divisible by 32      |
| `--target-spacing X Y Z`          | off (native)            | Resample every volume to this spacing; saved to `spatial_config`         |
| `--init-weights PT`               | —                       | Start from a full MAE checkpoint (MISFIT or nnssl, e.g. nnFoundationCNN) |
| `--mask-patch-size P`             | 16                      | Edge length of each masked 3D cube (voxels)                              |
| `--mask-ratio R`                  | 0.75                    | Fraction of cubes to mask                                                |
| `--normalization-policy NAME`     | `clip_and_z_score`      | `clip_and_z_score` or `z_score_only` (see Normalization)                 |
| `--loss NAME`                     | `normalized_masked_mse` | Loss function                                                            |
| `--epochs N`                      | 1000                    | Epochs of `--steps-per-epoch` steps each                                 |
| `--steps-per-epoch N`             | 250                     | Optimizer steps per epoch                                                |
| `--batch-size N`                  | 2                       | Per-GPU batch size                                                       |
| `--optimizer NAME`                | `adamw`                 | `adamw`, `adam`, `sgd`                                                   |
| `--learning-rate LR`              | 1e-4                    | Initial learning rate                                                    |
| `--weight-decay WD`               | 0.05                    | Weight decay                                                             |
| `--lr-scheduler NAME`             | `cosine`                | `cosine`, `polynomial`, `constant`                                       |
| `--warmup-epochs N`               | 50                      | Linear warmup, in epochs (LR updated every step)                         |
| `--gradient-accumulation-steps N` | 1                       | Accumulate gradients over N batches                                      |
| `--bucket-cap-mb MB`              | 200                     | DDP all-reduce bucket size (vs PyTorch default 25)                       |
| `--val-every N`                   | 1                       | Validate every N epochs (and after the last)                             |
| `--val-volumes N`                 | 128                     | Validate on the first N val volumes, with fixed masks                    |
| `--checkpoint-minutes M`          | 30                      | Also save a resumable checkpoint mid-epoch at most every M minutes       |
| `--num-cpu-workers N`             | 8                       | Data loader workers                                                      |
| `--seed N`                        | 42                      | Random seed                                                              |
| `--no-amp`                        | —                       | Request FP32 on a fresh run (ignored on `--resume`)                      |
| `--init-only`                     | —                       | Write config.json + skeleton dirs, exit without training                 |
| `--resume`                        | —                       | Resume from checkpoint (including mid-epoch)                             |
| `--overwrite`                     | —                       | Discard existing run and start fresh                                     |

Optimizer defaults are the same for every model (AdamW 1e-4, weight decay 0.05,
cosine, 50 warmup epochs). nnssl's ResEnc recipe uses SGD (Nesterov 0.99, 1e-2,
PolyLR), but don't present that as evidence SGD beats AdamW: the nnssl paper
takes it from nnU-Net without ablating optimizers.

**Validation:** runs every `--val-every` epochs on a fixed subset (the first
`--val-volumes` val rows) inside a forked, reseeded RNG, so every pass scores
the same masks on the same volumes (comparable curve) and the training RNG is
restored afterwards.

**Checkpoints:** store `(epoch, step_in_epoch, global_step)`. Rank 0 also saves
mid-epoch when `--checkpoint-minutes` have passed, and `--resume` continues the
same sample sequence mid-epoch. TensorBoard scalars are logged against
`global_step`: per step `loss/train_step`, `lr`, `grad_norm` (pre-clip); per
epoch `loss/train`, `loss/val`, `throughput/{steps,samples}_per_sec`.

`--resume` and `--overwrite` are mutually exclusive.`--resume` and `--overwrite`
are mutually exclusive. If neither is passed and `config.json` exists,
`misfit_train` refuses to run — `--init-only` follows the same rule (it will not
silently clobber a real run's config).

AMP is **BF16-only** — there is no `--amp-dtype` flag and no fp16/GradScaler
path (BF16 has float32's dynamic range). It is _requested_ on by default (pass
`--no-amp` to request FP32 instead), then resolved against the actual hardware
by `misfit.utils.hardware.resolve_amp` → `bf16_supported()`, which branches on
`get_accelerator_type()` (cuda / rocm / cpu, via `torch.version.hip`): CUDA
needs compute capability ≥ 8.0 (Ampere+); ROCm needs a `gcnArchName` in
`_ROCM_BF16_ACCELERATED_ARCHES` (CDNA + RDNA3+). Pre-Ampere NVIDIA GPUs (V100,
T4), older AMD GPUs (RDNA1/2 — gfx103x), and CPU fall back to **FP32** with a
warning. `misfit_train` resolves once (from `--no-amp`/the default, or from the
saved config on `--resume` — `--no-amp` is _not_ consulted there, the saved
value wins) and writes the effective value into `config.json`. To change AMP on
a **fresh** run, use `--no-amp` (combine with `--init-only` to see the resolved
config with no training at all); on a **resumed** run, hand-edit `"amp"` in
`config.json` and restart with `--resume` — `--no-amp` has no effect there.

**CPU / device path**: `misfit_train` runs on GPU (NCCL — RCCL on ROCm, same
backend name) or CPU (gloo); `MAETrainer.use_cuda` gates device placement,
backend, and cuDNN tuning. AMD ROCm reports as a GPU (`use_cuda` is `True` —
`torch.cuda` is a shim on ROCm builds), so it needs no special-casing. CPU
training works but is very slow — intended for tests and small debug runs, and
it warns once at startup. `misfit_evaluate` / `misfit_inspect` / `misfit_encode`
/ `misfit_embed` already accept `--device cpu`.

### Progress output

Training and validation each show a live Rich progress bar with loss and (for
training) learning rate refreshed every step — mirrors MIST's `TrainProgressBar`
/ `ValidationProgressBar` (`misfit.utils.progress_bar`):

```
Epoch 3/10 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5/5 • 0:00:00 • loss: 0.3000 • lr: 1.e-04
Validating ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3/3 • 0:00:00 • val_loss: 0.2800
```

With `--gradient-accumulation-steps > 1`, a fresh cross-rank-aggregated loss
only exists at a window end, so mid-window micro-steps advance the bar without
changing the displayed number — at the default of 1 this doesn't apply, every
step gets a fresh value. Once the loss converges past 4 decimal places, display
switches from `f"{loss:.4f}"` to scientific notation (`format_loss()`) instead
of printing a flat, uninformative `"0.0000"`: `loss: 3.100e-08` rather than
`loss: 0.0000`. Applies everywhere a loss gets printed (the bars, the epoch
summary, the `--resume` message, "Training complete") so they're always
consistent with each other.

### Output structure

```text
results/
    checkpoints/checkpoint.pt    Latest resumable checkpoint (end of epoch, or mid-epoch)
    models/best_model.pt         Lowest validation loss
    models/encoder_weights.pt    Encoder-only weights in the MIST model's key format
    logs/                        TensorBoard event files
    config.json                  Architecture + hyperparameters
```

### Immutable vs. mutable parameters (on resume)

**Immutable (hard error on change):** `model.architecture`, `model.patch_size`,
`model.mask_patch_size`, `spatial_config.target_spacing` (native ↔ resampled
counts as a change), and `data.normalization_policy`. Runs from before
step-based training (no `training.steps_per_epoch` in `config.json`) cannot be
resumed at all; start a new run.

**Mutable (warning only):** `epochs`, `steps_per_epoch`, `batch_size`,
`optimizer`, `learning_rate`, `weight_decay`, `lr_scheduler`, `warmup_epochs`,
`loss`, `gradient_accumulation_steps`, `bucket_cap_mb`, `val_every`,
`val_volumes`, `checkpoint_minutes`. `--init-weights` is ignored on `--resume`.

---

## Stage 3 — Evaluation (`misfit_evaluate`)

Computes masked reconstruction metrics **in the training loss's space** and
writes a per-volume CSV. Defaults to `val` split.

```console
misfit_evaluate --checkpoint /runs/exp1/models/best_model.pt \
                --index      /data/index.parquet \
                --config     /runs/exp1/config.json \
                --output-csv /runs/exp1/eval_results.csv
```

### Key flags

| Flag                     | Default                         | Description                                        |
| ------------------------ | ------------------------------- | -------------------------------------------------- |
| `--checkpoint PT`        | required                        | MISFIT checkpoint, or nnssl (e.g. nnFoundationCNN) |
| `--index PARQUET`        | required                        | Parquet index from `misfit_index`                  |
| `--config JSON`          | required                        | `config.json` from `misfit_train`                  |
| `--output-csv CSV`       | required                        | Destination for the results CSV                    |
| `--split SPLIT`          | `val`                           | Only rows with this split value; `""` = all rows   |
| `--metrics NAME [NAME…]` | config's `evaluation` section   | Overrides which metrics to compute                 |
| `--seed N`               | 42                              | Base RNG seed; the mask for tile _i_ = `seed + i`  |
| `--device DEVICE`        | `cuda` if available, else `cpu` | Torch device, e.g. `cuda:0`, `cpu`                 |

Key behaviour:

- **Metric space.** When the run used `normalized_masked_mse`, the target is
  normalized per `mask_patch_size` cube (exactly as the loss does), so
  `masked_mse` is directly comparable to the run's `best_val_loss`. Other losses
  keep the volume normalized by the run's normalization policy.
- **Same input path as training.** Each volume is normalized with the config's
  `data.normalization_policy`, cropped to its index `fg_*` bbox, and resampled
  to `spatial_config.target_spacing` if set, matching `MISFITDataset`.
- **Deterministic.** The mask for tile _i_ is drawn from `--seed + i` (default
  seed 42) — repeat runs are bit-identical.
- **Naive baseline.** Because a per-region-normalised error is ≈ 1.0 for any
  constant predictor, every metric also gets `<metric>_naive` (impute masked
  voxels with the visible-region mean) and `<metric>_skill` (`1 - model/naive`
  for lower-is-better, `model - naive` for higher-is-better; **positive = the
  model beats trivial**). A console `model | naive | verdict` summary is printed
  too. `_skill` is the robust "did pretraining work" number.

| Metric        | Direction        | Notes                                      |
| ------------- | ---------------- | ------------------------------------------ |
| `masked_mae`  | Lower is better  | default                                    |
| `masked_mse`  | Lower is better  | default; the `normalized_masked_mse` value |
| `masked_psnr` | Higher is better | default                                    |
| `ssim`        | Higher is better | **opt-in** via `--metrics ssim`            |

`--metrics NAME [NAME ...]` overrides the config's `evaluation` section. `ssim`
is not a default — it has no consistent space here (cube-boundary seams depress
it); use `misfit_inspect` for a viewer-space structural read.
`DEFAULT_METRICS = ("masked_mae", "masked_mse", "masked_psnr")` is what
`misfit_train` writes into `config.json`.

---

## Stage 3b — Inspection (`misfit_inspect`)

Reconstructs full volumes and saves NIfTI files for visual inspection in
ITK-SNAP or 3D Slicer.

```console
misfit_inspect --checkpoint /runs/exp1/models/best_model.pt \
               --index      /data/index.parquet \
               --config     /runs/exp1/config.json \
               --output-dir /runs/exp1/inspect
```

Output:

```text
output-dir/
    reconstructions/<volume_id>.nii.gz   Denormalized reconstruction (native grid)
    masks/<volume_id>.nii.gz             Binary: 1=masked, 0=visible
```

Overlay the mask in a viewer to see exactly which regions the model
reconstructed. Reconstructions are inverted with the normalization policy's
`IntensityScale` (`recon × scale + shift`); with `--target-spacing`,
reconstructions and masks are resampled back to the native grid. Judge quality
in the **masked** regions: an MAE is only trained there, so visible regions can
look wrong (in our smoke tests, nnFoundationCNN's visible voxels came out
inverted while masked voxels tracked the input).

---

## Stage 4 — Encoding (`misfit_encode`)

Extracts the full spatial bottleneck feature map `(N_crops, C, D', H', W')` for
every crop (`model.encoder(x)[-1]`). Shapes:

- `swinunetr-*`: `C = 8 × feature_size` (768 for base), `D' = patch / 32` (3 at
  96³).
- `resenc`: `C = 320`; a ≥128³ patch has 6 stages, so `D' = patch / 32` (4 at
  128³, 6 at 192³); 96³ has 5 stages, so `D' = 96 / 16 = 6`.

Every downstream command (encode, embed, evaluate, inspect) applies the run's
normalization policy and target spacing from `--config`, and accepts a MISFIT
checkpoint or an nnssl one. Use when you need spatially-rich features or want to
cache features for fast aggregator training.

```console
misfit_encode --encoder-checkpoint /runs/exp1/models/best_model.pt \
              --index               /data/index.parquet \
              --config              /runs/exp1/config.json \
              --output-dir          /data/encodings
```

Output `.npz` per volume:

- `feature_map`: `(N_crops, C, D', H', W')` — spatial bottleneck features
- `positions`: `(N_crops, 3)` — normalized 3D centre coordinates of each crop

---

## Stage 4b — Embedding (`misfit_embed`)

Encodes and aggregates crops into a single global `(C,)` embedding vector per
volume. With default `mean_pool`, no training is required — ready immediately
for zero-shot retrieval.

```console
misfit_embed --encoder-checkpoint /runs/exp1/models/best_model.pt \
             --index               /data/index.parquet \
             --config              /runs/exp1/config.json \
             --output-dir          /data/embeddings

# With trained attention aggregator
misfit_embed --encoder-checkpoint     /runs/exp1/models/best_model.pt \
             --index                   /data/index.parquet \
             --config                  /runs/exp1/config.json \
             --output-dir              /data/embeddings \
             --aggregator              attention_pool \
             --aggregator-checkpoint   /runs/agg/aggregator.pt
```

Output `.npz` per volume: `embedding: (C,)`.

---

## Stage 5 — Aggregator Training (`misfit_embed_train`)

Fine-tunes a lightweight aggregator on cached features from `misfit_encode`. The
encoder weights are **frozen** — only the aggregator trains.

```console
misfit_embed_train --input      /data/train_manifest.csv \
                   --output-dir /runs/agg \
                   --embed-dim  768   # 320 for resenc
```

### Input CSV format

| Column          | Description                                     |
| --------------- | ----------------------------------------------- |
| `volume_id`     | Volume identifier                               |
| `split`         | Only `split='train'` rows are used for training |
| `features_path` | Absolute path to `.npz` from `misfit_encode`    |
| `label`         | String label                                    |

### Key flags

| Flag                     | Default          | Description                                                    |
| ------------------------ | ---------------- | -------------------------------------------------------------- |
| `--aggregator NAME`      | `attention_pool` | `mean_pool` or `attention_pool`                                |
| `--objective NAME`       | `classification` | `classification` (cross-entropy) or `contrastive` (SupCon K=2) |
| `--embed-dim C`          | required         | Encoder bottleneck channels (768 swinunetr-base, 320 resenc)   |
| `--no-position-encoding` | off              | Disable 3D position encoding in `attention_pool`               |
| `--epochs N`             | 50               | Training epochs                                                |
| `--batch-size N`         | 32               | Must be even for contrastive objective                         |
| `--learning-rate LR`     | 1e-3             | Initial learning rate                                          |

---

## Model Architecture

### SwinMAE (`models/swinunetr/misfit_swinunetr_mae.py`)

`SwinMAE(MISFITModel)` wraps `SwinUNETR.swinViT` as `self.encoder` (UNet decoder
discarded). Components:

- **Encoder**: `SwinUNETR.swinViT` — 5 hidden states at spatial resolutions D/2
  … D/32
- **MAEDecoder**: Lightweight `ConvTranspose3d` stack — 5 upsampling steps
  restore original resolution. Channel schedule halves at each step, floored
  at 32.
- **mask_token**: Learnable scalar parameter broadcast over spatial dims.

**Masking strategy**: SimMIM-style — applied at the image level before encoding
(required because Swin windowed attention breaks with irregular token counts).
Masking generates a grid of non-overlapping `mask_patch_size^3` cubes and
randomly selects `mask_ratio` fraction.

**Spatial constraints**:

- All `img_size` dimensions must be divisible by 32 (SwinUNETR-V2 total
  downsampling)
- All `img_size` dimensions must be divisible by `mask_patch_size`

### ResEncMAE (`models/resenc/misfit_resenc_mae.py`, `--model resenc`)

MIST's full `ResEncUNet` used as the MAE network — the nnssl recipe: dense
masking on the image, then encoder + UNet decoder **with skip connections**
(masked regions are blank at every level, so skips can't leak the target).
`out_channels = in_channels` turns the decoder's full-resolution segmentation
layer into the reconstruction head (`self.head`); the lower-resolution
deep-supervision layers are deleted (they'd get no gradient, which DDP rejects).
`self.encoder` is MIST's `ResidualEncoder`, so `get_encoder_state_dict()`
returns MIST's `encoder.*` keys unchanged (no remap).

Topology comes from MIST's planner (`get_unet_params(img_size, spacing)`): the
run's `--target-spacing` if set, otherwise an isotropic 1 mm planning spacing
(only axis ratios matter, so this just means "isotropic topology"). Isotropic:
96³ → 5 stages (~63M params); ≥128³ → 6 stages (~102M, ~90M encoder). Same
`img_size` constraints as SwinMAE (÷32, ÷`mask_patch_size`). Patch ≥128³ is
recommended so the encoder has all six stages (nnssl's default MAE trainer uses
160³; nnFoundationCNN used 192³).

### Model variants

| Variant | `--model`         | Size              | Encoder params | Full MAE params |
| ------- | ----------------- | ----------------- | -------------- | --------------- |
| Small   | `swinunetr-small` | feature_size 24   | ~4.8M          | ~7.0M           |
| Base    | `swinunetr-base`  | feature_size 48   | ~18.6M         | ~27.1M          |
| Large   | `swinunetr-large` | feature_size 96   | ~73.9M         | ~107.7M         |
| ResEnc  | `resenc`          | 6 stages at ≥128³ | ~90M           | ~102M           |

"Encoder params" is what transfers to MIST (`encoder_weights.pt`); "Full MAE
params" adds the decoder and mask token used only during pretraining.

**Spacing:** without `--target-spacing`, crops are taken at native resolution
and `spacing_d/h/w` stay index metadata. With it, every path from NIfTI to model
input resamples with the same helpers (`misfit.utils.resampling`, trilinear;
shape `round(n × spacing / target)` per axis) — the dataset resamples after the
foreground crop, so only the foreground is interpolated. Index spacing columns
are in (x, y, z) order, like MIST's `--patch-size X Y Z`. The `SpacingEmbedding`
decoder conditioning was removed after 0.1.3-alpha; older SwinMAE checkpoints
still load (their `spacing_embed.*` keys are ignored).

**Model contract (every architecture):** `forward(x, mask_ratio=None)` returns
`{"reconstruction", "mask"}` in train and eval mode, `model.encoder(x)[-1]` is
the bottleneck, and every parameter must receive a gradient (DDP runs with
`find_unused_parameters=False`).

---

## config.json Structure

`config.json` is written to `--results` at training start and is the single
source of truth for model architecture. All downstream commands require
`--config`.

```json
{
  "misfit_version": "0.2.0-alpha",
  "data": {
    "index": "/data/index.parquet",
    "normalization_policy": "clip_and_z_score"
  },
  "model": {
    "architecture": "resenc",
    "patch_size": [128, 128, 128],
    "mask_patch_size": 16,
    "mask_ratio": 0.75
  },
  "spatial_config": {
    "patch_size": [128, 128, 128],
    "target_spacing": null
  },
  "training": {
    "epochs": 1000,
    "steps_per_epoch": 250,
    "batch_size": 2,
    "optimizer": "adamw",
    "learning_rate": 0.0001,
    "weight_decay": 0.05,
    "lr_scheduler": "cosine",
    "warmup_epochs": 50,
    "loss": "normalized_masked_mse",
    "amp": true,
    "seed": 42,
    "gradient_accumulation_steps": 1,
    "bucket_cap_mb": 200,
    "val_every": 1,
    "val_volumes": 128,
    "checkpoint_minutes": 30.0,
    "init_weights": null
  },
  "evaluation": {
    "masked_mae": {},
    "masked_mse": {},
    "masked_psnr": {}
  }
}
```

`spatial_config` duplicates `model.patch_size` on purpose: it is MIST's own
section layout, so `mist_train --pretrained-config` reads a MISFIT config
directly. A null `target_spacing` means native-resolution pretraining (MIST
2.4.0+ warns that spacing can't be checked instead of failing). A config without
`data.normalization_policy` (pre-0.2.0) means `clip_and_z_score`.

To disable AMP on a fresh run, pass `--no-amp` (or `--init-only --no-amp` to
just preview the config first). On a run already resuming, hand-edit
`"amp": false` in `config.json` and restart with `--resume` — `--no-amp` has no
effect there.

---

## Normalization Pipeline

Normalization is a registered **policy** (`misfit/normalization/`), chosen with
`--normalization-policy`, saved as `config["data"]["normalization_policy"]`,
fixed for a run (resume rejects a change), and applied by every input path
(`MISFITDataset`, `inference_utils.load_and_normalize` for encode/embed/inspect,
and the evaluator). Each policy returns the normalized volume plus an
`IntensityScale(shift, scale)` that `misfit_inspect` uses to denormalize.

| Policy                       | What it does                                                                                                | Use for                                |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| `clip_and_z_score` (default) | Clip to the foreground's `[p1, p99]` (index columns), z-score with foreground mean/std                      | Mixed CT + MRI corpora                 |
| `z_score_only`               | Z-score with the mean/std of the nonzero bounding box; no clipping, no foreground mask (stats at load time) | nnssl checkpoints such as nnFoundation |

`z_score_only` reproduces nnssl's preprocessing (crop to nonzero with holes
filled, then z-score). New policies subclass `NormalizationPolicy` and use
`@register_normalization_policy`.

**Load-failure handling**: every path in the index was already confirmed
loadable by `misfit_index` (bad files are dropped there, never indexed), so a
load failure inside `MISFITDataset.__getitem__` at train time means something
changed _since_ indexing — deleted, moved, a transient filesystem error. An
isolated failure warns (`UserWarning`:
`"<path>: failed to load (...); substituting a zero volume for this sample (N/3 consecutive failures ...)"`)
and substitutes a zero-filled tensor for that one sample so the run keeps going.
`max_load_failures` (default 3) _consecutive_ failures — no successful load in
between — raise `RuntimeError` and stop the run instead: that's a systemic
problem (a mount gone away, permissions revoked), not a one-off, and continuing
would silently train on an escalating fraction of zero-filled "volumes". The
counter resets to 0 on every successful load and lives on the `Dataset`
instance, which `persistent_workers=True` keeps alive for the whole run (not
just one epoch). If a user reports this warning: 1-2 isolated occurrences are
safe to ignore; if the run stopped with the `RuntimeError`, the filesystem/paths
need investigating before restarting, not a config change.

---

## Loss Functions

| Loss                    | `--loss`    | Notes                                                                               |
| ----------------------- | ----------- | ----------------------------------------------------------------------------------- |
| `normalized_masked_mse` | **Default** | Per-patch variance normalization before MSE. Recommended for CT+MRI mixed training. |
| `masked_mse`            | —           | Standard MSE on masked voxels only.                                                 |
| `masked_l1`             | —           | Mean absolute error on masked voxels. More robust to intensity outliers.            |

`normalized_masked_mse` normalizes each 3D patch's target to zero mean/unit
variance before computing MSE. This prevents high-contrast regions (CT bone)
from dominating the gradient signal and equalizes loss scale across modalities.

---

## Optimizers and LR Schedulers

**Optimizers**: `adamw` (default), `adam`, `sgd`

**Schedulers**: `cosine` (default), `polynomial`, `constant`

All schedulers support linear warmup via `--warmup-epochs` (default 50 of 1000
epochs). Because epochs are step-based, the scheduler steps every optimizer
step: `total_steps = epochs × steps_per_epoch`,
`warmup_steps = warmup_epochs × steps_per_epoch`.

**AMP notes**:

- AMP is BF16-only (`hardware.autocast_context` →
  `torch.autocast("cuda", dtype=torch.bfloat16)`; device type is `"cuda"` for
  both CUDA and ROCm). There is no `--amp-dtype` flag and no fp16 path.
- Requested on by default, then resolved by `misfit.utils.hardware.resolve_amp`
  → `bf16_supported()`. On **CUDA**: compute capability ≥ 8.0 (Ampere+: A100,
  H100, RTX 30xx+) — checked directly, not via `torch.cuda.is_bf16_supported()`
  (which lies on V100/T4). On **AMD ROCm**: the GPU's `gcnArchName` must be in
  `_ROCM_BF16_ACCELERATED_ARCHES` — an allow-list of CDNA (MFMA: MI100/200/300)
  and RDNA3+ (WMMA: RX 7000+) arches. The capability check is useless on ROCm
  (reports ≥ (9,0) for every AMD GPU) and `is_bf16_supported()` lies there too
  (True on RDNA1/2, which runs BF16 on shader ALUs with no speedup — measurably
  slower than FP32). Unrecognized arch → unsupported. Pre-Ampere NVIDIA GPUs,
  RDNA1/2, and CPU fall back to **FP32** with a `UserWarning`.
- BF16 has float32's dynamic range, so no GradScaler is used and the optimizer
  epsilon is the standard `1e-8` (`tc.NO_AMP_EPS`).
- `misfit_train` resolves once and persists the effective value to
  `config.json`. `misfit_evaluate` and `misfit_inspect` re-resolve the config
  value against their own hardware (they may run on a different GPU or CPU).
- `--no-amp` requests FP32 directly on a fresh run — set in `__init__`
  (`self.amp = not getattr(self.args, "no_amp", False)`), so it's the starting
  point `resolve_amp` sees, not a post-hoc override. Not consulted on `--resume`
  (saved `config.json` wins there); hand-edit the file instead.
- `--init-only` writes `config.json` + the results skeleton and returns before
  model/data/DDP setup — pair it with `--no-amp` to get an AMP-off config with
  no run/kill/edit/`--resume` dance.

---

## Distributed Training

`MAETrainer` reads `RANK`, `LOCAL_RANK`, `WORLD_SIZE` from torchrun environment
variables. The same class runs on 1 GPU or N×M GPUs (NCCL backend — RCCL on AMD
ROCm, same `"nccl"` name), and on 1 or N CPU processes (gloo backend) for
testing. `MAETrainer.use_cuda` (`torch.cuda.is_available()`, which is `True` on
ROCm via the `torch.cuda` shim) selects the device, the process-group backend,
whether `torch.cuda.set_device` / cuDNN tuning run, and whether DDP gets
`device_ids`.

`_setup_distributed` calls `torch.cuda.set_device(local_rank)` **before**
`dist.init_process_group`, and passes
`device_id=torch.device("cuda", local_rank)` explicitly. Order matters: a NCCL
group created while every rank is still on the default `cuda:0` binds DDP's
construction-time param-shape allgather to device 0 on _all_ ranks, so a
multi-GPU job hangs on the first collective (`rank 0 has inconsistent 0 params`,
10-min watchdog timeout) — fixed, but if a multi-GPU job still hangs at startup
after that, it's NCCL peer-to-peer transport (e.g. GPUs spanning two CPU sockets
on a shared node), not MISFIT; `NCCL_P2P_DISABLE=1` / `NCCL_P2P_LEVEL=NVL` are
the workarounds.

```console
# Single node, 4 GPUs — -m needs no shell/PATH (see Kubernetes below)
torchrun --nproc_per_node=4 \
    -m misfit.cli.train_entrypoint \
        --index      /data/index.parquet \
        --results    /runs/exp1 \
        --batch-size 2

# Multi-node (SLURM example)
torchrun --nnodes=2 \
         --nproc_per_node=4 \
         --node_rank=$SLURM_NODEID \
         --master_addr=$MASTER_ADDR \
         --master_port=29500 \
    -m misfit.cli.train_entrypoint \
        --index   /data/index.parquet \
        --results /runs/exp1
```

`--batch-size` is per-GPU. Effective global batch =
`batch_size × world_size × gradient_accumulation_steps`. When scaling up the GPU
count, consider scaling `--learning-rate` with the global batch (a common
heuristic, not something MISFIT has benchmarked).

**Kubernetes**: a pod's `command`/`args` go straight to `execve` — no shell — so
`$(which misfit_train)` reaches Python as the literal string `$(which`. Use
`-m misfit.cli.train_entrypoint` (any of the 7 CLIs works this way — each module
ends with `if __name__ == "__main__"`), or wrap the command in `["bash", "-lc"]`
if you want `$(...)`/env expansion.

---

## Embedding Aggregators

| Aggregator       | Training required          | Description                                                 |
| ---------------- | -------------------------- | ----------------------------------------------------------- |
| `mean_pool`      | No (zero-shot)             | Unweighted mean of all crop feature vectors                 |
| `attention_pool` | Yes (`misfit_embed_train`) | Multi-head cross-attention weighted by 3D spatial positions |

Use `mean_pool` for quick retrieval or UMAP visualization immediately after
pretraining. Use `attention_pool` when you have labeled data and want
task-specific pooling.

---

## MIST Integration

MISFIT encoders transfer directly to MIST (`mist-medical>=2.4.0`) for supervised
3D segmentation. `encoder_weights.pt` is saved whenever validation loss improves
(alongside `best_model.pt`), in the matching MIST model's key format:

| MISFIT `--model` | MIST `--model`     | Keys                                                 |
| ---------------- | ------------------ | ---------------------------------------------------- |
| `swinunetr-*`    | same `swinunetr-*` | `encoder.<name>` → `model.swinViT.<name>` (remapped) |
| `resenc`         | `resenc`           | `encoder.<name>` unchanged                           |

```console
mist_train \
    --numpy              /path/to/preprocessed/data \
    --results            /path/to/mist/results \
    --model              resenc \
    --pretrained-weights /runs/pretrain/models/encoder_weights.pt \
    --pretrained-config  /runs/pretrain/config.json
```

`--pretrained-config` (recommended) points MIST at the MISFIT `config.json` so
it validates architecture compatibility (and, for adaptive architectures, patch
size and spacing) before loading; omitting it only warns. **Use the same
architecture name in both.** For `resenc`, a patch size or spacing mismatch with
the MIST run only warns: every stage the two networks share transfers exactly,
extra MIST stages train from scratch, and anisotropic kernels that differ keep
random init. MIST prints a transfer summary and errors if nothing transfers.

### Channel mismatch handling

MISFIT trains single-channel; MIST tasks may be multi-channel:

| `--input-channel-strategy` | Behaviour                                          |
| -------------------------- | -------------------------------------------------- |
| `average` (default)        | Average source channels, then tile to target count |
| `first`                    | Use first source channel only, then tile           |
| `skip`                     | Keep the first convolution at random init          |

### Does pretraining help?

Don't promise gains. Whether a pretrained encoder helps depends on the task,
data, and training budget; MISFIT's own neuro results are early tests. The way
to know is to measure: train the same MIST configuration with and without
`--pretrained-weights` and compare with `mist_rank`. A warmup in MIST protects
pretrained features from the randomly initialized decoder's large early
gradients; learning rate and warmup are settings to tune, with no published
recommended values.

---

## Starting from nnFoundation

[nnFoundationCNN](https://huggingface.co/MIC-DKFZ/nnFoundationCNN)
([Harsy et al., 2026](https://arxiv.org/abs/2609.26924)) is an openly released
ResEnc pretrained with nnssl's MAE objective (16³ blocks, 75% masked,
zero-filled, plain masked MSE) on 2.1 million CT, MRI, and PET volumes in
single-channel 192³ patches. nnssl used its `noresample` config: **no
resampling** (native resolution; median spacing 1 mm), crop to nonzero, z-score
without mask or clipping. These facts are verified from the checkpoint's
`nnssl_adaptation_plan` and nnssl's source; don't say it was pretrained "at 1 mm
isotropic".

MISFIT's `resenc` with a ≥128³ patch and an isotropic topology is the same
network (decoder and head included; verified to match nnU-Net's
`ResidualEncoderUNet` output exactly). Loading is strict
(`misfit/models/pretrained.py`): every parameter must match except `mask_token`
(nnssl masks with zeros, the token's init).
`ResEncMAE.adapt_pretrained_state_dict` renames the last `decoder.seg_layers.N`
to `head` and drops the `decoder.encoder.*` aliases and other seg layers.

Closest match to nnssl — use **no `--target-spacing`** (native resolution, like
`noresample`; MISFIT plans the same isotropic topology), `z_score_only`, and
`masked_mse`:

```console
# As a ready-made encoder: write a matching config, then use it directly
misfit_train --index /data/index.parquet --results /runs/nnfoundation \
             --model resenc --patch-size 192 192 192 \
             --normalization-policy z_score_only --loss masked_mse --init-only
misfit_embed --encoder-checkpoint /models/checkpoint_final.pth \
             --index /data/index.parquet --config /runs/nnfoundation/config.json \
             --output-dir /data/embeddings_nnfoundation

# As a starting point: continue MAE pretraining on your own data
misfit_train --index /data/index.parquet --results /runs/nnfoundation_ft \
             --model resenc --patch-size 192 192 192 \
             --normalization-policy z_score_only --loss masked_mse \
             --init-weights /models/checkpoint_final.pth
```

`--target-spacing 1 1 1` also builds the same network but resamples (nnssl
didn't). An anisotropic target spacing or a patch below 128³ fails the strict
load with an error listing the mismatched weights. With the default
`normalized_masked_mse`, `misfit_evaluate` scores nnFoundation against
per-cube-standardized targets it never learned (in a phantom smoke test it
looked worse than the naive baseline); with `masked_mse`, its masked-MSE skill
was 0.84 (`z_score_only`) vs 0.80 (`clip_and_z_score`). nnFoundationViT
(PrimusX) isn't supported in MISFIT yet. Weights are CC BY-SA 4.0.

---

## Registry Pattern

Models, losses, metrics, aggregators, and objectives use a decorator-based
registry. Imports trigger registrations.

```python
# New loss
from misfit.loss_functions.base import ReconstructionLoss
from misfit.loss_functions.loss_registry import register_loss

@register_loss("my_loss")
class MyLoss(ReconstructionLoss):
    def forward(self, reconstruction, target, mask):
        ...
```

Place under `loss_functions/reconstruction/`, import in
`loss_functions/__init__.py`.

**New model**: build the network from `mist.models` (don't reimplement it),
subclass `MISFITModel`, implement `get_encoder_state_dict()` in the MIST model's
key format, apply `@register_model(name="<MIST model name>")`, place under
`models/<name>/`, import in `models/__init__.py`. Reuse `models/masking.py`,
follow the model contract above, and optionally implement
`adapt_pretrained_state_dict` to load foreign checkpoints.

**New normalization policy**: subclass `NormalizationPolicy`
(`normalize(volume, row) -> (normalized, IntensityScale)`), apply
`@register_normalization_policy`, place in `normalization/policies.py` (or a
module imported from `normalization/__init__.py`).

**New metric**: subclass `ReconstructionMetric`, apply `@register_metric` as a
class decorator (it instantiates and registers the class). Unlike the other
registries, metrics don't follow a one-file-per-class layout — the concrete
classes live directly in `metrics/metrics_registry.py`, and
`metrics/__init__.py` is empty. Add the class there (or in a new file imported
from `metrics_registry.py`).

**New aggregator**: subclass `AbstractAggregator`,
`@register_aggregator(name="...")`, place under `embedding/aggregators/`, import
in `embedding/aggregators/__init__.py`.

---

## Module Map

```
misfit/
  cli/                  Entry points + shared ArgParser / add_*_args
  preprocessing/        NIfTI indexer → Parquet (parallel, ProcessPoolExecutor)
  data_loading/         MISFITDataset + DataLoader (normalization policy, resampling);
                        EpochSampler (fixed samples per epoch for step-based training)
  models/               MISFITModel base class; SwinMAE (MIST SwinUNETR-V2 + MAE head);
                        ResEncMAE (MIST ResEnc UNet as the MAE network); masking;
                        pretrained (load MISFIT / nnssl checkpoints)
  normalization/        NormalizationPolicy registry: clip_and_z_score, z_score_only
  training/             MAETrainer; optimizer/LR-scheduler registries; training_utils
  loss_functions/       ReconstructionLoss base; masked_mse, masked_l1, normalized_mse
  metrics/              ReconstructionMetric base; masked_mae/mse/psnr (DEFAULT_METRICS) + ssim (opt-in)
  evaluation/           ReconstructionEvaluator; tiled full-volume inference + CSV output
  inference/            InferenceRunners; tiled reconstruct pipeline (pad→tile→stitch)
  embedding/            Embedder; EmbedTrainer; aggregators (mean_pool, attention_pool);
                        objectives (classification, contrastive)
  utils/                console (Rich), io (read/write JSON),
                        progress_bar (get_progress_bar; TrainProgressBar /
                        ValidationProgressBar — mirror mist.utils.progress_bar;
                        format_loss),
                        hardware (get_accelerator_type / bf16_supported /
                        resolve_amp / autocast_context),
                        normalization (normalize_patchwise / denormalize_patchwise),
                        resampling (--target-spacing helpers)
```

---

## Running Tests

Always use the `mist` mamba environment — MISFIT and MIST are both installed
editably there:

```bash
mamba run -n mist pytest
```

Never use plain `pytest` or `python -m pytest`.

---

## Common Workflows

### Minimal pretraining run

```console
misfit_index --input paths.csv --output index.parquet
misfit_train --index index.parquet --results /runs/exp1

# ResEnc with all six stages
misfit_train --index index.parquet --results /runs/resenc \
             --model resenc --patch-size 160 160 160
```

### Evaluate and inspect results

```console
misfit_evaluate --checkpoint /runs/exp1/models/best_model.pt \
                --index      index.parquet \
                --config     /runs/exp1/config.json \
                --output-csv /runs/exp1/eval.csv

misfit_inspect  --checkpoint /runs/exp1/models/best_model.pt \
                --index      index.parquet \
                --config     /runs/exp1/config.json \
                --output-dir /runs/exp1/inspect
```

### Zero-shot embeddings (no aggregator training)

```console
misfit_embed --encoder-checkpoint /runs/exp1/models/best_model.pt \
             --index               index.parquet \
             --config              /runs/exp1/config.json \
             --output-dir          /data/embeddings
```

### Task-specific embeddings with trained aggregator

```console
# Step 1: extract and cache encoder features
misfit_encode --encoder-checkpoint /runs/exp1/models/best_model.pt \
              --index               index.parquet \
              --config              /runs/exp1/config.json \
              --output-dir          /data/encodings

# Step 2: train aggregator on labeled subset
misfit_embed_train --input      /data/manifest.csv \
                   --output-dir /runs/agg \
                   --embed-dim  768

# Step 3: extract task-specific embeddings
misfit_embed --encoder-checkpoint     /runs/exp1/models/best_model.pt \
             --index                   index.parquet \
             --config                  /runs/exp1/config.json \
             --output-dir              /data/embeddings \
             --aggregator              attention_pool \
             --aggregator-checkpoint   /runs/agg/aggregator.pt
```

### Transfer to MIST segmentation

```console
mist_train \
    --numpy              /data/mist_preprocessed \
    --results            /runs/mist_finetune \
    --model              swinunetr-base \
    --pretrained-weights /runs/exp1/models/encoder_weights.pt \
    --pretrained-config  /runs/exp1/config.json
```

---

## Planned Features (Not Yet Implemented)

- `misfit_anomaly` — reconstruction error as anomaly score; zero-label anomaly
  detection
- `misfit_search` — FAISS-backed nearest-neighbor retrieval over embedding
  corpus
- `misfit_visualize` — UMAP of embedding space, attention weight heatmaps
