# VANET Trajectory Prediction with Transformers

Unsupervised pre-training of a Transformer encoder on VANET vehicle-trajectory data
via a **masked-value imputation** task (Zerveas et al., 2021).  The model is trained to
reconstruct randomly masked time-series features, learning rich temporal representations
of vehicle motion that can later be fine-tuned for downstream tasks (e.g., collision
prediction, lane-change detection).

---

## Table of Contents

1. [Repository Layout](#1-repository-layout)
2. [Data Pipeline](#2-data-pipeline)
3. [Model Architecture](#3-model-architecture)
4. [Training Loop](#4-training-loop)
5. [Loss Function](#5-loss-function)
6. [Normalisation](#6-normalisation)
7. [Configuration & CLI](#7-configuration--cli)
8. [Hyperparameter Tuning](#8-hyperparameter-tuning)
9. [Output Artefacts](#9-output-artefacts)
10. [Extension Guide — What to Edit for Each Change](#10-extension-guide)
11. [Quick-Start Commands](#11-quick-start-commands)

---

## 1 — Repository Layout

```
VANET_Trajectory_Pred_With_Transformers/
│
├── main.py                          # Entry-point: parse args → setup → run()
├── hyperparamer_tuning.py           # Ray Tune sweep entry-point
├── requirements.txt
│
├── resources/
│   └── VANET_data/raw/              # One CSV per vehicle trip (749 files)
│       └── data_car_<id>_t<time>.csv
│
└── src/
    ├── options.py                   # All CLI arguments & defaults
    ├── datasets/
    │   ├── data.py                  # Normalizer, BaseData, SINDData (CSV → chunks)
    │   ├── datasplit.py             # ShuffleSplit train/val split
    │   ├── masked_datasets.py       # ImputationDataset, collate_unsuperv, noise_mask
    │   └── plot.py                  # Visualisation utilities
    ├── transformer_model/
    │   ├── encoder.py               # TSTransformerEncoder + positional encodings
    │   └── model.py                 # create_model, train, validate, UnsupervisedAttentionModel
    └── utils/
        ├── config_setup.py          # setup(): dir creation, JSON config dump
        ├── load_data.py             # load_data(): normalise + build DataLoaders
        ├── model_helpers.py         # MaskedMSELoss, get_loss_module, optimizers, EarlyStopping
        ├── hyperparemer_tuning_config.py  # HyperOpt search space definition
        ├── print_helpers.py         # Printer, readable_time, count_parameters
        ├── record_data.py           # Append experiment results to an Excel file
        └── poly_process.py          # Polynomial feature utilities (unused in main pipeline)
```

---

## 2 — Data Pipeline

### 2.1 CSV Format

Each file (`data_car_<id>_t<start_time>.csv`) contains the trajectory of **one vehicle**
during one simulation run.  Columns:

| Column | Type | Range (approx.) | Notes |
|---|---|---|---|
| `Time` | float | 17 000 – 22 000 s | Sorting key only, not a feature |
| `X` | float | ~4 400 m | Absolute position east |
| `Y` | float | ~5 000 m | Absolute position north |
| `Speed` | float | 0 – 40 m/s | |
| `Acceleration` | float | −5 – 5 m/s² | |
| `Heading` | float | 0 – 360° | |
| `AngularVelocity` | float | | |
| `LaneID` | **string** | e.g. `"153638_1"` | Factorized → int before use |
| `LaneDist` | float | 0 – 4 m | Lateral offset within lane |
| `Neigh{1,2,3}_Rx/Ry` | float | | Relative position to 3 neighbours |
| `Neigh{1,2,3}_RSpeed` | float | | Relative speed |
| `Neigh{1,2,3}_RHeading` | float | | Relative heading |
| `AvgDistToSender` | float | | V2X communication metric |
| `AvgMsgDelay` | float | | V2X communication metric |
| `PacketLossRate` | float | 0 – 1 | V2X communication metric |

**Total: 23 features** fed to the model.

### 2.2 `SINDData` — `src/datasets/data.py`

```
SINDData.__init__()         sets feature_names (23 cols), config, max_seq_len
SINDData.load_data()        walks data_dir, calls load_single() per file
  ↓ load_single()           read_data() → sort_clean_data() → fillna(0)
  ↓ sort_clean_data()       factorize LaneID, sort by (track_id, Time),
                            make track_id globally unique (file_id + "_" + id),
                            drop stationary tracks (max Speed == 0)
  ↓ chunking loop           split each track into windows of max_seq_len rows
                            → stored in self.all_chunks (list of np.float32 arrays)
self.all_IDs = [0, 1, …, N-1]   integer index into all_chunks
```

**Key design note**: the data class stores raw chunks, not a DataFrame.
`feature_df` is `None`; downstream code uses `my_data.all_chunks[i]` directly.

### 2.3 `load_data()` — `src/utils/load_data.py`

```
load_data(config, logger)
  1. Instantiate SINDData, call load_data()         → all_chunks filled
  2. split_dataset()                                 → train_indices, val_indices
  3. Global normalisation (if data_normalization ≠ "none")
       stack training chunks only → compute mean/std
       normalise ALL chunks in-place                 → prevents data leakage
  4. ImputationDataset(train_data, train_indices)    → train_dataset
  5. DataLoader(train_dataset, collate_fn=collate_unsuperv)
  6. Same for val_dataset / val_loader
  return train_loader, val_loader, my_data
```

### 2.4 `ImputationDataset` — `src/datasets/masked_datasets.py`

```python
__getitem__(ind):
    X = my_data.all_chunks[IDs[ind]]   # (seq_len, 23)  float32
    mask = noise_mask(X, …)            # (seq_len, 23)  bool  — 0 = masked
    return torch.from_numpy(X), torch.from_numpy(mask), IDs[ind]
```

`update()` — called every `harden_step` epochs to progressively increase `masking_ratio`
and `mean_mask_length` (curriculum learning).

### 2.5 `collate_unsuperv()` — `src/datasets/masked_datasets.py`

Pads variable-length sequences in a batch to `max_seq_len`:

```
X            (B, max_len, 23)   masked input  (0s at masked positions)
targets      (B, max_len, 23)   original unmasked values
target_masks (B, max_len, 23)   bool: True = predict this position
padding_masks(B, max_len)       bool: True = real data (not padding)
```

### 2.6 Noise Mask Generation

Two modes:
- **`bernoulli`** — each position is masked independently with probability `masking_ratio`.
- **`geometric`** — Markov-chain-based; produces contiguous masked segments of average
  length `mean_mask_length`.

Two spatial modes:
- **`separate`** — each of the 23 features is masked independently.
- **`concurrent`** — all 23 features are masked/unmasked together at each time step.

---

## 3 — Model Architecture

### `TSTransformerEncoder` — `src/transformer_model/encoder.py`

```
Input X: (B, seq_len, 23)
   │
   ▼
project_inp: Linear(23 → embedding_dim)    scaled by √embedding_dim
   │
   ▼
pos_enc: positional encoding added
   │   ├─ FixedPositionalEncoding   — sine/cosine (Vaswani et al.)
   │   └─ LearnablePositionalEncoding — nn.Parameter, init uniform(−0.02, 0.02)
   │
   ▼
TransformerEncoder (num_layers blocks)
   each block = TransformerBatchNormEncoderLayer  (default)
              or TransformerEncoderLayer           (if norm="LayerNorm")
   │
   ▼  (B, seq_len, embedding_dim)
activation (relu or gelu)
   │
dropout1
   │
output_layer: Linear(embedding_dim → 23)
   │
   ▼
output: (B, seq_len, 23)   — reconstructed feature values
also returns embeddings (post-activation) and embeddings_original (pre-activation)
```

### `TransformerBatchNormEncoderLayer` (default norm)

Standard Transformer block with **BatchNorm1d** replacing LayerNorm:
```
Self-Attention  →  Dropout + Residual  →  BatchNorm1d
FFN (Linear-relu-Dropout-Linear)  →  Dropout + Residual  →  BatchNorm1d
```
BatchNorm requires the sequence dim to be permuted to position 2 temporarily.

### Key hyperparameters

| Arg | Default | Effect |
|---|---|---|
| `embedding_dim` | 128 | Width of the transformer's internal representation |
| `num_heads` | 8 | Attention heads (must divide `embedding_dim`) |
| `num_layers` | 3 | Number of stacked encoder blocks |
| `hidden_dim` | 512 | FFN width inside each block |
| `dropout` | 0.1 | Applied to attention, FFN, positional encoding, and output |
| `pos_encoding` | `"fixed"` | `"learnable"` recommended for VANET (set in tuning script) |
| `activation` | `"relu"` | `"gelu"` is an alternative |
| `normalization_layer` | `"BatchNorm"` | `"LayerNorm"` for smaller batch sizes |

---

## 4 — Training Loop

### `UnsupervisedAttentionModel.train_epoch()` — `src/transformer_model/model.py`

```
for each batch (X, targets, target_masks, padding_masks, IDs):
    predictions, _ = encoder(X, padding_masks)           # forward pass
    loss = MaskedMSELoss(predictions, targets,
                         padding_masks.unsqueeze(-1))     # loss over real (non-padding) positions
    batch_loss = sum(loss)
    mean_loss  = batch_loss / num_active_elements
    total_loss = mean_loss + l2_reg * l2_reg_loss(encoder)
    total_loss.backward()
    clip_grad_norm_(encoder.parameters(), max_norm)       # gradient clipping
    optimizer.step()
```

> **Note**: `target_masks` (the noise mask) is currently **not** applied as a second
> filter in the loss call — the loss is computed over **all non-padding positions**,
> not just the masked ones.  The commented line
> `# target_masks = target_masks * padding_masks.unsqueeze(-1)` in the source shows
> where this was intended.  If you want true imputation loss (only on masked positions),
> uncomment that line and pass `target_masks` instead of `padding_masks.unsqueeze(-1)`.

### Learning rate schedule

```
every lr_step epochs:  lr = lr * lr_decay
```
Default: `lr_step=1`, `lr_decay=1.0` (constant LR).

### Early stopping

`EarlyStopping(patience, delta)` — stops training if validation loss does not improve
by more than `delta` for `patience` consecutive epochs.

---

## 5 — Loss Function

### `MaskedMSELoss` — `src/utils/model_helpers.py`

```python
masked_pred = torch.masked_select(y_pred, mask)   # flatten active elements
masked_true = torch.masked_select(y_true, mask)
return F.mse_loss(masked_pred, masked_true, reduction="none")
# → shape (num_active,)  — one squared error per active element
```

The mask passed is `padding_masks.unsqueeze(-1)` — a `(B, seq_len, 1)` tensor broadcast
across features.  **Data must be normalised** before this makes sense numerically.

---

## 6 — Normalisation

**All controlled in `src/utils/load_data.py` lines 76–117.**

| Mode | Behaviour | When to use |
|---|---|---|
| `standardization` ✅ **default** | Global mean/std fitted on training chunks only; applied to all chunks | VANET (absolute position semantics preserved across samples) |
| `minmax` | Global min/max fitted on training; scales to [0, 1] | If you need bounded outputs |
| `per_sample_std` | Each chunk is standardized by its own mean/std | **Not recommended for VANET** |
| `per_sample_minmax` | Each chunk scaled to [0,1] by its own range | **Not recommended for VANET** |
| `none` | No normalisation — will cause loss divergence to ~10^16 with VANET data | Never |

Zero-std features (e.g. `PacketLossRate = 0` in dense scenarios) are protected by
`safe_std = where(std < 1e-8, 1.0, std)` to prevent NaN.

---

## 7 — Configuration & CLI

All arguments are defined in `src/options.py`.  They are parsed by `Options().parse()`
and converted to a flat dictionary by `config_setup.setup()`.

### Most important arguments

```bash
# Data
--data_dir          path to CSV folder           [default: ./data]
--data_class        dataset class key            [default: sind]
--pattern           regex filter on filenames    [default: *]
--data_chunk_len    sequence length / window     [default: 50]
--val_ratio         fraction held out for val    [default: 0.2]
--data_normalization  standardization|minmax|none  [default: standardization]

# Training
--epochs            number of epochs             [default: 500]
--batch_size                                     [default: 256]
--lr                learning rate                [default: 0.0005]
--lr_step           decay every N epochs         [default: 1]
--lr_decay          decay multiplier             [default: 1.0]  (1.0 = no decay)
--optimizer         Adam | RAdam                 [default: Adam]
--l2_reg            L2 weight decay              [default: 0.05]
--max_grad_norm     gradient clipping threshold  [default: 4.0]
--early_stopping_patience                        [default: 10]
--harden            enable curriculum masking
--harden_step       curriculum step interval     [default: 10]

# Masking
--masking_ratio     proportion masked            [default: 0.1]
--mean_mask_length  avg masked segment length    [default: 15]
--mask_mode         separate | concurrent        [default: concurrent]
--mask_distribution geometric | bernoulli        [default: bernoulli]
--exclude_feats     comma-sep feature indices to never mask

# Model
--embedding_dim                                  [default: 128]
--hidden_dim                                     [default: 512]
--num_heads                                      [default: 8]
--num_layers                                     [default: 3]
--dropout                                        [default: 0.1]
--pos_encoding      fixed | learnable            [default: fixed]
--activation        relu | gelu                  [default: relu]
--normalization_layer  BatchNorm | LayerNorm     [default: BatchNorm]

# Evaluation / output
--eval_only         skip training, evaluate only
--save_embeddings   save encoder embeddings during eval
--output_dir                                     [default: ./experiments]
--name              experiment label
```

---

## 8 — Hyperparameter Tuning

**Entry-point**: `hyperparamer_tuning.py`

Uses [Ray Tune](https://docs.ray.io/en/latest/tune/) + [HyperOpt](http://hyperopt.github.io/hyperopt/).

```
main block:
  ray.init()
  HyperOptSearch(space=hyperparameter_config, metric="loss", mode="min")
  tune.Tuner(trainable=run, num_samples=N_ITER)
  results = tuner.fit()

run(hyperparameter_config):           # called once per trial
  override: data_normalization = "standardization"   ← ALWAYS forced
  build args_list from config dict
  Options().parse(args_list)
  main.run(setup(opts), session)
  tune.report({"loss": val_loss})     ← reported back to Ray
```

### Search space — `src/utils/hyperparemer_tuning_config.py`

| Parameter | Space |
|---|---|
| `lr` | uniform(1e-6, 1e-3) |
| `batch_size` | choice([16, 32, 64, 128, 256, 512]) |
| `epochs` | choice(50…1000 step 10) |
| `optimizer` | choice([Adam, RAdam]) |
| `l2_reg` | quniform(0, 0.2, 0.05) |
| `hidden_dim` | choice([256, 512]) |
| `dropout` | quniform(0, 0.2, 0.05) |
| `activation` | choice([relu, gelu]) |
| `data_normalization` | choice([standardization, minmax, none]) |
| `data_chunk_len` | choice(50…150 step 10) |
| `harden_step` | choice([10, 15, 20]) |
| `mask_mode` | choice([separate, concurrent]) |
| `mask_distribution` | choice([geometric, bernoulli]) |

### Smoke test (1 trial × 1 epoch)

```bash
python hyperparamer_tuning.py --smoke
```

---

## 9 — Output Artefacts

Each run of `main.py` creates a timestamped directory under `output_dir`:

```
experiments/
└── <name>_<YYYY-MM-DD_HH-MM-SS>_<rand>/
    ├── configuration.json      full config dump
    ├── output.log              training log
    ├── data_indices.json       train/val chunk indices
    ├── original_data.pt        val data (eval-only mode)
    ├── output_data.pt          per-batch predictions (eval-only mode)
    ├── checkpoints/
    │   ├── model_last.pth      latest epoch weights + optimiser state
    │   └── model_best.pth      best val-loss weights
    └── tb_summaries/           TensorBoard event files
```

Load a saved model:
```bash
python main.py --load_model experiments/.../checkpoints/model_best.pth --eval_only …
```
---

## 11 — Quick-Start Commands

### Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Smoke test (verify loss < 10 in ~2 min)

```bash
python main.py \
  --data_dir=resources/VANET_data/raw/ \
  --data_class=sind \
  --pattern=data_car_ \
  --data_normalization=standardization \
  --epochs=1 \
  --batch_size=32 \
  --pos_encoding=learnable \
  --name=smoke_test
```

### Full training run

```bash
python main.py \
  --data_dir=resources/VANET_data/raw/ \
  --data_class=sind \
  --pattern=data_car_ \
  --data_normalization=standardization \
  --epochs=500 \
  --batch_size=256 \
  --pos_encoding=learnable \
  --name=vanet_pretrain \
  --harden \
  --early_stopping_patience=20
```

### Evaluate a saved model

```bash
python main.py \
  --data_dir=resources/VANET_data/raw/ \
  --data_class=sind \
  --pattern=data_car_ \
  --data_normalization=standardization \
  --load_model=experiments/vanet_pretrain_.../checkpoints/model_best.pth \
  --eval_only \
  --save_embeddings \
  --name=eval_run
```

### Hyperparameter sweep

```bash
python hyperparamer_tuning.py          # full sweep (1000 trials)
python hyperparamer_tuning.py --smoke  # 1 trial × 1 epoch sanity check
```
