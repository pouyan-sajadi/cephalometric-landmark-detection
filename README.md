# Cephalometric Landmark Detection

TensorFlow/Keras research code for localising 19 landmarks in lateral cephalometric radiographs with a ResNet50 encoder and U-Net-like decoder.

## Two-stage training workflow

Both stages use the stable v2 spatial-softmax objective. Stage 1 must be run before Stage 2:

```text
ImageNet encoder weights
        └── Stage 1 v2 ──> best_model_fixed.weights.h5
                                   └── Stage 2 v3 (in-house fine-tuning)
```

- **Stage 1 (pre-training):** starts from ImageNet encoder weights and learns general cephalometric representations from the public ISBI dataset. Uses spatial-softmax maps, soft argmax, spatial KL, smooth-L1, AdamW, and validation-loss scheduling/stopping/checkpoint selection. Writes the primary best-loss checkpoint to `best_model_fixed.weights.h5` and a secondary best-MRE checkpoint to `best_model_stage1_v2_best_mre.weights.h5`.
- **Stage 2 (fine-tuning):** initialises from the Stage 1 pre-trained model and adapts it to the in-house domain with a deliberately small trainable scope. The encoder and first two decoder blocks are frozen; only the final decoder block, head, and landmark-output layer are updated.

Dropout layers are unweighted, so Stage 2 can use decoder dropout 0.20 while strictly loading the Stage 1 weighted topology. All weighted layer names and shapes through `logits` remain H5-compatible.

## Stage 2 anti-overfitting protocol

Default Stage 2 fine-tuning settings:

| Setting | Value |
|---|---:|
| Maximum epochs | 50 |
| Batch size | 2 |
| Optimizer | AdamW |
| Learning rate | `5e-6` |
| Weight decay | `5e-5` |
| Decoder dropout | 0.20 |
| Trainable scope | `dec2_head_logits` |
| LR scheduler | `val_loss`, factor 0.5, patience 2, min delta `2e-4` |
| Early stopping | `val_loss`, patience 5, min delta `2e-4` |
| Primary checkpoint | lowest raw validation loss |
| Secondary checkpoint | lowest raw validation MRE |

The complete ResNet encoder and decoder blocks `dec4` and `dec3` are frozen. By default only `dec2`, `head`, and `logits` train. The `head_logits` scope is available for more conservative experiments. Dropout has no trainable parameters and is active only during online training (`training=True`).

Stage 2 uses modest augmentation: no horizontal reflection, ±5° rotation, ±5% scale, ±2.5% translation, and mild brightness, contrast, and noise. If a geometric transform sends any landmark out of frame, the whole geometric transform is rejected instead of silently clipping the target to the boundary.

## Manual 10/60 and 20/60 experiments

`src.config.validate_finetune_split()` and `make_finetune_experiment_config()` support user-selected lists:

```python
TRAIN_IMAGE_IDS = [...]  # 10 or 20 unique integers in 1..80
VAL_IMAGE_IDS = [...]    # exactly 60 unique integers in 1..80
EXPERIMENT_NAME = "finetune_n20"
TRAINABLE_SCOPE = "dec2_head_logits"  # or "head_logits"
```

Rules:

- training and validation IDs cannot overlap;
- 20/60 must account for all 80 IDs;
- 10/60 reports the remaining ten IDs as unused;
- an optional 20-image reference list asserts that the n10 training set is nested within the selected n20 set;
- artifact names contain `n10` or `n20` and separate `best_loss` and `best_mre`, preventing accidental overwrite.

The constants `FINETUNE_TRAIN_IDS` and `FINETUNE_VAL_IDS` remain defaults/reference lists.

## Notebook 02 workflow

`notebooks/02-finetuning-and-evaluation.ipynb` is ordered as follows:

1. locate the repository, in-house data, results paths, and pre-trained checkpoint;
2. seed before data/model creation;
3. build an inference-only ranking model and strictly load the pre-trained model;
4. evaluate IDs 1–80 individually without augmentation and save the ranking CSV;
5. let the user enter the experiment split/name/scope;
6. validate the split and build augmented online train loader, separate unaugmented train-evaluation loader, and unaugmented validation loader;
7. build a fresh training model and let `train()` perform its one strict checkpoint load;
8. plot unaugmented `train_eval_loss` against raw `val_loss`, with running-best validation loss separately labelled;
9. report both best-loss and best-MRE outcomes and leave the in-memory model restored to best loss.

Ranking is a model-guided difficulty audit, not an independent test. Because the 60 images drive scheduling, stopping, and checkpoint selection, they are validation data — not an untouched independent test cohort.

## Stage 1 notebook

`notebooks/01-pretraining.ipynb` preserves the ISBI data-loader construction. It uses a separate unaugmented, non-shuffled training-evaluation loader. Plots show unaugmented training versus raw validation loss, weighted validation KL/coordinate components, MRE, learning rate, and running-best validation loss. The notebook reports both selected checkpoint paths, best epochs, stopping reason, and restored-best validation metrics.

## Training history

Schema-v3 histories distinguish:

- `train_online_*`: metrics observed during augmented training;
- `train_eval_*`: the training IDs re-evaluated without augmentation in inference mode after every epoch;
- `val_*`: raw unaugmented validation metrics;
- `val_loss_running_best`: a separate running minimum for plotting; raw `val_loss` is never replaced or smoothed.

Metadata includes split/unused IDs, experiment name and sample count, config snapshot, trainable scope and parameter counts, all monitor settings, baseline metrics, best-loss and best-MRE epochs/values, stop reason, checkpoint/history paths, and restored-best-loss final validation metrics.

## Paths

```bash
export CEPHALO_PRETRAINED_WEIGHTS=/absolute/path/to/best_model_fixed.weights.h5
export CEPHALO_FT_ROOT=/absolute/path/to/inhouse-data
export CEPHALO_CHECKPOINT_DIR=/absolute/path/to/checkpoints
export CEPHALO_LOG_DIR=/absolute/path/to/results
```

`CEPHALO_PRETRAINED_WEIGHTS` sets the Stage 1 checkpoint used by Stage 2 fine-tuning. On Kaggle the notebook resolves this automatically; set it locally when running outside Kaggle.

## Metric and result qualification

MRE/SDR use the project's fixed 1935 × 2400 reference dimensions and nominal 0.1 mm/pixel scale. Verify acquisition spacing and crop transforms before clinical interpretation.

Validation results are not independent testing results. The 60 in-house images inform learning-rate adjustment, early stopping, and model selection.
