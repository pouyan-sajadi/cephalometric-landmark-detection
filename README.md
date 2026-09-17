# Cephalometric Landmark Detection

TensorFlow/Keras research code for localizing 19 landmarks in lateral cephalometric radiographs with a ResNet50 encoder and U-Net-like decoder.

## Active checkpoint workflow

The active workflow uses the stable v2 objective throughout. Stage 1 must be rerun before rerunning downstream stages:

```text
best_model_fixed.weights.h5
        └── Stage 2 v2 ──> best_model_stage2_v2.weights.h5
                                  └── Stage 3 anti-overfitting experiments
```

- **Stage 1 v2:** starts from ImageNet encoder weights and uses spatial-softmax maps, soft argmax, spatial KL, smooth-L1, AdamW, and validation-loss scheduling/stopping/checkpoint selection. It writes the primary best-loss checkpoint to `best_model_fixed.weights.h5` and a secondary best-MRE checkpoint to `best_model_stage1_v2_best_mre.weights.h5`.
- **Stage 2 v2:** initializes from the newly rerun Stage 1 `best_model_fixed.weights.h5` and retains its existing v2 trainability/monitor defaults.
- **Stage 3 v3:** strictly initializes from the trained Stage 2 v2 checkpoint. Supply it through `CEPHALO_STAGE2_WEIGHTS` or a supported repository/Kaggle path.

Dropout layers are unweighted, so Stage 3 can use decoder dropout 0.20 while strictly loading the Stage 2 weighted topology. All weighted layer names and shapes through `logits` remain H5-compatible.

## Stage 3 anti-overfitting protocol

Default Stage 3 settings are:

| Setting | Value |
|---|---:|
| Maximum epochs | 25 |
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

The complete ResNet encoder—including every BatchNormalization layer—and decoder blocks `dec4` and `dec3` are frozen. By default only `dec2`, `head`, and `logits` train. The `head_logits` scope is available for the more conservative 10-image experiment. Dropout has no trainable parameters and is active only during online training (`training=True`). Histories record trainable/non-trainable counts.

Stage 3 uses modest augmentation: no horizontal reflection, ±5° rotation, ±5% scale, ±2.5% translation, and mild brightness, contrast, and noise. If a geometric transform sends any landmark out of frame, Stage 3 rejects that whole geometric transform instead of silently clipping the target to the boundary. Legacy stages retain their clipping behavior.

No loss trajectory is guaranteed to decrease monotonically. The intended behavior is an early useful decrease, followed by stopping when validation loss no longer improves meaningfully and restoration of the observed minimum.

## Manual 10/60 and 20/60 experiments

`src.config.validate_stage3_split()` and `make_stage3_experiment_config()` support user-selected lists:

```python
TRAIN_IMAGE_IDS = [...]  # 10 or 20 unique integers in 1..80
VAL_IMAGE_IDS = [...]    # exactly 60 unique integers in 1..80
EXPERIMENT_NAME = "stage3_n20"
TRAINABLE_SCOPE = "dec2_head_logits"  # or "head_logits"
```

Rules:

- training and validation IDs cannot overlap;
- 20/60 must account for all 80 IDs;
- 10/60 reports the remaining ten IDs as unused;
- an optional 20-image reference list asserts that the n10 training set is nested within the selected n20 set;
- artifact names contain `n10` or `n20` and separate `best_loss` and `best_mre`, preventing accidental overwrite.

The constants `STAGE3_TRAIN_IDS` and `STAGE3_VAL_IDS` remain defaults/reference lists, not mandatory hard-wired choices.

## Notebook 03 workflow

`notebooks/03-finetuning-and-evaluation.ipynb` is ordered as follows:

1. robustly locate the repository, in-house data, results paths, and Stage 2 v2 checkpoint;
2. seed before data/model creation;
3. build an inference-only ranking model and strictly load Stage 2 once;
4. evaluate IDs 1–80 individually without augmentation and save `stage2_all80_image_ranking.csv`;
5. let the user enter the experiment split/name/scope in a separate cell;
6. validate the split and build an augmented online train loader, separate unaugmented train-evaluation loader, and unaugmented validation loader;
7. build a fresh training model and let `train()` perform its one strict checkpoint load;
8. plot unaugmented `train_eval_loss` against **raw** `val_loss`, with running-best validation loss separately labelled and epoch zero labelled Stage 2;
9. report both best-loss and best-MRE outcomes and leave the in-memory model restored to best loss.

Ranking is a model-guided difficulty audit, not an independent test. Selecting only hard images for training and easy images for validation biases results. Difficulty-stratified selection is recommended. Because the 60 images drive scheduling, stopping, and checkpoint selection, they are validation data—not an untouched independent test cohort.

## Stage 1 notebook and validation tracing

`notebooks/01-pretraining-initial-v3.ipynb` preserves its existing repository discovery and ISBI data-loader construction. Its downstream training cell now adds a separate unaugmented, non-shuffled training-evaluation loader and passes it to `train()`. The resulting plots show unaugmented training versus raw validation loss, weighted validation KL/coordinate components, MRE, learning rate, and the separately labelled running-best validation loss. The notebook reports both selected checkpoint paths, best epochs, stopping reason, and restored-best validation metrics.

Archived `docs/initial_training_log.json` and old notebook output belong to the former legacy Stage 1 run; they do not document the new v2 checkpoint. Rerun Notebook 01 and preserve `stage1_initial_v2_training_history.json` before making new Stage 1 claims.

## Training history

Schema-v3 histories distinguish:

- `train_online_*`: metrics observed during augmented training;
- `train_eval_*`: the training IDs re-evaluated without augmentation in inference mode after every epoch;
- `val_*`: raw unaugmented validation metrics;
- `val_loss_running_best`: a separate running minimum for honest plotting; raw `val_loss` is never replaced or smoothed.

Metadata includes split/unused IDs, experiment name and sample count, config snapshot, trainable scope and parameter counts, all monitor settings, epoch-zero Stage 2 baseline, best-loss and best-MRE epochs/values, stop reason, checkpoint/history paths, and restored-best-loss final validation metrics. Old `train_*`, `best_epoch`, and `best_val_mre` keys remain for compatibility.

## Paths

```bash
export CEPHALO_STAGE2_WEIGHTS=/absolute/path/to/best_model_stage2_v2.weights.h5
export CEPHALO_FT_ROOT=/absolute/path/to/inhouse-data
export CEPHALO_CHECKPOINT_DIR=/absolute/path/to/checkpoints
export CEPHALO_LOG_DIR=/absolute/path/to/results
```

`CEPHALO_FIXED_WEIGHTS` remains the Stage 2 incoming-checkpoint variable; it is not Stage 3's active input.

## Validation

Target environment: TensorFlow 2.17.

```bash
python3 -m compileall src tests
python3 -m json.tool notebooks/01-pretraining-initial-v3.ipynb >/dev/null
python3 -m json.tool notebooks/03-finetuning-and-evaluation.ipynb >/dev/null
python3 -m pytest -q tests/test_stage3_antioverfit.py tests/test_stability_v2.py
```

Tests cover selective freezing and parameter policy, split validation/subset enforcement, deterministic per-image ranking metrics, generic monitor comparisons, history schema, and strict Stage 2 loading when the real checkpoint is locally available. These checks do not run full training.

## Metric and legacy-result qualification

MRE/SDR use the project's fixed 1935 × 2400 reference dimensions and nominal 0.1 mm/pixel scale. Verify acquisition spacing and crop transforms before clinical interpretation.

Existing logs and archived notebook outputs remain legacy provenance. Do not attribute their results to this Stage 3 protocol. New claims require the exact Stage 2 input checkpoint, ranking CSV, manual split, versioned history, and selected checkpoint. Validation results are not independent testing results.
