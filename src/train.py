"""Explicit-config training loop with stable, generic validation control."""
from dataclasses import asdict, is_dataclass
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

try:
    from .augmentation import CephalometricAugmentation
    from .data import ISBIDataset, CephalometricDataLoader
    from .losses import compute_total_loss
    from .model import (
        apply_encoder_policy, apply_trainable_scope, build_model, parameter_counts,
        unfreeze_encoder,
    )
    from .reproducibility import set_global_seed
    from .utils import compute_mre, compute_sdr, decode_coordinates_to_original
except ImportError:
    from augmentation import CephalometricAugmentation
    from data import ISBIDataset, CephalometricDataLoader
    from losses import compute_total_loss
    from model import (
        apply_encoder_policy, apply_trainable_scope, build_model, parameter_counts,
        unfreeze_encoder,
    )
    from reproducibility import set_global_seed
    from utils import compute_mre, compute_sdr, decode_coordinates_to_original


LEGACY_INITIAL_WEIGHTS = "/kaggle/input/model-weights/best_model.weights.h5"
_SUPPORTED_MONITORS = {"val_loss", "val_mre"}


def _set_memory_growth():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as error:
        if "initialized" not in str(error).lower():
            raise


_set_memory_growth()


def _make_optimizer(config, learning_rate=None):
    learning_rate = config.learning_rate if learning_rate is None else learning_rate
    optimizer_name = getattr(config, "optimizer", "adam").lower()
    if optimizer_name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if optimizer_name != "adamw":
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=config.weight_decay,
    )
    if getattr(config, "exclude_bias_and_norm_from_weight_decay", True):
        exclude = getattr(optimizer, "exclude_from_weight_decay", None)
        if callable(exclude):
            try:
                exclude(var_names=["bias", "beta", "gamma"])
            except (TypeError, ValueError):
                pass
    return optimizer


def _current_lr(optimizer):
    value = optimizer.learning_rate
    if callable(value):
        value = value(optimizer.iterations)
    return float(tf.keras.backend.get_value(value))


def _set_lr(optimizer, value):
    learning_rate = optimizer.learning_rate
    try:
        learning_rate.assign(value)
    except AttributeError:
        optimizer.learning_rate = value


def _validate_monitor(monitor):
    if monitor not in _SUPPORTED_MONITORS:
        raise ValueError(f"Unsupported monitor {monitor!r}; expected one of {sorted(_SUPPORTED_MONITORS)}")
    return monitor


def monitor_value(val_metrics, monitor):
    """Return a configured validation monitor value from evaluation metrics."""
    monitor = _validate_monitor(monitor)
    return float(val_metrics[monitor.removeprefix("val_")])


def is_monitor_improvement(value, best, min_delta=0.0, mode="min"):
    """Finite, minimum-delta-aware monitor comparison used by LR/stop logic."""
    if not math.isfinite(float(value)):
        return False
    if mode == "min":
        return float(value) < float(best) - float(min_delta)
    if mode == "max":
        return float(value) > float(best) + float(min_delta)
    raise ValueError("monitor_mode must be 'min' or 'max'")


def _strict_improvement(value, best, mode="min"):
    return is_monitor_improvement(value, best, min_delta=0.0, mode=mode)


def _prepare_model(config, model=None, encoder_layer_names=None):
    """Build, strictly load the incoming checkpoint once, and apply policy."""
    if model is None:
        model, encoder_layer_names = build_model(config)
    if encoder_layer_names is None:
        raise ValueError("encoder_layer_names is required when supplying a custom model")

    incoming_checkpoint = None
    if config.pretrained_weights:
        incoming_checkpoint = str(Path(config.pretrained_weights).expanduser())
        if not os.path.isfile(incoming_checkpoint):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {incoming_checkpoint}")
        model.load_weights(incoming_checkpoint)  # strict topology load
    elif config.stage_name == "stage1_initial_legacy" and os.path.isfile(LEGACY_INITIAL_WEIGHTS):
        incoming_checkpoint = LEGACY_INITIAL_WEIGHTS
        model.load_weights(incoming_checkpoint)

    apply_encoder_policy(
        model,
        encoder_layer_names,
        config.encoder_mode,
        freeze_batch_norm=getattr(config, "freeze_encoder_batch_norm", False),
    )
    scope = getattr(config, "trainable_scope", None)
    counts = apply_trainable_scope(model, encoder_layer_names, scope) if scope else parameter_counts(model)
    return model, encoder_layer_names, incoming_checkpoint, counts


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _config_snapshot(config):
    if hasattr(config, "snapshot"):
        return config.snapshot()
    if is_dataclass(config):
        return asdict(config)
    return {key: value for key, value in vars(config).items() if not key.startswith("_")}


def _metric_series(history, prefix):
    for suffix in ("loss", "hm_loss", "coord_loss", "weighted_hm_loss", "weighted_coord_loss", "mre"):
        history[f"{prefix}_{suffix}"] = []


def _empty_history(config, incoming_checkpoint, baseline, counts, paths, train_eval_baseline=None):
    history = {
        "schema_version": 3,
        "stage_name": config.stage_name,
        "model_version": getattr(config, "model_version", "legacy"),
        "objective_version": getattr(config, "objective_version", "legacy"),
        "seed": int(getattr(config, "seed", 42)),
        "incoming_checkpoint": incoming_checkpoint,
        "output_checkpoint": paths["primary"],  # compatibility
        "checkpoint_paths": paths,
        "config": _config_snapshot(config),
        "baseline": baseline,
        "train_eval_baseline": train_eval_baseline,
        "experiment_name": getattr(config, "experiment_name", None),
        "sample_count": len(getattr(config, "train_image_ids", ())),
        "split_ids": {
            "train": list(getattr(config, "train_image_ids", ())),
            "validation": list(getattr(config, "val_image_ids", ())),
            "unused": list(getattr(config, "unused_image_ids", ())),
        },
        "trainable_scope": getattr(config, "trainable_scope", None),
        "parameter_counts": counts,
        "monitor_settings": {
            "checkpoint": getattr(config, "checkpoint_monitor", "val_mre"),
            "lr_scheduler": getattr(config, "lr_scheduler_monitor", "val_mre"),
            "early_stopping": getattr(config, "early_stopping_monitor", "val_mre"),
            "mode": getattr(config, "monitor_mode", "min"),
            "lr_min_delta": float(getattr(config, "lr_plateau_min_delta", 0.0)),
            "early_stopping_min_delta": float(getattr(config, "early_stopping_min_delta", 0.0)),
        },
        "epoch": [], "learning_rate": [],
        "val_loss_running_best": [],
        "val_sdr_2mm": [], "val_sdr_2_5mm": [], "val_sdr_3mm": [], "val_sdr_4mm": [],
        "nan_batches": [], "lr_reduced": [],
        "best_epoch": 0, "best_val_mre": float(baseline["mre"]),  # compatibility
        "best_loss_epoch": 0, "best_loss_value": float(baseline["loss"]),
        "best_mre_epoch": 0, "best_mre_value": float(baseline["mre"]),
        "primary_best_epoch": 0,
        "primary_best_value": monitor_value(baseline, getattr(config, "checkpoint_monitor", "val_mre")),
        "stopped_epoch": 0, "stop_reason": "maximum_epochs",
        "restored_best_weights": False,
        "restored_checkpoint": None,
        "restored_final_validation": None,
    }
    _metric_series(history, "train_online")
    _metric_series(history, "train_eval")
    _metric_series(history, "val")
    # Old train_* means online augmented training and remains for compatibility.
    for suffix in ("loss", "hm_loss", "coord_loss", "weighted_hm_loss", "weighted_coord_loss", "mre"):
        history[f"train_{suffix}"] = []
    return history


def _append_metrics(history, prefix, metrics):
    for suffix in ("loss", "hm_loss", "coord_loss", "weighted_hm_loss", "weighted_coord_loss", "mre"):
        history[f"{prefix}_{suffix}"].append(float(metrics[suffix]))


def train(config, train_loader=None, val_loader=None, model=None, encoder_layer_names=None,
          train_eval_loader=None):
    """Train one stage with generic monitors and distinct loss/MRE checkpoints.

    ``train_loader`` supplies online (possibly augmented) metrics.
    ``train_eval_loader`` must be a separate unaugmented, non-shuffled loader;
    when supplied it is evaluated in inference mode after every epoch.
    """
    set_global_seed(config.seed)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    if train_loader is None or val_loader is None:
        train_dataset = ISBIDataset(config.dataset_root, config, mode="train")
        val_dataset = ISBIDataset(config.dataset_root, config, mode="test")
        augmentation = CephalometricAugmentation(config) if config.augmentation_enabled else None
        train_loader = CephalometricDataLoader(train_dataset, config, shuffle=True, augmentation=augmentation)
        val_loader = CephalometricDataLoader(val_dataset, config, shuffle=False, augmentation=None)

    mode = getattr(config, "monitor_mode", "min")
    checkpoint_monitor = _validate_monitor(getattr(config, "checkpoint_monitor", "val_mre"))
    scheduler_monitor = _validate_monitor(getattr(config, "lr_scheduler_monitor", "val_mre"))
    stopping_monitor = _validate_monitor(getattr(config, "early_stopping_monitor", "val_mre"))

    model, encoder_layer_names, incoming_checkpoint, counts = _prepare_model(
        config, model, encoder_layer_names
    )
    print(
        f"Parameters — trainable: {counts['trainable']:,}; "
        f"non-trainable: {counts['non_trainable']:,}; total: {counts['total']:,}"
    )
    initial_lr = config.warmup_learning_rate if config.warmup_epochs > 0 else config.learning_rate
    optimizer = _make_optimizer(config, initial_lr)
    primary_path = os.path.join(config.checkpoint_dir, config.output_filename)
    mre_name = getattr(config, "best_mre_output_filename", None)
    mre_path = os.path.join(config.checkpoint_dir, mre_name) if mre_name else primary_path
    paths = {"primary": primary_path, "best_mre": mre_path}

    baseline = evaluate(model, val_loader, config)
    train_eval_baseline = evaluate(model, train_eval_loader, config) if train_eval_loader is not None else None
    history = _empty_history(config, incoming_checkpoint, baseline, counts, paths, train_eval_baseline)

    best_loss, best_loss_epoch = float(baseline["loss"]), 0
    best_mre, best_mre_epoch = float(baseline["mre"]), 0
    primary_best = monitor_value(baseline, checkpoint_monitor)
    primary_best_epoch = 0
    model.save_weights(primary_path)
    if mre_path != primary_path:
        model.save_weights(mre_path)
    print(
        f"Epoch 0 {config.stage_name} baseline — val loss: {best_loss:.5f}; "
        f"val MRE: {best_mre:.2f} mm"
    )

    scheduler_best = monitor_value(baseline, scheduler_monitor)
    stopping_best = monitor_value(baseline, stopping_monitor)
    scheduler_wait = stopping_wait = 0
    stopped_early = False
    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}\n" + "-" * 40)
        if config.encoder_mode == "stage1" and not getattr(config, "trainable_scope", None) and epoch == config.warmup_epochs + 1:
            print("  Unfreezing backbone — switching to Stage 1 post-warm-up LR")
            unfreeze_encoder(
                model, encoder_layer_names,
                freeze_batch_norm=getattr(config, "freeze_encoder_batch_norm", False),
            )
            optimizer = _make_optimizer(config, config.learning_rate)

        epoch_lr = _current_lr(optimizer)
        online = train_one_epoch(model, train_loader, optimizer, config)
        train_eval = evaluate(model, train_eval_loader, config) if train_eval_loader is not None else _empty_epoch_metrics()
        val = evaluate(model, val_loader, config)
        sdr = val["sdr"]
        print(
            f"  Train online — loss {online['loss']:.5f} | MRE {online['mre']:.2f} mm"
        )
        if train_eval_loader is not None:
            print(f"  Train eval   — loss {train_eval['loss']:.5f} | MRE {train_eval['mre']:.2f} mm")
        print(
            f"  Validation   — loss {val['loss']:.5f} | MRE {val['mre']:.2f} mm | "
            f"SDR@2 {sdr.get(2.0, float('nan')):.1f}% | SDR@2.5 {sdr.get(2.5, float('nan')):.1f}% | "
            f"SDR@3 {sdr.get(3.0, float('nan')):.1f}% | SDR@4 {sdr.get(4.0, float('nan')):.1f}%"
        )
        if online["nan_batches"]:
            print(f"  Skipped {online['nan_batches']} non-finite batches")

        val_loss, val_mre = float(val["loss"]), float(val["mre"])
        if _strict_improvement(val_loss, best_loss, "min"):
            best_loss, best_loss_epoch = val_loss, epoch
        if _strict_improvement(val_mre, best_mre, "min"):
            best_mre, best_mre_epoch = val_mre, epoch
            model.save_weights(mre_path)
            print(f"  New best MRE {best_mre:.3f} mm — saved {mre_path}")

        primary_value = monitor_value(val, checkpoint_monitor)
        if _strict_improvement(primary_value, primary_best, mode):
            primary_best, primary_best_epoch = primary_value, epoch
            model.save_weights(primary_path)
            print(f"  New best {checkpoint_monitor} {primary_best:.6g} — saved {primary_path}")

        stop_value = monitor_value(val, stopping_monitor)
        if is_monitor_improvement(stop_value, stopping_best, config.early_stopping_min_delta, mode):
            stopping_best, stopping_wait = stop_value, 0
        else:
            stopping_wait += 1

        scheduler_value = monitor_value(val, scheduler_monitor)
        if is_monitor_improvement(scheduler_value, scheduler_best, config.lr_plateau_min_delta, mode):
            scheduler_best, scheduler_wait = scheduler_value, 0
        else:
            scheduler_wait += 1
        lr_reduced = False
        if config.lr_plateau_patience > 0 and scheduler_wait >= config.lr_plateau_patience:
            old_lr = _current_lr(optimizer)
            new_lr = max(config.min_learning_rate, old_lr * config.lr_plateau_factor)
            if new_lr < old_lr:
                _set_lr(optimizer, new_lr)
                lr_reduced = True
                print(f"  {scheduler_monitor} plateau: learning rate {old_lr:.3g} -> {new_lr:.3g}")
            scheduler_wait = 0

        if config.checkpoint_interval > 0 and epoch % config.checkpoint_interval == 0:
            periodic_path = os.path.join(config.checkpoint_dir, f"{config.stage_name}_epoch_{epoch:03d}.weights.h5")
            model.save_weights(periodic_path)

        history["epoch"].append(epoch)
        history["learning_rate"].append(epoch_lr)
        _append_metrics(history, "train_online", online)
        _append_metrics(history, "train_eval", train_eval)
        _append_metrics(history, "val", val)
        for suffix in ("loss", "hm_loss", "coord_loss", "weighted_hm_loss", "weighted_coord_loss", "mre"):
            history[f"train_{suffix}"].append(float(online[suffix]))
        history["val_loss_running_best"].append(best_loss)
        history["val_sdr_2mm"].append(sdr.get(2.0, float("nan")))
        history["val_sdr_2_5mm"].append(sdr.get(2.5, float("nan")))
        history["val_sdr_3mm"].append(sdr.get(3.0, float("nan")))
        history["val_sdr_4mm"].append(sdr.get(4.0, float("nan")))
        history["nan_batches"].append(online["nan_batches"])
        history["lr_reduced"].append(lr_reduced)
        history.update({
            "best_epoch": best_mre_epoch if checkpoint_monitor == "val_mre" else primary_best_epoch,
            "best_val_mre": best_mre,
            "best_loss_epoch": best_loss_epoch,
            "best_loss_value": best_loss,
            "best_mre_epoch": best_mre_epoch,
            "best_mre_value": best_mre,
            "primary_best_epoch": primary_best_epoch,
            "primary_best_value": primary_best,
        })
        train_loader.on_epoch_end()

        if config.early_stopping_patience > 0 and stopping_wait >= config.early_stopping_patience:
            stopped_early = True
            history["stopped_epoch"] = epoch
            history["stop_reason"] = f"early_stopping_{stopping_monitor}"
            print(f"  Early stopping after {stopping_wait} epochs without meaningful {stopping_monitor} improvement")
            break

    if config.restore_best_weights:
        model.load_weights(primary_path)
        history["restored_best_weights"] = True
        history["restored_checkpoint"] = primary_path
        history["restored_final_validation"] = evaluate(model, val_loader, config)
        print(f"Restored primary best from epoch {primary_best_epoch}: {primary_path}")
    if not stopped_early:
        history["stopped_epoch"] = history["epoch"][-1] if history["epoch"] else 0

    history_name = config.history_filename or f"{config.stage_name}_training_history.json"
    history_path = os.path.join(config.log_dir, history_name)
    history["history_path"] = history_path
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(history), handle, indent=2, allow_nan=False)
    print(f"Training history saved to: {history_path}")
    return history


def train_step(model, images, true_heatmaps, true_coords, optimizer, config):
    with tf.GradientTape() as tape:
        outputs = model(images, training=True)
        components = compute_total_loss(
            outputs["heatmaps"], true_heatmaps, outputs["coords"], true_coords,
            config=config, return_components=True,
        )
    if not bool(tf.reduce_all(tf.math.is_finite(components["total"])).numpy()):
        return components, outputs["coords"], True
    gradients = tape.gradient(components["total"], model.trainable_variables)
    pairs = [(gradient, variable) for gradient, variable in zip(gradients, model.trainable_variables) if gradient is not None]
    if any(not bool(tf.reduce_all(tf.math.is_finite(gradient)).numpy()) for gradient, _ in pairs):
        return components, outputs["coords"], True
    if pairs:
        clipped, _ = tf.clip_by_global_norm([gradient for gradient, _ in pairs], config.gradient_clip_norm)
        optimizer.apply_gradients((gradient, variable) for gradient, (_, variable) in zip(clipped, pairs))
    return components, outputs["coords"], False


def _empty_epoch_metrics(nan_batches=0):
    return {
        "loss": float("nan"), "hm_loss": float("nan"), "coord_loss": float("nan"),
        "weighted_hm_loss": float("nan"), "weighted_coord_loss": float("nan"),
        "mre": float("nan"), "nan_batches": nan_batches,
    }


def train_one_epoch(model, data_loader, optimizer, config):
    keys = ("total", "heatmap", "coordinate", "weighted_heatmap", "weighted_coordinate")
    sums = {key: 0.0 for key in keys}
    all_pred, all_true = [], []
    valid = 0
    nan_batches = 0
    for batch_index in range(len(data_loader)):
        images, targets = data_loader[batch_index]
        components, pred_coords, skipped = train_step(
            model, images, targets["heatmaps"], targets["landmarks"], optimizer, config
        )
        if skipped:
            nan_batches += 1
            continue
        valid += 1
        for key in keys:
            sums[key] += float(components[key].numpy())
        all_pred.append(decode_coordinates_to_original(pred_coords.numpy(), config.original_width, config.original_height))
        all_true.append(decode_coordinates_to_original(np.asarray(targets["landmarks"]), config.original_width, config.original_height))
    if not valid:
        return _empty_epoch_metrics(nan_batches)
    mre, _ = compute_mre(np.concatenate(all_pred), np.concatenate(all_true), config.image_resolution)
    return {
        "loss": sums["total"] / valid,
        "hm_loss": sums["heatmap"] / valid,
        "coord_loss": sums["coordinate"] / valid,
        "weighted_hm_loss": sums["weighted_heatmap"] / valid,
        "weighted_coord_loss": sums["weighted_coordinate"] / valid,
        "mre": mre,
        "nan_batches": nan_batches,
    }


def evaluate(model, data_loader, config):
    if data_loader is None:
        raise ValueError("An evaluation loader is required")
    keys = ("total", "heatmap", "coordinate", "weighted_heatmap", "weighted_coordinate")
    sums = {key: 0.0 for key in keys}
    all_pred, all_true = [], []
    valid = 0
    for batch_index in range(len(data_loader)):
        images, targets = data_loader[batch_index]
        outputs = model(images, training=False)
        components = compute_total_loss(
            outputs["heatmaps"], targets["heatmaps"], outputs["coords"], targets["landmarks"],
            config=config, return_components=True,
        )
        if not bool(tf.reduce_all(tf.math.is_finite(components["total"])).numpy()):
            continue
        valid += 1
        for key in keys:
            sums[key] += float(components[key].numpy())
        all_pred.append(decode_coordinates_to_original(outputs["coords"].numpy(), config.original_width, config.original_height))
        all_true.append(decode_coordinates_to_original(np.asarray(targets["landmarks"]), config.original_width, config.original_height))
    if not all_pred:
        raise ValueError("No finite complete evaluation batch is available")
    pred, true = np.concatenate(all_pred), np.concatenate(all_true)
    mre, per_landmark = compute_mre(pred, true, config.image_resolution)
    return {
        "loss": sums["total"] / valid,
        "hm_loss": sums["heatmap"] / valid,
        "coord_loss": sums["coordinate"] / valid,
        "weighted_hm_loss": sums["weighted_heatmap"] / valid,
        "weighted_coord_loss": sums["weighted_coordinate"] / valid,
        "mre": mre,
        "per_landmark_mre": per_landmark,
        "sdr": compute_sdr(pred, true, config.precision_thresholds, config.image_resolution),
    }


if __name__ == "__main__":
    try:
        from .config import make_stage1_initial_config
    except ImportError:
        from config import make_stage1_initial_config
    train(make_stage1_initial_config())


__all__ = [
    "evaluate", "is_monitor_improvement", "monitor_value", "train", "train_one_epoch", "train_step",
]
