"""Explicit-config checkpoint evaluation and compatibility validation."""
import os
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from .data import ISBIDataset, CephalometricDataLoader, InHouseDataset
    from .model import build_model
    from .utils import compute_mre, compute_sdr, decode_coordinates_to_original
except ImportError:
    from data import ISBIDataset, CephalometricDataLoader, InHouseDataset
    from model import build_model
    from utils import compute_mre, compute_sdr, decode_coordinates_to_original


def load_compatible_weights(model, weights_path, config=None):
    """Strictly load H5 weights and annotate errors with protocol context."""
    path = Path(weights_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        status = model.load_weights(str(path))
    except Exception as error:
        version = getattr(config, "model_version", "unknown") if config is not None else "unknown"
        raise ValueError(
            f"Checkpoint {path} is incompatible with model_version={version!r}. "
            "The weighted ResNet/decoder/logits topology must match Stage 1; "
            "no mismatches were skipped."
        ) from error
    return status


def validate_checkpoint_compatibility(config, weights_path, model=None):
    """Build the configured graph, strictly load a checkpoint, and return facts."""
    if model is None:
        model, _ = build_model(config)
    load_compatible_weights(model, weights_path, config)
    heatmap_shape = tuple(model.output["heatmaps"].shape)
    coordinate_shape = tuple(model.output["coords"].shape)
    expected_heatmaps = (None, config.heatmap_height, config.heatmap_width, config.num_landmarks)
    expected_coords = (None, config.num_landmarks, 2)
    if heatmap_shape != expected_heatmaps or coordinate_shape != expected_coords:
        raise ValueError(
            f"Unexpected model outputs: {heatmap_shape}, {coordinate_shape}; "
            f"expected {expected_heatmaps}, {expected_coords}"
        )
    return {
        "checkpoint": str(Path(weights_path).expanduser().resolve()),
        "model_version": getattr(config, "model_version", "legacy"),
        "objective_version": getattr(config, "objective_version", "legacy"),
        "heatmap_shape": heatmap_shape,
        "coordinate_shape": coordinate_shape,
    }


def evaluate_model(config, weights_path, mode="test"):
    dataset = ISBIDataset(config.dataset_root, config, mode=mode)
    loader = CephalometricDataLoader(dataset, config, shuffle=False, augmentation=None)
    model, _ = build_model(config)
    load_compatible_weights(model, weights_path, config)
    if len(loader) == 0:
        raise ValueError("Dataset is smaller than one complete evaluation batch")

    predictions, truths = [], []
    for batch_index in range(len(loader)):
        images, targets = loader[batch_index]
        outputs = model(images, training=False)
        heatmaps = outputs["heatmaps"].numpy()
        coords = outputs["coords"].numpy()
        if not np.isfinite(heatmaps).all() or not np.isfinite(coords).all():
            raise FloatingPointError(f"Non-finite model output in evaluation batch {batch_index}")
        if getattr(config, "objective_version", "legacy").lower() == "v2":
            sums = heatmaps.sum(axis=(1, 2))
            if not np.allclose(sums, 1.0, atol=1e-5):
                raise ValueError("V2 heatmaps are not normalized spatial probabilities")
        predictions.append(decode_coordinates_to_original(coords, config.original_width, config.original_height))
        truths.append(decode_coordinates_to_original(np.asarray(targets["landmarks"]), config.original_width, config.original_height))
    predictions, truths = np.concatenate(predictions), np.concatenate(truths)
    mre, per_landmark = compute_mre(predictions, truths, config.image_resolution)
    sdr = compute_sdr(predictions, truths, config.precision_thresholds, config.image_resolution)
    print(f"MRE: {mre:.3f} mm")
    for threshold, rate in sdr.items():
        print(f"SDR@{threshold}mm: {rate:.1f}%")
    return mre, sdr, per_landmark


def rank_per_image_predictions(image_ids, predicted_norm, true_norm, config):
    """Return a deterministic worst-to-best per-image difficulty table.

    This pure metric helper accepts synthetic arrays and therefore does not
    require model construction or training in tests. Rank 1 is the worst MRE;
    equal MREs are ordered by image ID.
    """
    image_ids = list(image_ids)
    predicted_norm = np.asarray(predicted_norm, dtype=np.float64)
    true_norm = np.asarray(true_norm, dtype=np.float64)
    expected = (len(image_ids), config.num_landmarks, 2)
    if predicted_norm.shape != expected or true_norm.shape != expected:
        raise ValueError(f"Expected prediction/truth shapes {expected}; got {predicted_norm.shape}/{true_norm.shape}")
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("image_ids must be unique")
    predicted = decode_coordinates_to_original(predicted_norm, config.original_width, config.original_height)
    truth = decode_coordinates_to_original(true_norm, config.original_width, config.original_height)
    distances = np.linalg.norm(predicted - truth, axis=-1) * config.image_resolution
    rows = []
    for index, image_id in enumerate(image_ids):
        row = {
            "image_id": int(image_id),
            "mre_mm": float(np.mean(distances[index])),
            "max_landmark_error_mm": float(np.max(distances[index])),
        }
        for threshold in config.precision_thresholds:
            label = str(float(threshold)).replace(".", "_")
            row[f"sdr_{label}mm_percent"] = float(np.mean(distances[index] < threshold) * 100.0)
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(
        ["mre_mm", "image_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    table.insert(0, "rank_worst_to_best", np.arange(1, len(table) + 1, dtype=int))
    return table


def rank_inhouse_images(model, config, image_ids=range(1, 81), root=None,
                        image_dir=None, annotation_dir=None, csv_path=None):
    """Evaluate each in-house image separately, in inference mode, unaugmented."""
    image_ids = list(image_ids)
    predictions, truths = [], []
    for image_id in image_ids:
        loader = InHouseDataset(
            root or config.ft_root, [image_id], config, batch_size=1,
            training=False, shuffle=False, image_dir=image_dir,
            annotation_dir=annotation_dir,
        )
        images, targets = loader[0]
        outputs = model(images, training=False)
        coords = np.asarray(outputs["coords"])
        if not np.isfinite(coords).all():
            raise FloatingPointError(f"Non-finite coordinate output for image ID {image_id}")
        predictions.append(coords[0])
        truths.append(np.asarray(targets["landmarks"])[0])
    table = rank_per_image_predictions(image_ids, predictions, truths, config)
    if csv_path is not None:
        csv_path = Path(csv_path).expanduser()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(csv_path, index=False)
    return table


if __name__ == "__main__":
    import sys
    try:
        from .config import make_stage1_initial_config
    except ImportError:
        from config import make_stage1_initial_config
    config = make_stage1_initial_config()
    weights = sys.argv[1] if len(sys.argv) > 1 else os.path.join(config.checkpoint_dir, config.output_filename)
    evaluate_model(config, weights)


__all__ = [
    "evaluate_model", "load_compatible_weights", "rank_inhouse_images",
    "rank_per_image_predictions", "validate_checkpoint_compatibility",
]
