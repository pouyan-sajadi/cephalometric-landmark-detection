"""Configuration-driven legacy and stable-v2 landmark losses."""
import tensorflow as tf


_EPSILON = 1e-7


def compute_heatmap_loss(pred_heatmaps, true_heatmaps, loss_type="mse", epsilon=_EPSILON):
    """Compute legacy MSE or target-normalized spatial KL.

    For KL, each target landmark map is normalized over HxW. Predictions may
    already be v2 probabilities; they are normalized again defensively before
    stable log handling. The result is averaged over batch and landmarks.
    """
    loss_type = loss_type.lower()
    pred_heatmaps = tf.convert_to_tensor(pred_heatmaps)
    true_heatmaps = tf.cast(true_heatmaps, pred_heatmaps.dtype)
    if loss_type in {"mse", "legacy"}:
        return tf.reduce_mean(tf.square(pred_heatmaps - true_heatmaps))
    if loss_type not in {"spatial_kl", "kl", "kld"}:
        raise ValueError(f"Unknown heatmap loss: {loss_type}")

    eps = tf.cast(epsilon, pred_heatmaps.dtype)
    target_sum = tf.reduce_sum(true_heatmaps, axis=[1, 2], keepdims=True)
    target_prob = true_heatmaps / tf.maximum(target_sum, eps)
    pred_sum = tf.reduce_sum(pred_heatmaps, axis=[1, 2], keepdims=True)
    pred_prob = pred_heatmaps / tf.maximum(pred_sum, eps)
    log_target = tf.math.log(tf.maximum(target_prob, eps))
    log_pred = tf.math.log(tf.maximum(pred_prob, eps))
    per_landmark = tf.reduce_sum(target_prob * (log_target - log_pred), axis=[1, 2])
    return tf.reduce_mean(per_landmark)


def compute_coordinate_loss(pred_coords, true_coords, loss_type="l1", delta=0.01):
    """Compute legacy L1 or mean smooth-L1/Huber coordinate loss."""
    loss_type = loss_type.lower()
    pred_coords = tf.convert_to_tensor(pred_coords)
    true_coords = tf.cast(true_coords, pred_coords.dtype)
    error = pred_coords - true_coords
    if loss_type in {"l1", "mae", "legacy"}:
        return tf.reduce_mean(tf.abs(error))
    if loss_type not in {"smooth_l1", "huber"}:
        raise ValueError(f"Unknown coordinate loss: {loss_type}")
    delta_tensor = tf.cast(delta, pred_coords.dtype)
    tf.debugging.assert_positive(delta_tensor, message="smooth_l1_delta must be positive")
    absolute = tf.abs(error)
    element_loss = tf.where(
        absolute <= delta_tensor,
        0.5 * tf.square(error) / delta_tensor,
        absolute - 0.5 * delta_tensor,
    )
    return tf.reduce_mean(element_loss)


def compute_loss_components(pred_heatmaps, true_heatmaps, pred_coords, true_coords,
                            heatmap_weight=1.0, coord_weight=95.3,
                            heatmap_loss="mse", coord_loss="l1",
                            smooth_l1_delta=0.01):
    """Return raw and weighted components in a logging-friendly dictionary."""
    raw_heatmap = compute_heatmap_loss(pred_heatmaps, true_heatmaps, heatmap_loss)
    raw_coordinate = compute_coordinate_loss(pred_coords, true_coords, coord_loss, smooth_l1_delta)
    weighted_heatmap = tf.cast(heatmap_weight, raw_heatmap.dtype) * raw_heatmap
    weighted_coordinate = tf.cast(coord_weight, raw_coordinate.dtype) * raw_coordinate
    return {
        "total": weighted_heatmap + weighted_coordinate,
        "heatmap": raw_heatmap,
        "coordinate": raw_coordinate,
        "weighted_heatmap": weighted_heatmap,
        "weighted_coordinate": weighted_coordinate,
    }


def compute_total_loss(pred_heatmaps, true_heatmaps, pred_coords, true_coords,
                       heatmap_weight=1.0, coord_weight=95.3,
                       config=None, heatmap_loss=None, coord_loss=None,
                       smooth_l1_delta=None, return_components=False):
    """Compute combined loss while retaining the historical tuple API.

    Existing callers receive ``(total, raw_heatmap, raw_coordinate)``. New
    callers can request a component dictionary with ``return_components=True``.
    """
    if config is not None:
        heatmap_weight = config.heatmap_loss_weight
        coord_weight = config.coord_loss_weight
        heatmap_loss = config.heatmap_loss
        coord_loss = config.coord_loss
        smooth_l1_delta = config.smooth_l1_delta
    heatmap_loss = "mse" if heatmap_loss is None else heatmap_loss
    coord_loss = "l1" if coord_loss is None else coord_loss
    smooth_l1_delta = 0.01 if smooth_l1_delta is None else smooth_l1_delta
    components = compute_loss_components(
        pred_heatmaps, true_heatmaps, pred_coords, true_coords,
        heatmap_weight=heatmap_weight,
        coord_weight=coord_weight,
        heatmap_loss=heatmap_loss,
        coord_loss=coord_loss,
        smooth_l1_delta=smooth_l1_delta,
    )
    if return_components:
        return components
    return components["total"], components["heatmap"], components["coordinate"]


__all__ = [
    "compute_coordinate_loss", "compute_heatmap_loss", "compute_loss_components",
    "compute_total_loss",
]
