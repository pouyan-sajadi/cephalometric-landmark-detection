"""Metric and coordinate utilities with explicit reference parameters."""
import numpy as np


def decode_coordinates_to_original(norm_coords, original_width, original_height):
    return np.stack([norm_coords[..., 0] * original_width, norm_coords[..., 1] * original_height], axis=-1)


def compute_mre(pred_coords, true_coords, image_resolution):
    distances_mm = np.linalg.norm(pred_coords - true_coords, axis=-1) * image_resolution
    per_landmark = np.mean(distances_mm, axis=0) if distances_mm.ndim > 1 else distances_mm
    return float(np.mean(per_landmark)), per_landmark


def compute_sdr(pred_coords, true_coords, thresholds, image_resolution):
    distances_mm = np.linalg.norm(pred_coords - true_coords, axis=-1) * image_resolution
    return {float(t): float(np.mean(distances_mm < t) * 100.0) for t in thresholds}


def compute_reference_distances(pred_norm, true_norm, reference_width, reference_height, image_resolution):
    diagonal = np.sqrt(reference_width ** 2 + reference_height ** 2)
    return np.linalg.norm(pred_norm - true_norm, axis=-1) * diagonal * image_resolution
