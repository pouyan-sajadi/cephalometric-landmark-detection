"""Shared image, annotation, and batch preprocessing.

Every operation that depends on experiment dimensions receives an explicit
``config``.
"""
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import Sequence

try:
    from .augmentation import CephalometricAugmentation
except ImportError:
    from augmentation import CephalometricAugmentation


IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def read_points(path, config):
    """Read comma- or whitespace-separated ``x,y`` points."""
    points = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            values = re.split(r"[,\s]+", line)
            if len(values) < 2:
                raise ValueError(f"Could not parse landmark line {line!r} in {path}")
            points.append([float(values[0]), float(values[1])])
    result = np.asarray(points, dtype=np.float32)
    expected = (config.num_landmarks, 2)
    if result.shape != expected:
        raise ValueError(f"Expected {expected[0]} landmarks in {path}; got {result.shape}")
    return result


def normalize_image(image):
    return ((image.astype(np.float32) / 255.0) - IMAGE_MEAN) / IMAGE_STD


def generate_heatmaps(landmarks_norm, config, integer_centres=False):
    """Generate Gaussian targets using the configured coordinate convention.

    V2 aligns normalised endpoints with the inclusive decoder grid, so 0 maps
    to pixel 0 and 1 maps exactly to pixel ``size - 1``.
    """
    landmarks_norm = np.asarray(landmarks_norm, dtype=np.float32)
    if landmarks_norm.shape != (config.num_landmarks, 2):
        raise ValueError(f"Expected {(config.num_landmarks, 2)} landmarks; got {landmarks_norm.shape}")
    heatmaps = np.zeros((config.heatmap_height, config.heatmap_width, config.num_landmarks), dtype=np.float32)
    y_grid, x_grid = np.mgrid[:config.heatmap_height, :config.heatmap_width]
    is_v2 = getattr(config, "objective_version", "legacy").lower() == "v2"
    width_scale = config.heatmap_width - 1 if is_v2 else config.heatmap_width
    height_scale = config.heatmap_height - 1 if is_v2 else config.heatmap_height
    for index, (x_norm, y_norm) in enumerate(landmarks_norm):
        x = np.clip(x_norm, 0.0, 1.0) * width_scale
        y = np.clip(y_norm, 0.0, 1.0) * height_scale
        if integer_centres:
            x = np.clip(int(x), 0, config.heatmap_width - 1)
            y = np.clip(int(y), 0, config.heatmap_height - 1)
        heatmaps[:, :, index] = np.exp(
            -((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2.0 * config.heatmap_sigma ** 2)
        )
    return heatmaps


def preprocess(image, landmarks, config, augmentation=None):
    original_h, original_w = image.shape[:2]
    image = cv2.resize(image, (config.input_width, config.input_height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    landmarks = np.asarray(landmarks, dtype=np.float32).copy()
    landmarks[:, 0] *= config.input_width / original_w
    landmarks[:, 1] *= config.input_height / original_h
    if augmentation is not None:
        image, landmarks = augmentation(tf.constant(image), tf.constant(landmarks))
        image, landmarks = image.numpy(), landmarks.numpy()
    landmarks_norm = np.stack(
        [landmarks[:, 0] / config.input_width, landmarks[:, 1] / config.input_height], axis=-1
    ).astype(np.float32)
    return normalize_image(image), landmarks_norm


def _image_files(directory):
    extensions = {".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".png"}
    return sorted(path for path in Path(directory).rglob("*") if path.is_file() and path.suffix.lower() in extensions)


def collect_isbi_image_paths(root_path, subdirectories=None):
    """Collect ISBI images from explicitly selected dataset subdirectories.

    Callers that need only the original ISBI train/test split can use
    ``ISBIDataset`` directly. This helper is retained for compatibility
    with paths-based dataset construction.
    """
    root = Path(root_path)
    subdirectories = subdirectories or (
        "Dataset/Training", "Dataset/Testing/Test1", "Dataset/Testing/Test2"
    )
    image_paths = []
    for relative_dir in subdirectories:
        directory = root / relative_dir
        if not directory.is_dir():
            raise FileNotFoundError(f"ISBI image directory not found: {directory}")
        image_paths.extend(_image_files(directory))
    image_paths = sorted(set(image_paths))
    if not image_paths:
        raise ValueError(f"No ISBI images found below {root}")
    return image_paths


class ISBIPathDataset:
    """ISBI samples backed by an explicit image-path list.

    This is used when a stage deliberately combines multiple ISBI folders.
    Senior and junior annotations are averaged exactly as in ``ISBIDataset``.
    """
    def __init__(self, image_paths, config, annotation_root):
        self.image_paths = [Path(path) for path in image_paths]
        self.config = config
        self.senior = Path(annotation_root) / "Senior Orthodontist"
        self.junior = Path(annotation_root) / "Junior Orthodontist"
        missing = []
        for image_path in self.image_paths:
            annotation_name = image_path.stem + ".txt"
            if not (self.senior / annotation_name).is_file() or not (self.junior / annotation_name).is_file():
                missing.append(annotation_name)
        if missing:
            raise FileNotFoundError(f"Missing ISBI senior/junior annotations, including {missing[:3]}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        annotation_name = image_path.stem + ".txt"
        image = read_rgb(image_path)
        senior = read_points(self.senior / annotation_name, self.config)
        junior = read_points(self.junior / annotation_name, self.config)
        return image, (senior + junior) / 2.0


class ISBIDataset:
    def __init__(self, root_path, config, mode="train"):
        self.root = Path(root_path)
        self.config = config
        if mode not in {"train", "test", "test1", "test2", "valid"}:
            raise ValueError(f"Invalid mode: {mode}")
        image_root = self.root / "Dataset" / ("Training" if mode == "train" else "Testing")
        if not image_root.is_dir():
            raise FileNotFoundError(f"ISBI image directory not found: {image_root}")
        self.image_root = image_root
        self.image_files = _image_files(image_root)
        self.senior = self.root / "Annotations" / "Senior Orthodontist"
        self.junior = self.root / "Annotations" / "Junior Orthodontist"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        annotation_name = image_path.stem + ".txt"
        image = read_rgb(image_path)
        senior = read_points(self.senior / annotation_name, self.config)
        junior = read_points(self.junior / annotation_name, self.config)
        return image, (senior + junior) / 2.0


class CephalometricDataLoader(Sequence):
    def __init__(self, dataset, config, batch_size=None, shuffle=True, augmentation=None):
        self.dataset = dataset
        self.config = config
        self.batch_size = config.batch_size if batch_size is None else batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.indices = np.arange(len(dataset))
        self.on_epoch_end()

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, batch_index):
        start = batch_index * self.batch_size
        batch_indices = self.indices[start:start + self.batch_size]
        if len(batch_indices) != self.batch_size:
            raise IndexError("Requested an incomplete batch")
        images, landmarks = [], []
        for index in batch_indices:
            image, points = self.dataset[index]
            image, points = preprocess(image, points, self.config, self.augmentation)
            images.append(image)
            landmarks.append(points)
        landmarks = np.asarray(landmarks, dtype=np.float32)
        return np.asarray(images, dtype=np.float32), {
            "heatmaps": np.asarray([generate_heatmaps(item, self.config) for item in landmarks], dtype=np.float32),
            "landmarks": landmarks,
        }


class InHouseDataset(Sequence):
    """In-house loader supporting local ``images/annotations`` and Kaggle v2 dirs."""
    def __init__(self, root, image_ids, config, batch_size=None, training=False, shuffle=False,
                 image_dir=None, annotation_dir=None):
        self.root = Path(root) if root is not None else None
        self.config = config
        self.image_ids = list(image_ids)
        self.batch_size = config.batch_size if batch_size is None else batch_size
        self.training = training
        self.shuffle = shuffle
        self.image_dir = Path(image_dir or (self.root / "images"))
        self.annotation_dir = Path(annotation_dir or (self.root / "annotations"))
        self.augmentation = CephalometricAugmentation(config) if training and config.augmentation_enabled else None
        self.on_epoch_end()

    def __len__(self):
        return len(self.image_ids) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_ids)

    def __getitem__(self, batch_index):
        ids = self.image_ids[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        if len(ids) != self.batch_size:
            raise IndexError("Requested an incomplete batch")
        images, landmarks = [], []
        for image_id in ids:
            image_path = self.image_dir / f"({image_id}).tif"
            annotation_path = self.annotation_dir / f"({image_id}).txt"
            if not image_path.exists():
                # Some exports omit parentheses; support it without changing the recorded IDs.
                candidates = list(self.image_dir.glob(f"*{image_id}*"))
                if candidates:
                    image_path = candidates[0]
            if not annotation_path.exists():
                candidates = list(self.annotation_dir.glob(f"*{image_id}*"))
                if candidates:
                    annotation_path = candidates[0]
            image = read_rgb(image_path)
            points = read_points(annotation_path, self.config)
            image, points = preprocess(image, points, self.config, self.augmentation)
            images.append(image)
            landmarks.append(points)
        landmarks = np.asarray(landmarks, dtype=np.float32)
        return np.asarray(images, dtype=np.float32), {
            "heatmaps": np.asarray([generate_heatmaps(item, self.config) for item in landmarks], dtype=np.float32),
            "landmarks": landmarks,
        }


def make_batched_loader(dataset, config, batch_size=None, shuffle=False, augmentation=None):
    """Create the standard complete-batch loader for an explicit dataset."""
    return CephalometricDataLoader(
        dataset, config, batch_size=batch_size, shuffle=shuffle, augmentation=augmentation
    )


__all__ = [
    "CephalometricDataLoader", "IMAGE_MEAN", "IMAGE_STD", "ISBIDataset", "ISBIPathDataset",
    "make_batched_loader",
    "InHouseDataset", "collect_isbi_image_paths", "generate_heatmaps", "normalize_image",
    "preprocess", "read_points", "read_rgb",
]
