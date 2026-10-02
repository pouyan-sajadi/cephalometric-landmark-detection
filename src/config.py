"""Explicit experiment configuration and two-stage protocol factories.

Both training stages use the stable v2 spatial-softmax objective. Stage 1
(pre-training) starts from ImageNet encoder weights and learns general
cephalometric representations from the public ISBI dataset. Stage 2
(fine-tuning) adapts the resulting pre-trained model to the in-house domain
with a deliberately small trainable scope. No mutable module-level
configuration object is used.
"""
from dataclasses import asdict, dataclass, field, replace
from numbers import Integral
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


FINETUNE_TRAIN_IDS: Tuple[int, ...] = (
    25, 80, 71, 46, 26, 41, 33, 72, 32, 24,
    36, 31, 47, 50, 27, 76, 38, 6, 22, 69,
)
FINETUNE_VAL_IDS: Tuple[int, ...] = (
    39, 52, 1, 75, 67, 13, 17, 37, 60, 44,
    58, 29, 43, 48, 59, 61, 18, 65, 54, 57,
    5, 45, 2, 70, 28, 23, 16, 15, 8, 21,
    62, 19, 11, 34, 78, 20, 14, 3, 10, 68,
    9, 55, 49, 42, 40, 64, 7, 63, 30, 53,
    66, 51, 35, 74, 79, 73, 77, 56, 4, 12,
)


def _env_path(name: str, relative_default: str) -> str:
    value = os.getenv(name)
    return str(Path(value).expanduser()) if value else str(REPOSITORY_ROOT / relative_default)


def _pretrained_weights_path() -> str:
    """Resolve the Stage 1 pre-trained checkpoint used by Stage 2 fine-tuning."""
    return _env_path("CEPHALO_PRETRAINED_WEIGHTS", "checkpoints/best_model_fixed.weights.h5")


def _output_path(name: str, relative_default: str) -> str:
    value = os.getenv(name)
    if value:
        return str(Path(value).expanduser())
    kaggle_working = Path("/kaggle/working")
    return str(kaggle_working if kaggle_working.is_dir() else REPOSITORY_ROOT / relative_default)


def resolve_data_dir(root: os.PathLike | str, relative: os.PathLike | str) -> str:
    """Resolve a data directory across local and nested Kaggle layouts."""
    root_path = Path(root).expanduser()
    relative_path = Path(relative)
    exact = root_path / relative_path
    if exact.is_dir():
        return str(exact)
    if root_path.is_dir():
        matches = sorted(p for p in root_path.rglob(relative_path.name) if p.is_dir())
        if matches:
            return str(matches[0])
    return str(exact)


def validate_finetune_split(train_ids, val_ids, reference_train20=None, require_val_count=60):
    """Validate an intended 10/60 or 20/60 in-house fine-tuning split.

    IDs must be unique built-in/NumPy integer values in 1..80. A 20/60 split
    must account for all images. A 10/60 split reports the other ten as unused
    and, when ``reference_train20`` is supplied, must be nested within it.
    """
    train_ids, val_ids = list(train_ids), list(val_ids)
    for name, values in (("train_ids", train_ids), ("val_ids", val_ids)):
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
            raise TypeError(f"{name} must contain only integer IDs")
        values[:] = [int(value) for value in values]
        if len(values) != len(set(values)):
            raise ValueError(f"{name} contains duplicate IDs")
        invalid = sorted(value for value in values if value < 1 or value > 80)
        if invalid:
            raise ValueError(f"{name} contains IDs outside 1..80: {invalid}")
    if len(train_ids) not in (10, 20):
        raise ValueError(f"Fine-tuning requires 10 or 20 training IDs; got {len(train_ids)}")
    if len(val_ids) != require_val_count:
        raise ValueError(f"Fine-tuning requires {require_val_count} validation IDs; got {len(val_ids)}")
    overlap = sorted(set(train_ids) & set(val_ids))
    if overlap:
        raise ValueError(f"Training and validation IDs overlap: {overlap}")
    accounted = set(train_ids) | set(val_ids)
    all_ids = set(range(1, 81))
    if len(train_ids) == 20 and accounted != all_ids:
        raise ValueError("A 20/60 split must account for every ID in 1..80")
    if reference_train20 is not None:
        reference = list(reference_train20)
        if (len(reference) != 20 or len(set(reference)) != 20
                or any(isinstance(value, bool) or not isinstance(value, Integral) or value < 1 or value > 80
                       for value in reference)):
            raise ValueError("reference_train20 must contain 20 unique integer IDs in 1..80")
        if not set(train_ids).issubset(reference):
            missing = sorted(set(train_ids) - set(reference))
            raise ValueError(f"10-image training IDs must be a subset of reference_train20; outside IDs: {missing}")
    return {
        "train_ids": tuple(train_ids),
        "val_ids": tuple(val_ids),
        "unused_ids": tuple(sorted(all_ids - accounted)),
        "sample_count": len(train_ids),
    }


def _experiment_slug(experiment_name, sample_count):
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(experiment_name)).strip("_")
    marker = f"n{sample_count}"
    if marker not in slug.lower():
        slug = f"{slug}_{marker}"
    return slug or f"finetune_{marker}"


def make_finetune_experiment_config(train_ids, val_ids, experiment_name="finetune_n20",
                                    trainable_scope="dec2_head_logits", reference_train20=None,
                                    **overrides):
    """Create validated fine-tuning config with non-overwriting artifact names."""
    split = validate_finetune_split(train_ids, val_ids, reference_train20=reference_train20)
    slug = _experiment_slug(experiment_name, split["sample_count"])
    values = dict(
        experiment_name=slug,
        trainable_scope=trainable_scope,
        train_image_ids=split["train_ids"],
        val_image_ids=split["val_ids"],
        unused_image_ids=split["unused_ids"],
        output_filename=f"best_model_finetune_v3_{slug}_best_loss.weights.h5",
        best_mre_output_filename=f"best_model_finetune_v3_{slug}_best_mre.weights.h5",
        history_filename=f"finetune_v3_{slug}_training_history.json",
    )
    values.update(overrides)
    return make_stage2_finetune_config(**values)


def _landmark_names():
    return {
        str(i): name for i, name in enumerate([
            "Sella", "Nasion", "Orbitale", "Porion", "Subspinale",
            "Supramentale", "Pogonion", "Menton", "Gnathion", "Gonion",
            "Lower incisal incision", "Upper incisal incision", "Upper lip",
            "Lower lip", "Subnasale", "Soft tissue pogonion",
            "Posterior nasal spine", "Anterior nasal spine", "Articulare",
        ])
    }


@dataclass(frozen=True)
class Config:
    """Schema for one independent training/evaluation stage.

    ``objective_version='legacy'`` remains available for archived checkpoints
    with sigmoid heatmaps, MSE, L1 coordinates, and the historical heatmap
    centre convention. Version ``v2``—used by all active stage factories—uses
    spatial probabilities, KL, smooth-L1, and inclusive grid alignment.
    """

    stage_name: str = "custom"
    model_version: str = "legacy"
    objective_version: str = "legacy"
    decoding_version: str = "legacy"
    seed: int = 42

    dataset_root: str = field(default_factory=lambda: _env_path("CEPHALO_ISBI_ROOT", "data/isbi/ISBI Dataset"))
    inhouse_root: str = field(default_factory=lambda: _env_path("CEPHALO_INHOUSE_ROOT", "data/inhouse"))
    ft_root: str = field(default_factory=lambda: _env_path("CEPHALO_FT_ROOT", "data/inhouse"))
    ft_images: Optional[str] = None
    ft_annotations: Optional[str] = None

    num_landmarks: int = 19
    input_height: int = 512
    input_width: int = 416
    heatmap_height: int = 128
    heatmap_width: int = 104
    heatmap_sigma: float = 2.0
    spatial_softmax_temperature: float = 1.0
    smooth_l1_delta: float = 0.01

    batch_size: int = 4
    epochs: int = 150
    learning_rate: float = 9.78e-5
    warmup_learning_rate: float = 1e-4
    warmup_epochs: int = 0
    coord_loss_weight: float = 95.3
    heatmap_loss_weight: float = 1.0
    coord_loss: str = "l1"
    heatmap_loss: str = "mse"

    decoder_filters: Tuple[int, int, int] = (1024, 512, 256)
    decoder_dropout_rate: float = 0.0
    decoder_layers_per_block: int = 2
    backbone_weights: Optional[str] = "imagenet"
    encoder_mode: str = "unfrozen"  # stage1, unfrozen, or conv5_only
    freeze_encoder_batch_norm: bool = False
    # None preserves the stage's encoder policy and leaves the decoder trainable.
    # Stage 2 fine-tuning uses dec2/head/logits or only head/logits.
    trainable_scope: Optional[str] = None

    optimizer: str = "adam"
    weight_decay: float = 0.0
    exclude_bias_and_norm_from_weight_decay: bool = True
    gradient_clip_norm: float = 5.0
    lr_plateau_factor: float = 0.5
    lr_plateau_patience: int = 10
    lr_plateau_min_delta: float = 0.0
    lr_scheduler_monitor: str = "val_mre"
    min_learning_rate: float = 1e-7
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    early_stopping_monitor: str = "val_mre"
    checkpoint_monitor: str = "val_mre"
    monitor_mode: str = "min"
    restore_best_weights: bool = True
    checkpoint_interval: int = 10

    # Project reference convention; see documentation before clinical use.
    original_height: int = 2400
    original_width: int = 1935
    image_resolution: float = 0.1
    precision_thresholds: List[float] = field(default_factory=lambda: [2.0, 2.5, 3.0, 4.0])

    augmentation_enabled: bool = True
    horizontal_flip: bool = True
    horizontal_flip_prob: float = 0.5
    rotation_range: float = 10.0
    rotation_prob: float = 0.5
    scale_range: float = 0.1
    scale_prob: float = 0.5
    translation_range: float = 0.05
    translation_prob: float = 0.5
    brightness_range: float = 0.2
    brightness_prob: float = 0.3
    contrast_range: float = 0.2
    contrast_prob: float = 0.3
    gaussian_noise_std: float = 0.02
    noise_prob: float = 0.2
    out_of_frame_landmark_policy: str = "clip"  # legacy-compatible clip or reject_transform

    checkpoint_dir: str = field(default_factory=lambda: _output_path("CEPHALO_CHECKPOINT_DIR", "checkpoints"))
    log_dir: str = field(default_factory=lambda: _output_path("CEPHALO_LOG_DIR", "results"))
    output_filename: str = "best_model.weights.h5"
    best_mre_output_filename: Optional[str] = None
    history_filename: Optional[str] = None
    pretrained_weights: Optional[str] = None
    initialization_source: str = "none"
    experiment_name: Optional[str] = None
    train_image_ids: Tuple[int, ...] = ()
    val_image_ids: Tuple[int, ...] = ()
    unused_image_ids: Tuple[int, ...] = ()
    anatomical_landmarks: dict = field(default_factory=_landmark_names)

    @classmethod
    def from_env(cls, **overrides):
        config = cls()
        if config.ft_images is None:
            config = replace(
                config,
                ft_images=resolve_data_dir(config.ft_root, os.path.join("images_v2", "images_v2")),
                ft_annotations=resolve_data_dir(config.ft_root, os.path.join("annotations_v2", "annotations_v2")),
            )
        return replace(config, **overrides) if overrides else config

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-safe configuration snapshot for run metadata."""
        return asdict(self)


def _stage_config(**values) -> Config:
    return replace(Config.from_env(), **values)


def make_stage1_initial_config(**overrides) -> Config:
    """Stable-v2 Stage 1 — public-data pre-training with validation-loss model selection.

    The weighted architecture is retained unchanged between stages, using the
    same spatial-softmax/KL/smooth-L1 formulation as Stage 2 fine-tuning.
    """
    values = dict(
        stage_name="stage1_initial_v2",
        model_version="v2",
        objective_version="v2",
        decoding_version="spatial_softmax",
        decoder_filters=(1024, 512, 256),
        decoder_dropout_rate=0.10,
        epochs=150,
        batch_size=4,
        learning_rate=3e-5,
        warmup_learning_rate=1e-4,
        warmup_epochs=5,
        coord_loss_weight=10.0,
        heatmap_loss_weight=0.1,
        coord_loss="smooth_l1",
        heatmap_loss="spatial_kl",
        encoder_mode="stage1",
        freeze_encoder_batch_norm=True,
        optimizer="adamw",
        weight_decay=1e-5,
        gradient_clip_norm=2.0,
        lr_plateau_factor=0.5,
        lr_plateau_patience=8,
        lr_plateau_min_delta=2e-4,
        lr_scheduler_monitor="val_loss",
        min_learning_rate=1e-7,
        early_stopping_patience=20,
        early_stopping_min_delta=2e-4,
        early_stopping_monitor="val_loss",
        checkpoint_monitor="val_loss",
        output_filename="best_model_fixed.weights.h5",
        best_mre_output_filename="best_model_stage1_v2_best_mre.weights.h5",
        history_filename="stage1_initial_v2_training_history.json",
        pretrained_weights=None,
        initialization_source="imagenet",
    )
    values.update(overrides)
    return _stage_config(**values)


def make_stage2_finetune_config(**overrides) -> Config:
    """Anti-overfitting Stage 2 fine-tuning initialized from the Stage 1 pre-trained model."""
    values = dict(
        stage_name="stage2_finetune_v3",
        model_version="v2",
        objective_version="v2",
        decoding_version="spatial_softmax",
        ft_root=os.getenv("CEPHALO_FT_ROOT", "/kaggle/input/fine-tuning-dataset-final")
        if Path("/kaggle").is_dir() else os.getenv("CEPHALO_FT_ROOT", str(REPOSITORY_ROOT / "data/inhouse")),
        decoder_filters=(1024, 512, 256),
        decoder_dropout_rate=0.20,
        epochs=50,
        batch_size=2,
        learning_rate=5e-6,
        warmup_epochs=0,
        coord_loss_weight=10.0,
        heatmap_loss_weight=0.1,
        coord_loss="smooth_l1",
        heatmap_loss="spatial_kl",
        encoder_mode="stage1",
        freeze_encoder_batch_norm=True,
        trainable_scope="dec2_head_logits",
        optimizer="adamw",
        weight_decay=5e-5,
        gradient_clip_norm=1.0,
        lr_plateau_factor=0.5,
        lr_plateau_patience=2,
        lr_plateau_min_delta=2e-4,
        lr_scheduler_monitor="val_loss",
        min_learning_rate=1e-7,
        early_stopping_patience=5,
        early_stopping_min_delta=2e-4,
        early_stopping_monitor="val_loss",
        checkpoint_monitor="val_loss",
        horizontal_flip=False,
        horizontal_flip_prob=0.0,
        rotation_range=5.0,
        rotation_prob=0.35,
        scale_range=0.05,
        scale_prob=0.35,
        translation_range=0.025,
        translation_prob=0.35,
        brightness_range=0.10,
        brightness_prob=0.25,
        contrast_range=0.10,
        contrast_prob=0.25,
        gaussian_noise_std=0.01,
        noise_prob=0.15,
        image_resolution=0.08,
        out_of_frame_landmark_policy="reject_transform",
        output_filename="best_model_finetune_v3_n20_best_loss.weights.h5",
        best_mre_output_filename="best_model_finetune_v3_n20_best_mre.weights.h5",
        history_filename="finetune_v3_n20_training_history.json",
        pretrained_weights=_pretrained_weights_path(),
        initialization_source="stage1_fixed",
        experiment_name="finetune_n20",
        train_image_ids=FINETUNE_TRAIN_IDS,
        val_image_ids=FINETUNE_VAL_IDS,
    )
    values.update(overrides)
    config = _stage_config(**values)
    return replace(
        config,
        ft_images=resolve_data_dir(config.ft_root, os.path.join("images_v2", "images_v2")),
        ft_annotations=resolve_data_dir(config.ft_root, os.path.join("annotations_v2", "annotations_v2")),
    )


stage1 = make_stage1_initial_config
stage2 = make_stage2_finetune_config


__all__ = [
    "Config", "REPOSITORY_ROOT", "FINETUNE_TRAIN_IDS", "FINETUNE_VAL_IDS",
    "resolve_data_dir", "validate_finetune_split", "make_stage1_initial_config",
    "make_stage2_finetune_config", "make_finetune_experiment_config",
    "stage1", "stage2",
]
