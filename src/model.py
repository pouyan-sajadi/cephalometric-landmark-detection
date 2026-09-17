"""Checkpoint-compatible ResNet50 landmark model and decoding policies."""
import tensorflow as tf
from tensorflow.keras import Model, layers


def spatial_softmax(logits, temperature=1.0):
    """Normalize every landmark channel over its HxW spatial support."""
    temperature = tf.cast(temperature, logits.dtype)
    tf.debugging.assert_positive(temperature, message="spatial_softmax_temperature must be positive")
    shape = tf.shape(logits)
    # [B,H,W,L] -> [B,L,H*W], so every landmark is normalized independently.
    flat = tf.reshape(tf.transpose(logits, [0, 3, 1, 2]), [shape[0], shape[3], -1])
    probabilities = tf.nn.softmax(flat / temperature, axis=-1)
    return tf.transpose(tf.reshape(probabilities, [shape[0], shape[3], shape[1], shape[2]]), [0, 2, 3, 1])


def soft_argmax_coordinates(probabilities):
    """Decode spatial probabilities on an inclusive normalized [0,1] grid."""
    height = tf.shape(probabilities)[1]
    width = tf.shape(probabilities)[2]
    grid_x = tf.reshape(tf.linspace(tf.cast(0.0, probabilities.dtype), tf.cast(1.0, probabilities.dtype), width), [1, 1, -1, 1])
    grid_y = tf.reshape(tf.linspace(tf.cast(0.0, probabilities.dtype), tf.cast(1.0, probabilities.dtype), height), [1, -1, 1, 1])
    cx = tf.reduce_sum(probabilities * grid_x, axis=[1, 2])
    cy = tf.reduce_sum(probabilities * grid_y, axis=[1, 2])
    return tf.stack([cx, cy], axis=-1)


def legacy_heatmaps(logits):
    """Exact legacy Stage 1 output transform."""
    return tf.sigmoid(tf.clip_by_value(logits, -6.0, 6.0))


def legacy_center_of_mass(heatmaps):
    """Legacy COM decoder on the same inclusive grid used historically."""
    height = tf.shape(heatmaps)[1]
    width = tf.shape(heatmaps)[2]
    grid_x = tf.reshape(tf.linspace(tf.cast(0.0, heatmaps.dtype), tf.cast(1.0, heatmaps.dtype), width), [1, 1, -1, 1])
    grid_y = tf.reshape(tf.linspace(tf.cast(0.0, heatmaps.dtype), tf.cast(1.0, heatmaps.dtype), height), [1, -1, 1, 1])
    weights = heatmaps / (tf.reduce_sum(heatmaps, axis=[1, 2], keepdims=True) + tf.cast(1e-6, heatmaps.dtype))
    cx = tf.reduce_sum(weights * grid_x, axis=[1, 2])
    cy = tf.reduce_sum(weights * grid_y, axis=[1, 2])
    return tf.stack([cx, cy], axis=-1)


def build_model(config, decoder_filters=None, dropout_rate=None, layers_per_block=None):
    """Build the weighted-layer-compatible legacy or v2 model.

    The ResNet, decoder, head, and ``logits`` weighted layers retain their
    topology and names.  Only unweighted Lambda transforms after ``logits``
    differ, allowing Stage 1 H5 weights to load strictly into a v2 graph.
    """
    decoder_filters = config.decoder_filters if decoder_filters is None else decoder_filters
    dropout_rate = config.decoder_dropout_rate if dropout_rate is None else dropout_rate
    layers_per_block = config.decoder_layers_per_block if layers_per_block is None else layers_per_block
    if len(decoder_filters) != 3:
        raise ValueError("decoder_filters must contain three decoder widths")
    if layers_per_block not in (2, 3):
        raise ValueError("layers_per_block must be 2 or 3")

    inputs = layers.Input(shape=(config.input_height, config.input_width, 3), name="input_image")
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights=getattr(config, "backbone_weights", "imagenet"),
        input_tensor=inputs,
    )
    c2 = base.get_layer("conv2_block3_out").output
    c3 = base.get_layer("conv3_block4_out").output
    c4 = base.get_layer("conv4_block6_out").output
    c5 = base.get_layer("conv5_block3_out").output
    encoder_layer_names = {layer.name for layer in base.layers}

    def _conv(x, filters, name):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv")(x)
        x = layers.GroupNormalization(groups=min(8, filters), epsilon=1e-5, name=f"{name}_gn")(x)
        return layers.ReLU(name=f"{name}_relu")(x)

    def _decoder(x, skip, filters, name):
        x = layers.UpSampling2D(2, interpolation="bilinear", name=f"{name}_up")(x)
        x = layers.Concatenate(name=f"{name}_cat")([x, skip])
        if dropout_rate > 0.0:
            x = layers.SpatialDropout2D(dropout_rate, name=f"{name}_drop")(x)
        x = _conv(x, filters, f"{name}_a")
        x = _conv(x, filters, f"{name}_b")
        if layers_per_block == 3:
            x = _conv(x, filters, f"{name}_c")
        return x

    x = _decoder(c5, c4, decoder_filters[0], "dec4")
    x = _decoder(x, c3, decoder_filters[1], "dec3")
    x = _decoder(x, c2, decoder_filters[2], "dec2")
    x = _conv(x, 64, "head")
    # Do not rename or reshape this layer: it is the Stage 1 checkpoint boundary.
    logits = layers.Conv2D(config.num_landmarks, 1, padding="same", name="logits")(x)

    objective_version = getattr(config, "objective_version", "legacy").lower()
    decoding_version = getattr(config, "decoding_version", "legacy").lower()
    if objective_version == "legacy" and decoding_version == "legacy":
        heatmaps = layers.Lambda(legacy_heatmaps, name="heatmaps")(logits)
        coords = layers.Lambda(legacy_center_of_mass, name="coords")(heatmaps)
    elif objective_version == "v2" and decoding_version in {"v2", "spatial_softmax", "soft_argmax"}:
        temperature = float(getattr(config, "spatial_softmax_temperature", 1.0))
        heatmaps = layers.Lambda(
            lambda tensor: spatial_softmax(tensor, temperature), name="heatmaps"
        )(logits)
        coords = layers.Lambda(soft_argmax_coordinates, name="coords")(heatmaps)
    else:
        raise ValueError(
            f"Unsupported objective/decoding combination: {objective_version!r}/{decoding_version!r}"
        )

    return Model(
        inputs=inputs,
        outputs={"heatmaps": heatmaps, "coords": coords},
        name="CephalometricNet",
    ), encoder_layer_names


def apply_encoder_policy(model, encoder_layer_names, mode, freeze_batch_norm=False):
    """Apply encoder policy while always leaving decoder normalization alone."""
    if mode not in {"stage1", "unfrozen", "conv5_only"}:
        raise ValueError(f"Unknown encoder_mode: {mode}")
    for layer in model.layers:
        if layer.name not in encoder_layer_names:
            continue
        if mode == "stage1":
            trainable = False
        elif mode == "unfrozen":
            trainable = True
        else:
            trainable = "conv5" in layer.name
        if freeze_batch_norm and isinstance(layer, layers.BatchNormalization):
            trainable = False
        layer.trainable = trainable


def apply_trainable_scope(model, encoder_layer_names, scope):
    """Apply an explicit fine-tuning scope without changing legacy policies.

    ``dec2_head_logits`` freezes the full encoder and decoder blocks ``dec4``
    and ``dec3``. ``head_logits`` additionally freezes ``dec2``. Dropout has no
    variables, but follows its containing block's ``trainable`` flag for an
    auditable layer-level policy.
    """
    allowed = {
        "dec2_head_logits": ("dec2_", "head_", "logits"),
        "head_logits": ("head_", "logits"),
    }
    if scope not in allowed:
        raise ValueError(f"Unknown trainable_scope: {scope!r}; expected one of {sorted(allowed)}")
    prefixes = allowed[scope]
    for layer in model.layers:
        if layer.name in encoder_layer_names:
            layer.trainable = False
        else:
            layer.trainable = any(
                layer.name == prefix or layer.name.startswith(prefix)
                for prefix in prefixes
            )
    return parameter_counts(model)


def parameter_counts(model):
    """Return total/trainable/non-trainable scalar parameter counts."""
    trainable = int(sum(tf.keras.backend.count_params(value) for value in model.trainable_weights))
    non_trainable = int(sum(tf.keras.backend.count_params(value) for value in model.non_trainable_weights))
    return {
        "total": trainable + non_trainable,
        "trainable": trainable,
        "non_trainable": non_trainable,
    }


def freeze_encoder(model, encoder_layer_names, unfreeze_conv5=False, freeze_batch_norm=False):
    """Backward-compatible encoder freeze helper."""
    apply_encoder_policy(
        model,
        encoder_layer_names,
        "conv5_only" if unfreeze_conv5 else "stage1",
        freeze_batch_norm=freeze_batch_norm,
    )


def unfreeze_encoder(model, encoder_layer_names, freeze_batch_norm=False):
    """Unfreeze encoder convolutions, optionally keeping encoder BN frozen."""
    apply_encoder_policy(model, encoder_layer_names, "unfrozen", freeze_batch_norm=freeze_batch_norm)


__all__ = [
    "apply_encoder_policy", "apply_trainable_scope", "build_model", "freeze_encoder",
    "legacy_center_of_mass", "legacy_heatmaps", "parameter_counts",
    "soft_argmax_coordinates", "spatial_softmax", "unfreeze_encoder",
]
