"""Synchronized stochastic image/landmark augmentation."""
import math
import tensorflow as tf


class CephalometricAugmentation:
    def __init__(self, config):
        """Create augmentation from the explicitly supplied stage config."""
        self.config = config

    def __call__(self, image, landmarks):
        original_image, original_landmarks = image, landmarks
        image, landmarks = self._random_horizontal_flip(image, landmarks)
        image, landmarks = self._random_rotation(image, landmarks)
        image, landmarks = self._random_scale(image, landmarks)
        image, landmarks = self._random_translation(image, landmarks)
        h = tf.cast(tf.shape(image)[0], tf.float32)
        w = tf.cast(tf.shape(image)[1], tf.float32)
        policy = getattr(self.config, "out_of_frame_landmark_policy", "clip")
        if policy == "reject_transform":
            valid = tf.reduce_all(
                (landmarks[:, 0] >= 0.0) & (landmarks[:, 0] <= w - 1.0)
                & (landmarks[:, 1] >= 0.0) & (landmarks[:, 1] <= h - 1.0)
            )
            # Reject the whole geometric transform rather than silently moving
            # an out-of-frame target onto the image boundary.
            image, landmarks = tf.cond(
                valid,
                lambda: (image, landmarks),
                lambda: (original_image, original_landmarks),
            )
        elif policy == "clip":
            landmarks = tf.clip_by_value(landmarks, [0.0, 0.0], [w - 1.0, h - 1.0])
        else:
            raise ValueError(f"Unknown out_of_frame_landmark_policy: {policy}")
        image = self._random_brightness(image)
        image = self._random_contrast(image)
        image = self._random_gaussian_noise(image)
        return image, landmarks

    def _random_horizontal_flip(self, image, landmarks):
        if not self.config.horizontal_flip or tf.random.uniform([]) >= self.config.horizontal_flip_prob:
            return image, landmarks
        image = tf.image.flip_left_right(image)
        w = tf.cast(tf.shape(image)[1], tf.float32)
        return image, tf.stack([w - 1.0 - landmarks[:, 0], landmarks[:, 1]], axis=-1)

    def _random_rotation(self, image, landmarks):
        if tf.random.uniform([]) >= self.config.rotation_prob:
            return image, landmarks
        angle = tf.random.uniform([], -self.config.rotation_range * math.pi / 180.0, self.config.rotation_range * math.pi / 180.0)
        h = tf.cast(tf.shape(image)[0], tf.float32)
        w = tf.cast(tf.shape(image)[1], tf.float32)
        cx, cy = w / 2.0, h / 2.0
        image = self._rotate_image(image, angle)
        cos_a, sin_a = tf.cos(angle), tf.sin(angle)
        x, y = landmarks[:, 0] - cx, landmarks[:, 1] - cy
        return image, tf.stack([x * cos_a - y * sin_a + cx, x * sin_a + y * cos_a + cy], axis=-1)

    @staticmethod
    def _rotate_image(image, angle):
        image = tf.expand_dims(image, 0)
        cos_a, sin_a = tf.cos(angle), tf.sin(angle)
        h = tf.cast(tf.shape(image)[1], tf.float32)
        w = tf.cast(tf.shape(image)[2], tf.float32)
        cx, cy = w / 2.0, h / 2.0
        transform = tf.reshape(tf.stack([cos_a, sin_a, cx - cos_a * cx - sin_a * cy, -sin_a, cos_a, cy + sin_a * cx - cos_a * cy, 0.0, 0.0]), [1, 8])
        return tf.squeeze(tf.raw_ops.ImageProjectiveTransformV3(images=image, transforms=transform, output_shape=tf.shape(image)[1:3], interpolation="BILINEAR", fill_mode="NEAREST", fill_value=0.0), 0)

    def _random_scale(self, image, landmarks):
        if tf.random.uniform([]) >= self.config.scale_prob:
            return image, landmarks
        scale = tf.random.uniform([], 1.0 - self.config.scale_range, 1.0 + self.config.scale_range)
        h, w = tf.shape(image)[0], tf.shape(image)[1]
        image = tf.image.resize(image, [tf.cast(tf.cast(h, tf.float32) * scale, tf.int32), tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)])
        return self._crop_or_pad(image, landmarks * scale, h, w)

    @staticmethod
    def _crop_or_pad(image, landmarks, target_h, target_w):
        h, w = tf.shape(image)[0], tf.shape(image)[1]
        pad_h, pad_w = tf.maximum(target_h - h, 0), tf.maximum(target_w - w, 0)
        image = tf.pad(image, [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0, 0]])
        landmarks += tf.cast(tf.stack([pad_w // 2, pad_h // 2]), tf.float32)
        h, w = tf.shape(image)[0], tf.shape(image)[1]
        sh, sw = (h - target_h) // 2, (w - target_w) // 2
        return image[sh:sh + target_h, sw:sw + target_w], landmarks - tf.cast(tf.stack([sw, sh]), tf.float32)

    def _random_translation(self, image, landmarks):
        if tf.random.uniform([]) >= self.config.translation_prob:
            return image, landmarks
        h, w = tf.cast(tf.shape(image)[0], tf.float32), tf.cast(tf.shape(image)[1], tf.float32)
        dx = tf.random.uniform([], -w * self.config.translation_range, w * self.config.translation_range)
        dy = tf.random.uniform([], -h * self.config.translation_range, h * self.config.translation_range)
        transform = tf.reshape(tf.stack([1.0, 0.0, -dx, 0.0, 1.0, -dy, 0.0, 0.0]), [1, 8])
        image = tf.squeeze(tf.raw_ops.ImageProjectiveTransformV3(images=tf.expand_dims(image, 0), transforms=transform, output_shape=tf.shape(tf.expand_dims(image, 0))[1:3], interpolation="BILINEAR", fill_mode="NEAREST", fill_value=0.0), 0)
        return image, landmarks + tf.stack([dx, dy])

    def _random_brightness(self, image):
        if tf.random.uniform([]) < self.config.brightness_prob:
            image = tf.image.random_brightness(image, self.config.brightness_range * 255.0)
        return tf.clip_by_value(image, 0.0, 255.0)

    def _random_contrast(self, image):
        if tf.random.uniform([]) < self.config.contrast_prob:
            image = tf.image.random_contrast(image, 1.0 - self.config.contrast_range, 1.0 + self.config.contrast_range)
        return tf.clip_by_value(image, 0.0, 255.0)

    def _random_gaussian_noise(self, image):
        if tf.random.uniform([]) < self.config.noise_prob:
            image += tf.random.normal(tf.shape(image), 0.0, self.config.gaussian_noise_std * 255.0)
        return tf.clip_by_value(image, 0.0, 255.0)
