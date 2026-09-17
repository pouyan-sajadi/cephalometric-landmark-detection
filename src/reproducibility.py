"""Reproducibility helpers.

Call :func:`set_global_seed` before constructing datasets or models when a
repeatable experiment seed is desired. TensorFlow GPU kernels can still be
nondeterministic depending on the installed CUDA/cuDNN stack.
"""
import os
import random

import numpy as np


def set_global_seed(seed=42, enable_tensorflow_determinism=False):
    """Seed Python, NumPy, and TensorFlow random generators.

    TensorFlow is imported lazily so lightweight documentation and data tools
    do not need to import it merely to inspect this helper.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    import tensorflow as tf

    tf.random.set_seed(seed)
    if enable_tensorflow_determinism:
        try:
            tf.config.experimental.enable_op_determinism()
        except AttributeError:
            os.environ["TF_DETERMINISTIC_OPS"] = "1"


__all__ = ["set_global_seed"]
