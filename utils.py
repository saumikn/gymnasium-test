def import_tf():
    import os
    import logging
    import warnings

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_KERAS_BACKEND_DISABLE_WARNINGS"] = "1"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    # Specifically filter the VarianceScaling warning
    warnings.filterwarnings("ignore", message=".*VarianceScaling is unseeded.*")

    # # Monkey patch the warning method in Keras to silence this specific warning
    # import keras.src.initializers.initializers as keras_init

    # original_warning = keras_init.warnings.warn

    # def filtered_warning(message, *args, **kwargs):
    #     if "VarianceScaling is unseeded" not in str(message):
    #         original_warning(message, *args, **kwargs)

    # keras_init.warnings.warn = filtered_warning

    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    tf.autograph.set_verbosity(0)

    from tensorflow.python.util import deprecation

    deprecation._PRINT_DEPRECATION_WARNINGS = False
    os.environ["CUDA_CACHE_DISABLE"] = "1"
    return tf
