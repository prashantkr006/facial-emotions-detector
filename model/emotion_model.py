"""
Emotion Recognition CNN — Mini-Xception Architecture
=====================================================
A lightweight depthwise-separable CNN inspired by Xception, optimised for
48×48 grayscale facial-expression images (FER-2013).

Key design choices for high accuracy:
  • Depthwise-separable convolutions → fewer parameters, less overfitting
  • Residual connections → smoother gradient flow
  • Global Average Pooling → no dense hidden layers, acts as regulariser
  • Batch Normalisation + Dropout throughout
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers


def _depthwise_separable_block(x, filters, kernel_size=(3, 3)):
    """Depthwise-separable conv block with BN + ReLU."""
    x = layers.SeparableConv2D(
        filters, kernel_size, padding="same",
        depthwise_regularizer=regularizers.l2(1e-4),
        pointwise_regularizer=regularizers.l2(1e-4),
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def _residual_block(x, filters):
    """Two separable-conv layers + skip connection."""
    residual = layers.Conv2D(filters, (1, 1), strides=(2, 2),
                             padding="same", use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = _depthwise_separable_block(x, filters)
    x = _depthwise_separable_block(x, filters)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    x = layers.Add()([x, residual])
    return x


def build_mini_xception(input_shape=(48, 48, 1), num_classes=7):
    """
    Build the Mini-Xception emotion classifier.

    Parameters
    ----------
    input_shape : tuple  – (H, W, C) — default 48×48 grayscale
    num_classes : int     – number of emotion categories (default 7)

    Returns
    -------
    tf.keras.Model
    """
    inputs = layers.Input(shape=input_shape, name="face_input")

    # ── Entry flow ──────────────────────────────────────────────
    x = layers.Conv2D(8, (3, 3), strides=(1, 1), padding="same",
                      use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(8, (3, 3), strides=(1, 1), padding="same",
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # ── Middle flow — residual blocks with increasing filters ───
    for filters in [16, 32, 64, 128]:
        x = _residual_block(x, filters)
        x = layers.Dropout(0.25)(x)

    # ── Exit flow ───────────────────────────────────────────────
    x = layers.Conv2D(num_classes, (3, 3), padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Activation("softmax", name="emotion_output")(x)

    model = Model(inputs, outputs, name="mini_xception")
    return model


def build_emotion_model_v2(input_shape=(48, 48, 1), num_classes=7):
    """
    Alternative deeper VGG-style model for higher accuracy when
    compute budget is less constrained.

    Achieves ~73-75% on FER-2013 with proper augmentation.
    """
    inputs = layers.Input(shape=input_shape, name="face_input")

    # Block 1
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Classifier head
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           name="emotion_output")(x)

    model = Model(inputs, outputs, name="emotion_vgg_deep")
    return model


if __name__ == "__main__":
    # Quick sanity check
    model = build_mini_xception()
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")

    model_v2 = build_emotion_model_v2()
    model_v2.summary()
    print(f"\nTotal parameters (V2): {model_v2.count_params():,}")
