"""
FER-2013 Data Loader
====================
Loads the FER-2013 CSV, applies aggressive augmentation to
maximise validation accuracy.

Dataset: https://www.kaggle.com/datasets/msambare/fer2013
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight


EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
IMG_SIZE = 48
NUM_CLASSES = 7


def load_fer2013(csv_path: str = "fer2013.csv"):
    """
    Load and parse the FER-2013 CSV file.

    Returns
    -------
    (X_train, y_train), (X_val, y_val), (X_test, y_test)
    Each X is float32 (N, 48, 48, 1), each y is one-hot (N, 7).
    """
    df = pd.read_csv(csv_path)

    pixels = df["pixels"].apply(
        lambda p: np.fromstring(p, sep=" ", dtype=np.float32)
    )
    X = np.stack(pixels).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    y = tf.keras.utils.to_categorical(df["emotion"].values, NUM_CLASSES)

    # FER-2013 splits: Training / PublicTest / PrivateTest
    train_mask = df["Usage"] == "Training"
    val_mask = df["Usage"] == "PublicTest"
    test_mask = df["Usage"] == "PrivateTest"

    return (
        (X[train_mask], y[train_mask]),
        (X[val_mask], y[val_mask]),
        (X[test_mask], y[test_mask]),
    )


def load_fer2013_directory(base_dir: str, val_split: float = 0.1, batch_size: int = 64):
    """
    Alternative loader for the image-folder variant of FER-2013
    (e.g. from Kaggle's 'msambare/fer2013' with train/ and test/ dirs).

    Returns (train_ds, val_ds, test_ds) as tf.data.Dataset objects.
    """
    normalise = tf.keras.layers.Rescaling(1.0 / 255.0)

    train_gen = tf.keras.utils.image_dataset_from_directory(
        f"{base_dir}/train",
        validation_split=val_split,
        subset="training",
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
    )
    val_gen = tf.keras.utils.image_dataset_from_directory(
        f"{base_dir}/train",
        validation_split=val_split,
        subset="validation",
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )
    test_gen = tf.keras.utils.image_dataset_from_directory(
        f"{base_dir}/test",
        image_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    train_gen = train_gen.map(lambda x, y: (normalise(x), y)).prefetch(tf.data.AUTOTUNE)
    val_gen = val_gen.map(lambda x, y: (normalise(x), y)).prefetch(tf.data.AUTOTUNE)
    test_gen = test_gen.map(lambda x, y: (normalise(x), y)).prefetch(tf.data.AUTOTUNE)

    return train_gen, val_gen, test_gen


def get_augmentation_generator() -> ImageDataGenerator:
    """
    Heavy augmentation generator for training.
    These transforms are critical for reaching ~72-75% accuracy.
    """
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )


def compute_class_weights(y_train: np.ndarray) -> dict:
    """
    Compute balanced class weights for the imbalanced FER-2013 dataset.
    'Disgust' class is very under-represented (~1.5% of data).
    """
    y_int = np.argmax(y_train, axis=1)
    weights = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_int)
    return dict(enumerate(weights))


def create_tf_dataset(X: np.ndarray, y: np.ndarray,
                      batch_size: int = 64,
                      augment: bool = False,
                      shuffle: bool = True) -> tf.data.Dataset:
    """
    Build a performant tf.data.Dataset pipeline.
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=42)

    if augment:
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
        ])
        ds = ds.map(lambda x, lbl: (augmentation(x, training=True), lbl),
                     num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
