#!/usr/bin/env python3
"""
Training Script — Emotion Recognition Model
=============================================
Trains the Mini-Xception (or deeper V2) model on FER-2013.

Usage (CSV format):
    python train_emotion_model.py --dataset_path fer2013.csv --epochs 100

Usage (image folder format from kagglehub):
    python train_emotion_model.py --dataset_path ~/.cache/kagglehub/datasets/msambare/fer2013/versions/1

Expected accuracy:
    Mini-Xception : ~70-72% on FER-2013 private test
    V2 (deep)     : ~73-75% on FER-2013 private test
"""

import argparse
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)

# Local imports
from model.emotion_model import build_mini_xception, build_emotion_model_v2
from utils.data_loader import (
    load_fer2013, load_fer2013_directory,
    compute_class_weights, create_tf_dataset
)


def parse_args():
    p = argparse.ArgumentParser(description="Train emotion recognition model")
    p.add_argument("--dataset_path", type=str, default="fer2013.csv",
                   help="Path to fer2013.csv OR image folder (train/test dirs)")
    p.add_argument("--model", type=str, default="mini_xception",
                   choices=["mini_xception", "v2"],
                   help="Model architecture to use")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Initial learning rate")
    p.add_argument("--output_dir", type=str, default="checkpoints",
                   help="Directory to save model weights")
    p.add_argument("--use_class_weights", action="store_true", default=True,
                   help="Use balanced class weights (CSV mode only)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load Data ────────────────────────────────────────────
    is_directory = os.path.isdir(args.dataset_path)
    class_weights = None
    y_test = None

    if is_directory:
        print(f"[INFO] Loading dataset from image folder: {args.dataset_path} ...")
        train_ds, val_ds, test_ds = load_fer2013_directory(
            args.dataset_path, batch_size=args.batch_size
        )
        print("[INFO] Note: class weights not available in image-folder mode.")
    else:
        print(f"[INFO] Loading dataset from CSV: {args.dataset_path} ...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_fer2013(
            args.dataset_path
        )
        print(f"  Train : {X_train.shape[0]:,} samples")
        print(f"  Val   : {X_val.shape[0]:,} samples")
        print(f"  Test  : {X_test.shape[0]:,} samples")

        train_ds = create_tf_dataset(X_train, y_train, args.batch_size,
                                     augment=True, shuffle=True)
        val_ds = create_tf_dataset(X_val, y_val, args.batch_size,
                                   augment=False, shuffle=False)
        test_ds = create_tf_dataset(X_test, y_test, args.batch_size,
                                    augment=False, shuffle=False)

        if args.use_class_weights:
            class_weights = compute_class_weights(y_train)
            print(f"[INFO] Class weights: {class_weights}")

    # ── 2. Build Model ──────────────────────────────────────────
    if args.model == "v2":
        print("[INFO] Building deep VGG-style model (V2) ...")
        model = build_emotion_model_v2()
    else:
        print("[INFO] Building Mini-Xception model ...")
        model = build_mini_xception()

    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # ── 3. Callbacks ────────────────────────────────────────────
    best_model_path = os.path.join(args.output_dir, "best_emotion_model.keras")

    callbacks = [
        ModelCheckpoint(
            best_model_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=20,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    # ── 4. Train ────────────────────────────────────────────────
    print(f"\n[INFO] Training for up to {args.epochs} epochs ...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # ── 5. Evaluate on Test Set ─────────────────────────────────
    print("\n[INFO] Evaluating on test set ...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\n{'='*50}")
    print(f"  TEST ACCURACY : {test_acc:.4f}  ({test_acc:.1%})")
    print(f"  TEST LOSS     : {test_loss:.4f}")
    print(f"{'='*50}")

    # ── 6. Save Final Model ─────────────────────────────────────
    final_path = os.path.join(args.output_dir, "emotion_model_final.keras")
    model.save(final_path)
    print(f"\n[INFO] Final model saved to {final_path}")
    print(f"[INFO] Best model saved to {best_model_path}")

    # ── 7. Per-class accuracy report ────────────────────────────
    from utils.data_loader import EMOTION_LABELS
    from sklearn.metrics import classification_report

    y_pred = model.predict(test_ds)
    y_pred_classes = np.argmax(y_pred, axis=1)

    if y_test is not None:
        y_true_classes = np.argmax(y_test, axis=1)
    else:
        y_true_classes = np.concatenate(
            [np.argmax(y.numpy(), axis=1) for _, y in test_ds]
        )

    print("\n[INFO] Per-class results:")
    print(classification_report(
        y_true_classes, y_pred_classes,
        target_names=EMOTION_LABELS, digits=3
    ))


if __name__ == "__main__":
    main()
