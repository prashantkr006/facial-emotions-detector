#!/usr/bin/env python3
"""
Predict Emotion from a Single Image
====================================
Usage:
    python predict_image.py --image photo.jpg
    python predict_image.py --image photo.jpg --model_path checkpoints/best_emotion_model.keras
"""

import argparse
import os
import sys

import cv2
import numpy as np
import tensorflow as tf

from utils.face_detector import (
    get_face_detector, draw_results, EMOTION_LABELS, EMOTION_COLOURS
)
from utils.preprocessor import preprocess_face, postprocess_prediction


def parse_args():
    p = argparse.ArgumentParser(description="Predict emotion from an image")
    p.add_argument("--image", type=str, required=True, help="Path to image")
    p.add_argument("--model_path", type=str,
                   default="checkpoints/best_emotion_model.keras")
    p.add_argument("--use_mtcnn", action="store_true")
    p.add_argument("--output", type=str, default=None,
                   help="Save annotated image to this path")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    if not os.path.exists(args.model_path):
        print(f"[ERROR] Model not found: {args.model_path}")
        sys.exit(1)

    # Load model
    model = tf.keras.models.load_model(args.model_path)
    print(f"[INFO] Model loaded from {args.model_path}")

    # Load image
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"[ERROR] Failed to read image: {args.image}")
        sys.exit(1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if args.use_mtcnn else None

    # Detect faces
    detector = get_face_detector(use_mtcnn=args.use_mtcnn)
    faces = detector.detect(rgb if args.use_mtcnn else gray)
    print(f"[INFO] Detected {len(faces)} face(s)")

    if len(faces) == 0:
        print("[WARN] No faces detected in the image.")
        sys.exit(0)

    # Process each face
    for i, (x, y, w, h) in enumerate(faces):
        face_crop = frame[y:y+h, x:x+w]
        if face_crop.size == 0:
            continue

        face_input = preprocess_face(face_crop)
        pred = model.predict(face_input, verbose=0)
        result = postprocess_prediction(pred, EMOTION_LABELS)

        print(f"\n  Face {i+1}:")
        print(f"    Dominant: {result['dominant_emotion']} "
              f"({result['confidence']:.1%})")
        print(f"    All emotions:")
        for label, prob in sorted(result["all_emotions"].items(),
                                   key=lambda kv: -kv[1]):
            bar = "█" * int(prob * 30)
            print(f"      {label:10s} {prob:6.1%}  {bar}")

        frame = draw_results(frame, (x, y, w, h),
                              result["dominant_emotion"],
                              result["confidence"])

    # Save or display
    output_path = args.output or f"output_{os.path.basename(args.image)}"
    cv2.imwrite(output_path, frame)
    print(f"\n[INFO] Annotated image saved to {output_path}")

    # Try to display (may fail in headless environments)
    try:
        cv2.imshow("Emotion Prediction", frame)
        print("[INFO] Press any key to close ...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        pass


if __name__ == "__main__":
    main()
