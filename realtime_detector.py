#!/usr/bin/env python3
"""
Real-Time Face Emotion Detector
================================
Captures frames from your webcam, detects faces, and classifies
the emotion of each face in real time.

Usage:
    python realtime_detector.py
    python realtime_detector.py --model_path checkpoints/best_emotion_model.keras
    python realtime_detector.py --use_mtcnn   # higher-accuracy face detection
    python realtime_detector.py --camera 1    # use second camera

Controls:
    q / ESC  — quit
    s        — save current frame as screenshot
    d        — toggle emotion detail overlay
"""

import argparse
import time
import os
from collections import defaultdict

import cv2
import numpy as np
import tensorflow as tf

from utils.face_detector import (
    get_face_detector, extract_face_roi, draw_results, EMOTION_LABELS
)
from utils.preprocessor import (
    preprocess_face, postprocess_prediction, apply_temporal_smoothing
)


def parse_args():
    p = argparse.ArgumentParser(description="Real-time emotion detector")
    p.add_argument("--model_path", type=str,
                   default="checkpoints/best_emotion_model.keras",
                   help="Path to trained .keras model")
    p.add_argument("--camera", type=int, default=0,
                   help="Camera index")
    p.add_argument("--use_mtcnn", action="store_true",
                   help="Use MTCNN for face detection (slower, more accurate)")
    p.add_argument("--resolution", type=str, default="640x480",
                   help="Camera resolution WxH")
    p.add_argument("--smoothing_window", type=int, default=5,
                   help="Temporal smoothing window size")
    p.add_argument("--confidence_threshold", type=float, default=0.4,
                   help="Minimum confidence to display emotion")
    return p.parse_args()


def draw_emotion_bars(frame: np.ndarray, emotions: dict,
                      x: int, y: int, w: int):
    """Draw a mini bar chart of all emotion probabilities."""
    bar_height = 12
    bar_max_w = w
    start_y = y

    for label, prob in sorted(emotions.items(), key=lambda kv: -kv[1]):
        bar_w = int(bar_max_w * prob)
        colour = (100, 200, 100) if prob == max(emotions.values()) else (150, 150, 150)

        cv2.rectangle(frame, (x, start_y), (x + bar_w, start_y + bar_height),
                       colour, -1)
        cv2.rectangle(frame, (x, start_y), (x + bar_max_w, start_y + bar_height),
                       (80, 80, 80), 1)

        text = f"{label[:3]} {prob:.0%}"
        cv2.putText(frame, text, (x + bar_max_w + 5, start_y + bar_height - 2),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)

        start_y += bar_height + 3


def setup_camera(camera_index, res_w, res_h):
    """Initialize and validate camera with proper error handling."""
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        print(f"[ERROR] Tried to open camera index: {camera_index}")
        print("[ERROR] On macOS, ensure camera permissions are granted in System Preferences > Security & Privacy.")
        return None
    
    # Apply settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_h)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Use single buffer to avoid stale frames
    
    # Allow camera to warm up
    print("[INFO] Warming up camera...")
    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            break
        time.sleep(0.2)
    
    # Verify camera is working
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("[ERROR] Camera opened but failed to read frames.")
        print("[ERROR] Possible causes:")
        print("  1. Camera permissions not granted to Terminal/Python")
        print("  2. Camera is in use by another application")
        print("  3. Wrong camera index specified")
        print("[ERROR] Try:")
        print("  - Closing other camera apps (Zoom, FaceTime, etc)")
        print("  - Checking System Preferences > Security & Privacy > Camera")
        print("  - Running: python realtime_detector.py --camera 0")
        cap.release()
        return None
    
    actual_w = int(test_frame.shape[1])
    actual_h = int(test_frame.shape[0])
    print(f"[INFO] Camera initialized ({actual_w}x{actual_h}). Press 'q' to quit.")
    return cap


def main():
    args = parse_args()

    # ── Load Model ──────────────────────────────────────────────
    if not os.path.exists(args.model_path):
        print(f"[ERROR] Model not found at '{args.model_path}'.")
        print("        Train a model first with: python train_emotion_model.py")
        print("        Or download a pre-trained model.")
        return

    print(f"[INFO] Loading model from {args.model_path} ...")
    model = tf.keras.models.load_model(args.model_path)
    print("[INFO] Model loaded successfully.")

    # ── Face Detector ───────────────────────────────────────────
    print(f"[INFO] Face detector: {'MTCNN' if args.use_mtcnn else 'Haar Cascade'}")
    face_detector = get_face_detector(use_mtcnn=args.use_mtcnn)

    # ── Camera Setup ────────────────────────────────────────────
    res_w, res_h = map(int, args.resolution.split("x"))
    cap = setup_camera(args.camera, res_w, res_h)
    if cap is None:
        return

    # ── State ───────────────────────────────────────────────────
    smoothing_histories = defaultdict(list)  # per-face smoothing
    show_details = True
    fps_history = []
    frame_count = 0
    screenshot_dir = "screenshots"
    consecutive_failures = 0
    max_failures = 10

    # ── Main Loop ───────────────────────────────────────────────
    while True:
        t_start = time.time()

        ret, frame = cap.read()
        if not ret or frame is None:
            consecutive_failures += 1
            if consecutive_failures > max_failures:
                print(f"[ERROR] Camera failed to read {consecutive_failures} times in a row. Exiting.")
                break
            if consecutive_failures == 1 or consecutive_failures % 5 == 0:
                print(f"[WARN] Failed to read frame ({consecutive_failures}/{max_failures}), retrying ...")
            time.sleep(0.05)
            continue
        
        consecutive_failures = 0  # Reset on successful read

        # Debug: Check frame properties on first frame
        if frame_count == 0:
            print(f"[DEBUG] Frame shape: {frame.shape}, dtype: {frame.dtype}, min: {frame.min()}, max: {frame.max()}")

        # Ensure frame is valid
        if frame.size == 0:
            print("[WARN] Empty frame received")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if args.use_mtcnn else None

        # Detect faces
        faces = face_detector.detect(rgb if args.use_mtcnn else gray)

        for i, (x, y, w, h) in enumerate(faces):
            # Preprocess face ROI
            face_crop = frame[y:y+h, x:x+w]
            if face_crop.size == 0:
                continue
            face_input = preprocess_face(face_crop)

            # Predict
            raw_pred = model.predict(face_input, verbose=0)

            # Temporal smoothing (per face index — simple tracking)
            smoothed = apply_temporal_smoothing(
                smoothing_histories[i], raw_pred,
                window=args.smoothing_window
            )

            result = postprocess_prediction(smoothed, EMOTION_LABELS)

            # Draw results
            if result["confidence"] >= args.confidence_threshold:
                frame = draw_results(
                    frame, (x, y, w, h),
                    result["dominant_emotion"],
                    result["confidence"]
                )

                if show_details:
                    draw_emotion_bars(frame, result["all_emotions"],
                                       x, y + h + 20, w)

        # ── FPS counter ─────────────────────────────────────────
        dt = time.time() - t_start
        fps = 1.0 / max(dt, 1e-6)
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = np.mean(fps_history)

        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 28),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 56),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)

        # ── Display ─────────────────────────────────────────────
        # Ensure frame is in proper uint8 format for OpenCV display
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        cv2.imshow("Emotion Detector", frame)

        # ── Key handling ────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            break
        elif key == ord("s"):
            os.makedirs(screenshot_dir, exist_ok=True)
            path = os.path.join(screenshot_dir, f"emotion_{frame_count:04d}.png")
            cv2.imwrite(path, frame)
            print(f"[INFO] Screenshot saved: {path}")
        elif key == ord("d"):
            show_details = not show_details

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()