"""
main.py
───────
Real-time face recognition + emotion detection from your webcam.

Controls (press while the camera window is focused):
  q       → Quit
  e       → Enroll mode – type a name in the terminal, then look at the camera
  s       → Screenshot current frame to disk
  t       → Toggle Test-Time Augmentation (TTA) for emotion (slower but better)
  d       → Toggle debug overlay (FPS, face count, confidence)
"""

import cv2
import numpy as np
import time
import os
import sys

# Allow TF to use GPU memory growth (avoid OOM)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

from face_recognition_module import FaceRecognizer, DetectedFace
from emotion_model import (
    build_emotion_model, predict_with_tta, EMOTION_LABELS, IMG_SIZE
)


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
CAMERA_INDEX = 0                 # 0 = default webcam
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
EMOTION_MODEL_PATH = "saved_model/emotion_model.keras"
ENCODER_PATH = "saved_model/face_encoder.keras"
GALLERY_PATH = "gallery.json"

# Color palette (BGR)
COLORS = {
    "angry":    (0,   0,   255),
    "disgust":  (0,   140, 255),
    "fear":     (200, 100, 255),
    "happy":    (0,   220, 0),
    "sad":      (255, 150, 0),
    "surprise": (0,   255, 255),
    "neutral":  (200, 200, 200),
    "unknown":  (128, 128, 128),
}


# ──────────────────────────────────────────────
# Emotion Classifier Wrapper
# ──────────────────────────────────────────────
class EmotionClassifier:
    def __init__(self, model_path: str = EMOTION_MODEL_PATH):
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"✅ Emotion model loaded from {model_path}")
        else:
            print("⚠️  No trained emotion model found – building untrained model.")
            print("   Train with:  python emotion_model.py <fer2013.csv>")
            self.model = build_emotion_model()

    def predict(self, face_bgr: np.ndarray, use_tta: bool = False) -> tuple:
        """
        Takes a BGR face crop of any size, returns (label, probabilities).
        """
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        gray = gray.astype("float32") / 255.0

        if use_tta:
            probs = predict_with_tta(self.model, gray, n_aug=8)
        else:
            inp = gray.reshape(1, IMG_SIZE, IMG_SIZE, 1)
            probs = self.model.predict(inp, verbose=0)[0]

        label = EMOTION_LABELS[np.argmax(probs)]
        return label, probs


# ──────────────────────────────────────────────
# Drawing Helpers
# ──────────────────────────────────────────────
def draw_face(frame: np.ndarray, face: DetectedFace, show_debug: bool = True):
    """Draw bounding box, name, emotion, and confidence bars."""
    x, y, w, h = face.bbox
    emotion = face.emotion or "neutral"
    color = COLORS.get(emotion, (200, 200, 200))

    # Bounding box with rounded corners effect
    thickness = 2
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

    # Corner accents
    corner_len = min(w, h) // 4
    for (cx, cy), (dx, dy) in [
        ((x, y), (1, 1)), ((x + w, y), (-1, 1)),
        ((x, y + h), (1, -1)), ((x + w, y + h), (-1, -1)),
    ]:
        cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), color, 3)
        cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), color, 3)

    # Identity label
    name = face.identity or "Unknown"
    label = f"{name}"
    if face.identity_distance < 1.0:
        conf_pct = max(0, (1 - face.identity_distance)) * 100
        label += f" ({conf_pct:.0f}%)"

    cv2.putText(frame, label, (x, y - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # Emotion label
    if face.emotion_probs is not None:
        top_prob = face.emotion_probs.max() * 100
        emo_text = f"{emotion} {top_prob:.0f}%"
        cv2.putText(frame, emo_text, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Mini emotion bar chart on the right side of the box
        if show_debug:
            bar_x = x + w + 8
            bar_y = y
            bar_w = 80
            bar_h = 12
            for i, (lbl, prob) in enumerate(
                    zip(EMOTION_LABELS, face.emotion_probs)):
                by = bar_y + i * (bar_h + 3)
                fill = int(bar_w * prob)
                cv2.rectangle(frame, (bar_x, by),
                              (bar_x + bar_w, by + bar_h), (50, 50, 50), -1)
                cv2.rectangle(frame, (bar_x, by),
                              (bar_x + fill, by + bar_h),
                              COLORS.get(lbl, (200, 200, 200)), -1)
                cv2.putText(frame, f"{lbl[:3]}", (bar_x + bar_w + 4, by + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)


def draw_hud(frame: np.ndarray, fps: float, face_count: int,
             tta_on: bool, enroll_mode: bool):
    """Heads-up display: FPS, face count, mode indicators."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Faces: {face_count}", (150, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if tta_on:
        cv2.putText(frame, "TTA ON", (300, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if enroll_mode:
        cv2.putText(frame, "ENROLL MODE (look at camera)", (430, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Key hints at bottom
    hints = "Q:Quit  E:Enroll  S:Screenshot  T:TTA  D:Debug"
    cv2.putText(frame, hints, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


# ──────────────────────────────────────────────
# Main Loop
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  FACE RECOGNITION + EMOTION DETECTION")
    print("=" * 60)
    print("Initialising models …")

    recognizer = FaceRecognizer(
        encoder_path=ENCODER_PATH,
        gallery_path=GALLERY_PATH,
    )
    emotion_clf = EmotionClassifier()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("❌ Cannot open camera. Check CAMERA_INDEX in main.py.")
        sys.exit(1)

    print("Camera opened. Press 'q' to quit.\n")

    use_tta = False
    show_debug = True
    enroll_mode = False
    enroll_name = ""
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Detect & recognise faces ──
        faces = recognizer.process_frame(frame)

        # ── Classify emotion for each face ──
        for face in faces:
            x, y, w, h = face.bbox
            crop = frame[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            face.emotion, face.emotion_probs = emotion_clf.predict(
                crop, use_tta=use_tta
            )

        # ── Enroll mode ──
        if enroll_mode and faces:
            largest = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
            recognizer.gallery.add(enroll_name, largest.embedding)
            recognizer.gallery.save(recognizer.gallery_path)
            print(f"  ✅ Enrolled sample for '{enroll_name}'")
            enroll_mode = False

        # ── Draw ──
        for face in faces:
            draw_face(frame, face, show_debug=show_debug)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        draw_hud(frame, fps, len(faces), use_tta, enroll_mode)

        cv2.imshow("Face Recognition + Emotion", frame)

        # ── Keyboard ──
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("e"):
            enroll_name = input("  Enter name to enroll: ").strip()
            if enroll_name:
                enroll_mode = True
                print(f"  Look at the camera to enroll '{enroll_name}' …")
        elif key == ord("s"):
            fname = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print(f"  📸 Saved {fname}")
        elif key == ord("t"):
            use_tta = not use_tta
            print(f"  TTA {'ON' if use_tta else 'OFF'}")
        elif key == ord("d"):
            show_debug = not show_debug

    cap.release()
    cv2.destroyAllWindows()
    print("Goodbye!")


if __name__ == "__main__":
    main()
