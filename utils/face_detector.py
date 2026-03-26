"""
Face Detection Utilities
========================
Two detection backends:
  1. Haar Cascade  — fast, works everywhere (default)
  2. MTCNN          — higher accuracy, needs `pip install mtcnn`
"""

import cv2
import numpy as np
from typing import List, Tuple

# Emotion label mapping
EMOTION_LABELS = [
    "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"
]

# Colour for each emotion (BGR)
EMOTION_COLOURS = {
    "Angry":    (0, 0, 255),
    "Disgust":  (0, 128, 0),
    "Fear":     (128, 0, 128),
    "Happy":    (0, 255, 255),
    "Sad":      (255, 128, 0),
    "Surprise": (0, 255, 0),
    "Neutral":  (200, 200, 200),
}


class HaarFaceDetector:
    """OpenCV Haar Cascade face detector — fast, CPU-only."""

    def __init__(self, scale_factor: float = 1.3, min_neighbours: int = 5,
                 min_size: Tuple[int, int] = (48, 48)):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        self.scale_factor = scale_factor
        self.min_neighbours = min_neighbours
        self.min_size = min_size

    def detect(self, gray_frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Returns list of (x, y, w, h) bounding boxes.
        """
        faces = self.detector.detectMultiScale(
            gray_frame,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbours,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return faces if len(faces) > 0 else []


class MTCNNFaceDetector:
    """MTCNN-based face detector — higher accuracy, slightly slower."""

    def __init__(self, min_face_size: int = 48, confidence_threshold: float = 0.9):
        try:
            from mtcnn import MTCNN
        except ImportError:
            raise ImportError(
                "MTCNN not installed. Run: pip install mtcnn"
            )
        self.detector = MTCNN(min_face_size=min_face_size)
        self.confidence_threshold = confidence_threshold

    def detect(self, rgb_frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Expects an RGB frame. Returns list of (x, y, w, h) bounding boxes.
        """
        results = self.detector.detect_faces(rgb_frame)
        faces = []
        for r in results:
            if r["confidence"] >= self.confidence_threshold:
                x, y, w, h = r["box"]
                # MTCNN can return negative coords — clamp
                x, y = max(0, x), max(0, y)
                faces.append((x, y, w, h))
        return faces


def get_face_detector(use_mtcnn: bool = False):
    """Factory: return the requested detector."""
    if use_mtcnn:
        return MTCNNFaceDetector()
    return HaarFaceDetector()


def extract_face_roi(gray_frame: np.ndarray, bbox: Tuple[int, int, int, int],
                     target_size: Tuple[int, int] = (48, 48)) -> np.ndarray:
    """
    Crop, resize, and normalise a face ROI for the emotion model.

    Returns a (1, H, W, 1) float32 array ready for model.predict().
    """
    x, y, w, h = bbox
    roi = gray_frame[y:y + h, x:x + w]

    # Guard against empty crops
    if roi.size == 0:
        return np.zeros((1, *target_size, 1), dtype=np.float32)

    roi = cv2.resize(roi, target_size, interpolation=cv2.INTER_AREA)
    roi = roi.astype("float32") / 255.0
    roi = np.expand_dims(roi, axis=(0, -1))  # (1, 48, 48, 1)
    return roi


def draw_results(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                 emotion: str, confidence: float) -> np.ndarray:
    """
    Draw bounding box, emotion label, and confidence bar on the frame.
    """
    x, y, w, h = bbox
    colour = EMOTION_COLOURS.get(emotion, (255, 255, 255))

    # Bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)

    # Label background
    label = f"{emotion} {confidence:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x, y - th - 14), (x + tw + 8, y), colour, -1)
    cv2.putText(frame, label, (x + 4, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Confidence bar (below the box)
    bar_w = int(w * confidence)
    cv2.rectangle(frame, (x, y + h + 4), (x + bar_w, y + h + 12), colour, -1)
    cv2.rectangle(frame, (x, y + h + 4), (x + w, y + h + 12), colour, 1)

    return frame
