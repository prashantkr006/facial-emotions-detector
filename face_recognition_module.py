"""
face_recognition_module.py
───────────────────────────
Face detection (MTCNN) + face recognition via 128-d embeddings.

Workflow:
  1. MTCNN detects faces → bounding boxes + landmarks
  2. Faces are aligned using eye landmarks (affine transform)
  3. A lightweight MobileNet-based encoder maps each face → 128-d vector
  4. Recognition is done via cosine similarity against a stored gallery

The encoder can be swapped for a full FaceNet / ArcFace checkpoint when
you have access to one.  The architecture below is a strong baseline
that you can fine-tune on your own dataset.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import os


# ──────────────────────────────────────────────
# 1.  DATA CLASSES
# ──────────────────────────────────────────────
@dataclass
class DetectedFace:
    """Represents a single detected face in a frame."""
    bbox: Tuple[int, int, int, int]          # (x, y, w, h)
    confidence: float
    landmarks: Optional[Dict] = None         # from MTCNN
    aligned_face: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    identity: Optional[str] = None
    identity_distance: float = 1.0
    emotion: Optional[str] = None
    emotion_probs: Optional[np.ndarray] = None


@dataclass
class FaceGallery:
    """Stores known face embeddings for recognition."""
    embeddings: Dict[str, List[np.ndarray]] = field(default_factory=dict)

    def add(self, name: str, embedding: np.ndarray):
        self.embeddings.setdefault(name, []).append(embedding)

    def match(self, embedding: np.ndarray, threshold: float = 0.55
              ) -> Tuple[str, float]:
        """Cosine-similarity lookup. Returns (name, distance)."""
        best_name, best_dist = "Unknown", 1.0
        for name, embs in self.embeddings.items():
            for ref in embs:
                dist = _cosine_distance(embedding, ref)
                if dist < best_dist:
                    best_dist = dist
                    best_name = name
        if best_dist > threshold:
            return "Unknown", best_dist
        return best_name, best_dist

    def save(self, path: str):
        data = {k: [e.tolist() for e in v] for k, v in self.embeddings.items()}
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "FaceGallery":
        g = cls()
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            g.embeddings = {
                k: [np.array(e, dtype="float32") for e in v]
                for k, v in data.items()
            }
        return g


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ──────────────────────────────────────────────
# 2.  FACE DETECTOR  (MTCNN wrapper)
# ──────────────────────────────────────────────
class FaceDetector:
    """Wraps MTCNN for face detection + 5-point landmark alignment."""

    def __init__(self, min_face_size: int = 40, confidence_threshold: float = 0.92):
        from mtcnn import MTCNN
        self.detector = MTCNN(min_face_size=min_face_size)
        self.conf_thresh = confidence_threshold

    def detect(self, frame_rgb: np.ndarray) -> List[DetectedFace]:
        results = self.detector.detect_faces(frame_rgb)
        faces = []
        for r in results:
            if r["confidence"] < self.conf_thresh:
                continue
            x, y, w, h = r["box"]
            x, y = max(0, x), max(0, y)
            face = DetectedFace(
                bbox=(x, y, w, h),
                confidence=r["confidence"],
                landmarks=r.get("keypoints"),
            )
            face.aligned_face = self._align_and_crop(frame_rgb, face)
            faces.append(face)
        return faces

    @staticmethod
    def _align_and_crop(img: np.ndarray, face: DetectedFace,
                        target_size: int = 160) -> np.ndarray:
        """Align face using eye landmarks → 160×160 RGB crop."""
        x, y, w, h = face.bbox
        if face.landmarks and "left_eye" in face.landmarks:
            le = np.array(face.landmarks["left_eye"], dtype="float32")
            re = np.array(face.landmarks["right_eye"], dtype="float32")
            angle = np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0]))
            center = ((le + re) / 2).astype("float32")
            M = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        # Add 15 % margin
        margin = int(max(w, h) * 0.15)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img.shape[1], x + w + margin)
        y2 = min(img.shape[0], y + h + margin)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            crop = img[y:y + h, x:x + w]
        crop = cv2.resize(crop, (target_size, target_size))
        return crop


# ──────────────────────────────────────────────
# 3.  FACE EMBEDDING MODEL (MobileNetV2 backbone)
# ──────────────────────────────────────────────
def build_face_encoder(input_shape=(160, 160, 3), embedding_dim: int = 128) -> Model:
    """
    Lightweight face encoder based on MobileNetV2.
    Outputs L2-normalised 128-d embeddings.

    For production accuracy, replace this with a pre-trained ArcFace or
    FaceNet checkpoint (see README).
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    # Freeze early layers, fine-tune later ones
    for layer in base.layers[:100]:
        layer.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(embedding_dim, activation=None, name="raw_embedding")(x)
    x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1),
                      name="l2_embedding")(x)

    return Model(base.input, x, name="FaceEncoder")


class FaceRecognizer:
    """High-level recogniser: detector → encoder → gallery lookup."""

    def __init__(self, encoder_path: Optional[str] = None,
                 gallery_path: str = "gallery.json"):
        self.detector = FaceDetector()
        if encoder_path and os.path.exists(encoder_path):
            self.encoder = tf.keras.models.load_model(encoder_path)
        else:
            print("⚠️  No trained encoder found – building a fresh MobileNetV2 encoder.")
            print("   For best results, fine-tune with triplet/ArcFace loss on your data.")
            self.encoder = build_face_encoder()
        self.gallery = FaceGallery.load(gallery_path)
        self.gallery_path = gallery_path

    def process_frame(self, frame_bgr: np.ndarray) -> List[DetectedFace]:
        """Detect, encode, and identify all faces in a BGR frame."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect(frame_rgb)

        if not faces:
            return []

        aligned = np.array([f.aligned_face for f in faces], dtype="float32")
        aligned = tf.keras.applications.mobilenet_v2.preprocess_input(aligned)
        embeddings = self.encoder.predict(aligned, verbose=0)

        for face, emb in zip(faces, embeddings):
            face.embedding = emb
            face.identity, face.identity_distance = self.gallery.match(emb)

        return faces

    def enroll(self, frame_bgr: np.ndarray, name: str) -> bool:
        """Capture the largest face in frame and add it to the gallery."""
        faces = self.process_frame(frame_bgr)
        if not faces:
            return False
        # Pick the largest face
        largest = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        self.gallery.add(name, largest.embedding)
        self.gallery.save(self.gallery_path)
        print(f"✅ Enrolled '{name}' ({len(self.gallery.embeddings[name])} samples)")
        return True
