"""
Image Preprocessing Pipeline
=============================
Handles all pre/post-processing for the emotion recognition pipeline.
"""

import cv2
import numpy as np


def preprocess_face(face_img: np.ndarray,
                    target_size: tuple = (48, 48)) -> np.ndarray:
    """
    Full preprocessing pipeline for a single face crop.

    Steps:
      1. Convert to grayscale (if needed)
      2. Apply CLAHE histogram equalisation for lighting robustness
      3. Resize to target_size
      4. Normalise to [0, 1]
      5. Reshape to (1, H, W, 1)
    """
    # To grayscale
    if len(face_img.shape) == 3 and face_img.shape[2] == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img.copy()

    # CLAHE — adaptive histogram equalisation (greatly helps accuracy
    # under varying lighting conditions)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Resize
    gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

    # Normalise
    gray = gray.astype("float32") / 255.0

    # Expand dims → (1, 48, 48, 1)
    return np.expand_dims(gray, axis=(0, -1))


def postprocess_prediction(predictions: np.ndarray,
                           emotion_labels: list) -> dict:
    """
    Convert raw softmax output to a structured result.

    Returns
    -------
    dict with keys:
      - dominant_emotion : str
      - confidence       : float (0-1)
      - all_emotions     : dict[str, float]
    """
    probs = predictions[0]
    top_idx = int(np.argmax(probs))

    return {
        "dominant_emotion": emotion_labels[top_idx],
        "confidence": float(probs[top_idx]),
        "all_emotions": {
            label: float(prob)
            for label, prob in zip(emotion_labels, probs)
        },
    }


def apply_temporal_smoothing(history: list, current: np.ndarray,
                             window: int = 5, decay: float = 0.7) -> np.ndarray:
    """
    Exponentially-weighted moving average over recent predictions.
    Reduces jitter in real-time detection.

    Parameters
    ----------
    history : list of np.ndarray — previous softmax outputs
    current : np.ndarray         — current softmax output (1, num_classes)
    window  : int                — max history length
    decay   : float              — exponential decay factor (higher = more smoothing)
    """
    history.append(current[0])
    if len(history) > window:
        history.pop(0)

    weights = np.array([decay ** i for i in range(len(history) - 1, -1, -1)])
    weights /= weights.sum()

    smoothed = np.zeros_like(history[0])
    for w, h in zip(weights, history):
        smoothed += w * h

    return smoothed.reshape(1, -1)
