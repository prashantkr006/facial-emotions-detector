# Face Recognition + Emotion Detection System

Real-time face recognition and 7-class emotion detection from your webcam, built with TensorFlow, MTCNN, and OpenCV.

---

## Features

| Capability | Details |
|---|---|
| **Face Detection** | MTCNN – multi-task cascaded CNN with 5-point landmark alignment |
| **Face Recognition** | MobileNetV2 encoder → 128-d L2-normalised embeddings, cosine-similarity gallery matching |
| **Emotion Detection** | Deep residual CNN with SE-attention, 7 emotions (angry, disgust, fear, happy, sad, surprise, neutral) |
| **Accuracy Boosters** | Label smoothing, Mixup augmentation, SpatialDropout, cosine LR decay, Test-Time Augmentation (TTA) |
| **Real-time HUD** | FPS counter, per-face emotion bar charts, identity + confidence display |
| **Enrollment** | Press `E` to add a face to the gallery in real-time |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU recommended.** Install `tensorflow[and-cuda]` for NVIDIA GPU support.

### 2. Train the emotion model (one-time)

Download the FER-2013 dataset:
https://www.kaggle.com/datasets/msambare/fer2013

```bash
python emotion_model.py path/to/fer2013.csv
```

This saves `saved_model/emotion_model.keras` (~73-75% validation accuracy).

### 3. Run the live camera

```bash
python main.py
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `E` | Enroll a face – type a name, then look at the camera |
| `S` | Save a screenshot |
| `T` | Toggle Test-Time Augmentation (slower, more accurate emotions) |
| `D` | Toggle debug overlay |

---

## Architecture

### Emotion Model (EmotionNet)

```
Input (48x48x1 grayscale)
  |
  +-- Conv2D 64, 5x5 -> BN -> ReLU -> MaxPool
  |
  +-- ResBlock(128) + SE-Attention + SpatialDropout(0.25)
  +-- ResBlock(256) + SE-Attention + SpatialDropout(0.25)
  +-- ResBlock(512) + SE-Attention + SpatialDropout(0.40)
  |
  +-- GlobalAveragePooling
  +-- Dense(256) -> Dropout(0.5)
  +-- Dense(7, softmax)
```

**Why this achieves high accuracy:**

- **Residual connections** allow training deeper networks without degradation
- **Squeeze-and-Excitation blocks** learn channel-wise feature importance
- **SpatialDropout** regularises entire feature maps (better than standard dropout for CNNs)
- **Label smoothing (0.1)** prevents overconfident predictions
- **Mixup augmentation** creates virtual training samples by blending image pairs
- **Cosine LR decay** with warm-up for smooth convergence
- **TTA at inference** averages predictions over augmented copies for ~1-2% accuracy boost

### Face Encoder

```
MobileNetV2 (ImageNet pretrained, partially frozen)
  -> GlobalAveragePooling
  -> Dense(256, ReLU)
  -> Dense(128)
  -> L2 Normalisation
```

For production, swap with a pre-trained ArcFace / FaceNet model.

---

## Upgrading to Production-Grade Accuracy

### For Face Recognition
Replace the MobileNetV2 encoder with a pre-trained ArcFace model:

```python
# In face_recognition_module.py, update FaceRecognizer.__init__:
self.encoder = tf.keras.models.load_model("path/to/arcface_model")
```

ArcFace achieves >99.8% on LFW benchmark.

### For Emotion Detection
Fine-tune on additional datasets alongside FER-2013:
- **AffectNet** (~440K images, 8 classes)
- **RAF-DB** (30K images)
- **ExpW** (91K images)

Combined training typically pushes accuracy to 78-82% on FER-2013.

---

## File Structure

```
face_emotion/
  main.py                      # Camera loop + HUD
  emotion_model.py             # Emotion CNN: build, train, TTA
  face_recognition_module.py   # MTCNN detector + encoder + gallery
  requirements.txt             # Python dependencies
  gallery.json                 # Auto-created face gallery
  saved_model/
    emotion_model.keras        # After training
    face_encoder.keras         # Optional fine-tuned encoder
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera won't open | Change `CAMERA_INDEX` in `main.py` (try 1 or 2) |
| Low FPS | Disable TTA (T key), reduce resolution, use GPU |
| Bad emotion accuracy | Train the emotion model on FER-2013 first |
| Face not recognised | Enroll multiple samples (different angles/lighting) |
| MTCNN import error | `pip install mtcnn` (requires TensorFlow >= 2.x) |
