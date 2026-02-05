# Models Directory

This directory contains model files used by the hand gesture classification system.

## Files

### `hand_landmarker.task`
MediaPipe hand landmark detection model used for feature extraction.

- Purpose: Detects 21 hand landmark points in images
- Used by: `FeatureExtractor` class
- Source: MediaPipe hand landmarker model
- Download: If missing, manually download from:
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
  Or see official documentation: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
  The system will also attempt to use MediaPipe's bundled model if available.

### `rps_classifier.joblib`
Trained scikit-learn classifier for rock-paper-scissors gesture classification.

- Purpose: Classifies extracted hand features into rock, paper, or scissors
- Created by: Training script (`handmotion.train`)
- Used by: Prediction script (`handmotion.predict`)
- Format: Joblib-serialized scikit-learn model

### `metrics.json`
Evaluation metrics from model training.

- Purpose: Stores accuracy, confusion matrix, and classification report
- Created by: Training script (`handmotion.train`)
- Format: JSON file with evaluation results

## Model Pipeline

1. `hand_landmarker.task` extracts hand landmarks from raw images
2. Features are processed and saved to `data/processed/rps/data.npz`
3. `rps_classifier.joblib` is trained on these features
4. Trained classifier is used for prediction on new images
