# Copyright (C) 2026 Julia Jia
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Manual test for webcam image capture and gesture prediction.

This test is marked as manual and excluded from CI.

Run manually with:

    pixi run test tests/test_webcam.py -m manual -v -s
"""

from pathlib import Path

import pytest

from handmotion.model import HandGestureClassifier
from handmotion.predict import predict_image

pytestmark = pytest.mark.manual


def test_webcam_capture_and_predict():
    """Test capturing a frame from webcam and predicting gesture."""
    try:
        import cv2
        from PIL import Image
    except ImportError:
        pytest.skip("opencv-python or PIL not installed")

    # Check if model exists
    model_path = Path("models/rps_classifier.joblib")
    if not model_path.exists():
        pytest.skip(f"Model not found: {model_path}")

    # Load model
    classifier = HandGestureClassifier()
    classifier.load(model_path)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        pytest.skip("Webcam not available")

    try:
        # Capture a single frame
        ret, frame = cap.read()
        assert ret, "Failed to capture frame from webcam"
        assert frame is not None, "Frame is None"
        assert frame.shape[0] > 0, "Frame height must be greater than 0"
        assert frame.shape[1] > 0, "Frame width must be greater than 0"

        # Convert BGR to RGB and then to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        # Predict using clean API
        prediction, confidence = predict_image(classifier, image, threshold=0.5)

        if prediction is None:
            pytest.skip("No hand detected in frame or confidence below threshold")
        assert confidence is not None

        # Validate prediction
        class_names = classifier.label_encoder.classes_
        assert prediction in class_names, f"Prediction {prediction} not in class names"
        assert 0.0 <= confidence <= 1.0, "Confidence out of range"
        assert confidence >= 0.5, "Confidence below threshold"

        # Print results
        print(f"\nPrediction: {prediction}")
        print(f"Confidence: {confidence:.4f}")

    finally:
        cap.release()
