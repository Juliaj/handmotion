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

"""Prediction script for hand gesture classifier.

Examples:
    # Predict on a single image
    python -m handmotion.predict --model models/rps_classifier.joblib --image path/to/image.jpg

    # Predict with confidence scores
    python -m handmotion.predict \
        --model models/rps_classifier.joblib \
        --image path/to/image.jpg --show-proba
"""

import argparse
import logging
from pathlib import Path

from PIL import Image

from handmotion.data.feature_extractor import FeatureExtractor
from handmotion.model import HandGestureClassifier

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def predict_image(classifier, extractor, image_path):
    """
    Predict gesture for a single image.

    Args:
        classifier: Loaded HandGestureClassifier
        extractor: FeatureExtractor instance
        image_path: Path to image file

    Returns:
        Tuple of (predicted_label, confidence_dict) or (None, None) if no hand detected
    """
    # Load image
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Extract features
    features_dict = extractor.extract(image, image_format="rgb")
    if features_dict is None:
        logger.warning(f"No hand detected in {image_path}")
        return None, None

    # Prepare features for classifier
    landmarks = features_dict["landmarks"].reshape(1, -1)  # (1, 63)
    finger_states = features_dict["finger_states"].reshape(1, -1)  # (1, 5)
    features = classifier._prepare_features(landmarks, finger_states)  # (1, 68)

    # Predict
    prediction = classifier.predict(features)[0]
    probabilities = classifier.predict_proba(features)[0]

    # Get class names and create confidence dict
    class_names = classifier.label_encoder.classes_
    confidence_dict = {class_names[i]: float(prob) for i, prob in enumerate(probabilities)}

    return prediction, confidence_dict


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Predict hand gesture from image")
    parser.add_argument(
        "--model",
        type=str,
        default="models/rps_classifier.joblib",
        help="Path to trained model file (default: models/rps_classifier.joblib)",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image file",
    )
    parser.add_argument(
        "--show-proba",
        action="store_true",
        help="Show probability scores for all classes",
    )

    args = parser.parse_args()

    # Validate paths
    model_path = Path(args.model)
    image_path = Path(args.image)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load model
    logger.info(f"Loading model from {model_path}...")
    classifier = HandGestureClassifier()
    classifier.load(model_path)

    # Initialize feature extractor
    extractor = FeatureExtractor()

    # Predict
    logger.info(f"Processing image: {image_path}")
    prediction, confidence = predict_image(classifier, extractor, image_path)

    if prediction is None:
        print("No hand detected in image.")
        return

    # Print results
    print(f"\nPrediction: {prediction}")
    print(f"Confidence: {confidence[prediction]:.4f}")

    if args.show_proba:
        print("\nAll class probabilities:")
        for class_name, prob in sorted(confidence.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {prob:.4f}")


if __name__ == "__main__":
    main()
