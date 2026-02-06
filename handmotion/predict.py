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
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image

from handmotion.data.feature_extractor import FeatureExtractor
from handmotion.model import HandGestureClassifier

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _extract_and_predict(
    classifier: HandGestureClassifier,
    image: Union[Image.Image, str, Path],
    image_format: str = "rgb",
) -> Tuple[Optional[str], Optional[np.ndarray], Optional[list]]:
    """
    Internal helper to extract features and get predictions.

    Args:
        classifier: Loaded HandGestureClassifier
        image: PIL Image, or path to image file
        image_format: "rgb" or "bgr" (default: "rgb")

    Returns:
        Tuple of (prediction, probabilities, class_names) or (None, None, None) if no hand detected
    """
    # Load image if path provided
    if isinstance(image, (str, Path)):
        image = Image.open(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_format = "rgb"  # PIL images are always RGB

    # Initialize feature extractor
    extractor = FeatureExtractor()

    # Extract features
    features_dict = extractor.extract(image, image_format=image_format)
    if features_dict is None:
        return None, None, None

    # Prepare features for classifier
    landmarks = features_dict["landmarks"].reshape(1, -1)  # (1, 63)
    finger_states = features_dict["finger_states"].reshape(1, -1)  # (1, 5)
    features = classifier._prepare_features(landmarks, finger_states)  # (1, 68)

    # Predict
    prediction = classifier.predict(features)[0]
    probabilities = classifier.predict_proba(features)[0]
    class_names = classifier.label_encoder.classes_

    return prediction, probabilities, class_names


def predict_image(
    classifier: HandGestureClassifier,
    image: Union[Image.Image, str, Path],
    threshold: float = 0.5,
    image_format: str = "rgb",
) -> Tuple[Optional[str], Optional[float]]:
    """
    Predict gesture from an image.

    Args:
        classifier: Loaded HandGestureClassifier
        image: PIL Image, or path to image file
        threshold: Minimum confidence to return prediction (0.0-1.0).
                  If confidence is below threshold, returns (None, None)
        image_format: "rgb" or "bgr" (default: "rgb"). Only used if image is PIL Image.

    Returns:
        Tuple of (label, confidence) or (None, None) if:
        - No hand detected
        - Confidence below threshold
    """
    prediction, probabilities, class_names = _extract_and_predict(classifier, image, image_format)
    if prediction is None:
        logger.debug("No hand detected in image")
        return None, None

    # Get confidence for predicted class
    pred_idx = list(class_names).index(prediction)
    confidence = float(probabilities[pred_idx])

    # Check threshold
    if confidence < threshold:
        logger.debug(f"Confidence {confidence:.4f} below threshold {threshold} for {prediction}")
        return None, None

    return prediction, confidence


def predict_image_with_proba(
    classifier: HandGestureClassifier,
    image: Union[Image.Image, str, Path],
    image_format: str = "rgb",
) -> Tuple[Optional[str], Optional[dict]]:
    """
    Predict gesture from an image with full probability distribution.

    Args:
        classifier: Loaded HandGestureClassifier
        image: PIL Image, or path to image file
        image_format: "rgb" or "bgr" (default: "rgb")

    Returns:
        Tuple of (predicted_label, confidence_dict) or (None, None) if no hand detected
    """
    prediction, probabilities, class_names = _extract_and_predict(classifier, image, image_format)
    if prediction is None:
        logger.warning("No hand detected in image")
        return None, None

    # Get class names and create confidence dict
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

    # Predict
    logger.info(f"Processing image: {image_path}")
    if args.show_proba:
        prediction, confidence = predict_image_with_proba(classifier, image_path)
        if prediction is None:
            print("No hand detected in image.")
            return
        print(f"\nPrediction: {prediction}")
        print(f"Confidence: {confidence[prediction]:.4f}")
        print("\nAll class probabilities:")
        for class_name, prob in sorted(confidence.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {prob:.4f}")
    else:
        prediction, confidence = predict_image(classifier, image_path)
        if prediction is None:
            print("No hand detected in image.")
            return
        print(f"\nPrediction: {prediction}")
        print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
