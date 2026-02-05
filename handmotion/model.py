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

"""Model for classifying hand gestures using scikit-learn."""

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class HandGestureClassifier:
    """Classifier for rock-paper-scissors hand gestures."""

    def __init__(self, classifier=None):
        """
        Initialize the classifier.

        Args:
            classifier: scikit-learn classifier instance. If None, uses RandomForestClassifier.
                       RandomForestClassifier: Good for fixed-size feature vectors with non-linear
                       relationships, provides feature importance, no scaling needed. Limitations:
                       Less interpretable than simple models (e.g., LogisticRegression with
                       interpretable coefficients), can overfit with many trees, slower than
                       linear models.
        """
        if classifier is None:
            self.classifier = RandomForestClassifier(random_state=42)
        else:
            self.classifier = classifier

        self.label_encoder = LabelEncoder()
        self.is_trained = False

    def _prepare_features(self, landmarks, finger_states):
        """
        Combine landmarks and finger_states into a single feature vector.

        Args:
            landmarks: Array of shape (n_samples, 21, 3) or (21, 3)
            finger_states: Array of shape (n_samples, 5) or (5,)

        Returns:
            Feature array of shape (n_samples, 68) or (68,)
        """
        # Flatten landmarks: (n_samples, 21, 3) -> (n_samples, 63)
        if landmarks.ndim == 3:
            landmarks_flat = landmarks.reshape(landmarks.shape[0], -1)
        else:
            landmarks_flat = landmarks.reshape(1, -1)

        # Ensure finger_states is 2D
        if finger_states.ndim == 1:
            finger_states = finger_states.reshape(1, -1)

        # Combine: (n_samples, 63) + (n_samples, 5) -> (n_samples, 68)
        features = np.hstack([landmarks_flat, finger_states])

        return features

    def load_data(self, data_path):
        """
        Load data from npz file.

        Args:
            data_path: Path to the npz file

        Returns:
            Tuple of (features, labels) where features is (n_samples, 68) and labels is (n_samples,)
        """
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        data = np.load(data_path, allow_pickle=True)
        landmarks = data["landmarks"]
        finger_states = data["finger_states"]
        labels = data["labels"]

        features = self._prepare_features(landmarks, finger_states)
        labels = np.array(labels)

        logger.info(f"Loaded {len(features)} samples from {data_path}")
        return features, labels

    def train(self, features, labels, test_size=0.2, random_state=42):
        """
        Train the classifier.

        Args:
            features: Feature array of shape (n_samples, 68)
            labels: Label array of shape (n_samples,)
            test_size: Proportion of data to use for testing
            random_state: Random state for train_test_split

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=labels_encoded,
        )

        # Train classifier
        self.classifier.fit(X_train, y_train)
        self.is_trained = True

        logger.info(f"Trained classifier on {len(X_train)} samples")
        return X_train, X_test, y_train, y_test

    def predict(self, features):
        """
        Predict labels for given features.

        Args:
            features: Feature array of shape (n_samples, 68) or (68,)

        Returns:
            Predicted labels as strings
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before prediction")

        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Predict
        predictions_encoded = self.classifier.predict(features)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)

        return predictions

    def predict_proba(self, features):
        """
        Predict class probabilities for given features.

        This provides confidence scores (e.g., [0.9, 0.05, 0.05] vs [0.4, 0.35, 0.25]),
        enables uncertainty detection when probabilities are close, allows threshold-based
        rejection of low-confidence predictions, and helps with debugging model behavior.

        Args:
            features: Feature array of shape (n_samples, 68) or (68,)

        Returns:
            Probability array of shape (n_samples, n_classes)
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before prediction")

        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)

        return self.classifier.predict_proba(features)

    def save(self, model_path):
        """
        Save the trained model to disk.

        Args:
            model_path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "classifier": self.classifier,
            "label_encoder": self.label_encoder,
        }

        joblib.dump(model_data, model_path)
        logger.info(f"Saved model to {model_path}")

    def load(self, model_path):
        """
        Load a trained model from disk.

        Args:
            model_path: Path to the saved model
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model_data = joblib.load(model_path)
        self.classifier = model_data["classifier"]
        self.label_encoder = model_data["label_encoder"]
        self.is_trained = True

        logger.info(f"Loaded model from {model_path}")
