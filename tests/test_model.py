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

"""Unit tests for hand gesture classifier model."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from handmotion.model import HandGestureClassifier


class TestHandGestureClassifier:
    """Test cases for HandGestureClassifier."""

    def test_init_default(self):
        """Test initialization with default classifier."""
        classifier = HandGestureClassifier()
        assert isinstance(classifier.classifier, RandomForestClassifier)
        assert classifier.classifier.random_state == 42
        assert not classifier.is_trained

    def test_init_custom_classifier(self):
        """Test initialization with custom classifier."""
        custom_clf = LogisticRegression()
        classifier = HandGestureClassifier(classifier=custom_clf)
        assert classifier.classifier is custom_clf
        assert not classifier.is_trained

    def test_prepare_features_3d_landmarks(self):
        """Test _prepare_features with 3D landmarks array."""
        classifier = HandGestureClassifier()
        landmarks = np.random.rand(5, 21, 3)
        finger_states = np.random.rand(5, 5)

        features = classifier._prepare_features(landmarks, finger_states)

        assert features.shape == (5, 68)
        assert features.ndim == 2

    def test_prepare_features_2d_landmarks(self):
        """Test _prepare_features with 2D landmarks array."""
        classifier = HandGestureClassifier()
        landmarks = np.random.rand(21, 3)
        finger_states = np.array([True, False, True, False, True])

        features = classifier._prepare_features(landmarks, finger_states)

        assert features.shape == (1, 68)
        assert features.ndim == 2

    def test_prepare_features_1d_finger_states(self):
        """Test _prepare_features with 1D finger_states (single sample)."""
        classifier = HandGestureClassifier()
        landmarks = np.random.rand(21, 3)  # Single sample, 2D
        finger_states = np.array([True, False, True, False, True])  # 1D

        features = classifier._prepare_features(landmarks, finger_states)

        assert features.shape == (1, 68)

    def test_prepare_features_batch_with_2d_finger_states(self):
        """Test _prepare_features with 3D landmarks and 2D finger_states (batch)."""
        classifier = HandGestureClassifier()
        landmarks = np.random.rand(3, 21, 3)  # Batch of 3
        finger_states = np.random.rand(3, 5)  # Batch of 3, matching size

        features = classifier._prepare_features(landmarks, finger_states)

        assert features.shape == (3, 68)

    def test_load_data(self, tmp_path):
        """Test load_data loads npz file correctly."""
        # Use helper function from conftest
        from conftest import create_test_npz

        classifier = HandGestureClassifier()
        data_path, landmarks, finger_states, labels = create_test_npz(tmp_path, n_samples=10)

        features, loaded_labels = classifier.load_data(data_path)

        assert features.shape == (10, 68)
        assert len(loaded_labels) == 10
        assert np.array_equal(loaded_labels, labels)

    def test_load_data_nonexistent(self):
        """Test load_data raises error for nonexistent file."""
        classifier = HandGestureClassifier()
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            classifier.load_data("/nonexistent/path/data.npz")

    def test_train(self):
        """Test train method trains classifier correctly."""
        classifier = HandGestureClassifier()
        features = np.random.rand(100, 68)
        labels = np.array(["rock", "paper", "scissors"] * 33 + ["rock"])

        X_train, X_test, y_train, y_test = classifier.train(features, labels, test_size=0.2)

        assert classifier.is_trained
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
        assert X_train.shape[1] == 68
        assert X_test.shape[1] == 68

    def test_train_custom_test_size(self):
        """Test train with custom test_size."""
        classifier = HandGestureClassifier()
        features = np.random.rand(100, 68)
        labels = np.array(["rock", "paper", "scissors"] * 33 + ["rock"])

        X_train, X_test, y_train, y_test = classifier.train(features, labels, test_size=0.3)

        assert len(X_train) == 70
        assert len(X_test) == 30

    def test_predict_not_trained(self):
        """Test predict raises error when model not trained."""
        classifier = HandGestureClassifier()
        features = np.random.rand(68)

        with pytest.raises(ValueError, match="Classifier must be trained before prediction"):
            classifier.predict(features)

    def test_predict_1d_features(self):
        """Test predict with 1D feature array."""
        classifier = HandGestureClassifier()
        features = np.random.rand(100, 68)
        labels = np.array(["rock", "paper", "scissors"] * 33 + ["rock"])

        classifier.train(features, labels)

        single_feature = np.random.rand(68)
        predictions = classifier.predict(single_feature)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 1
        assert predictions[0] in ["rock", "paper", "scissors"]

    def test_predict_2d_features(self):
        """Test predict with 2D feature array."""
        classifier = HandGestureClassifier()
        features = np.random.rand(100, 68)
        labels = np.array(["rock", "paper", "scissors"] * 33 + ["rock"])

        classifier.train(features, labels)

        test_features = np.random.rand(5, 68)
        predictions = classifier.predict(test_features)

        assert len(predictions) == 5
        assert all(pred in ["rock", "paper", "scissors"] for pred in predictions)

    def test_predict_proba_not_trained(self):
        """Test predict_proba raises error when model not trained."""
        classifier = HandGestureClassifier()
        features = np.random.rand(68)

        with pytest.raises(ValueError, match="Classifier must be trained before prediction"):
            classifier.predict_proba(features)

    def test_predict_proba(self):
        """Test predict_proba returns probabilities."""
        classifier = HandGestureClassifier()
        features = np.random.rand(100, 68)
        labels = np.array(["rock", "paper", "scissors"] * 33 + ["rock"])

        classifier.train(features, labels)

        test_features = np.random.rand(3, 68)
        probabilities = classifier.predict_proba(test_features)

        assert probabilities.shape == (3, 3)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)

    def test_predict_proba_1d_features(self):
        """Test predict_proba with 1D feature array."""
        classifier = HandGestureClassifier()
        features = np.random.rand(100, 68)
        labels = np.array(["rock", "paper", "scissors"] * 33 + ["rock"])

        classifier.train(features, labels)

        single_feature = np.random.rand(68)
        probabilities = classifier.predict_proba(single_feature)

        assert probabilities.shape == (1, 3)
        assert np.allclose(probabilities.sum(), 1.0)

    def test_save_not_trained(self, tmp_path):
        """Test save raises error when model not trained."""
        classifier = HandGestureClassifier()
        model_path = tmp_path / "model.joblib"

        with pytest.raises(ValueError, match="Cannot save untrained model"):
            classifier.save(model_path)

    def test_save_load(self, tmp_path):
        """Test save and load model."""
        classifier = HandGestureClassifier()
        features = np.random.rand(100, 68)
        labels = np.array(["rock", "paper", "scissors"] * 33 + ["rock"])

        classifier.train(features, labels)
        model_path = tmp_path / "model.joblib"

        classifier.save(model_path)
        assert model_path.exists()

        # Create new classifier and load
        new_classifier = HandGestureClassifier()
        new_classifier.load(model_path)

        assert new_classifier.is_trained
        assert isinstance(new_classifier.classifier, RandomForestClassifier)

        # Test that loaded model can predict
        test_features = np.random.rand(5, 68)
        predictions = new_classifier.predict(test_features)
        assert len(predictions) == 5

    def test_save_creates_parent_dirs(self, tmp_path):
        """Test save creates parent directories if needed."""
        classifier = HandGestureClassifier()
        features = np.random.rand(100, 68)
        labels = np.array(["rock", "paper", "scissors"] * 33 + ["rock"])

        classifier.train(features, labels)
        model_path = tmp_path / "models" / "subdir" / "model.joblib"

        classifier.save(model_path)
        assert model_path.exists()

    def test_load_nonexistent(self):
        """Test load raises error for nonexistent file."""
        classifier = HandGestureClassifier()
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            classifier.load("/nonexistent/path/model.joblib")

    def test_predict_after_load(self, tmp_path):
        """Test that predictions are consistent after save/load."""
        classifier = HandGestureClassifier()
        features = np.random.rand(100, 68)
        labels = np.array(["rock", "paper", "scissors"] * 33 + ["rock"])

        classifier.train(features, labels)
        model_path = tmp_path / "model.joblib"
        classifier.save(model_path)

        # Get predictions before load
        test_features = np.random.rand(3, 68)
        predictions_before = classifier.predict(test_features)

        # Load and get predictions
        new_classifier = HandGestureClassifier()
        new_classifier.load(model_path)
        predictions_after = new_classifier.predict(test_features)

        # Predictions should be the same
        assert np.array_equal(predictions_before, predictions_after)
