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

"""Unit tests for prediction script."""

from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from handmotion.predict import predict_image, predict_image_with_proba


class TestPredictImage:
    """Test cases for predict_image function."""

    @patch("handmotion.predict.FeatureExtractor")
    def test_predict_image_success(self, mock_extractor_class, sample_image, mock_classifier):
        """Test successful prediction."""
        # Setup mock extractor
        mock_extractor = MagicMock()
        mock_extractor_class.return_value = mock_extractor
        mock_extractor.extract.return_value = {
            "landmarks": np.random.rand(21, 3),
            "finger_states": np.array([True, False, True, False, True]),
        }

        prediction, confidence = predict_image(mock_classifier, sample_image)

        assert prediction == "rock"
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        mock_extractor.extract.assert_called_once()

    @patch("handmotion.predict.FeatureExtractor")
    def test_predict_image_no_hand_detected(
        self, mock_extractor_class, sample_image, mock_classifier
    ):
        """Test prediction when no hand is detected."""
        mock_extractor = MagicMock()
        mock_extractor_class.return_value = mock_extractor
        mock_extractor.extract.return_value = None

        prediction, confidence = predict_image(mock_classifier, sample_image)

        assert prediction is None
        assert confidence is None
        mock_extractor.extract.assert_called_once()
        mock_classifier.predict.assert_not_called()

    @patch("handmotion.predict.FeatureExtractor")
    def test_predict_image_below_threshold(
        self, mock_extractor_class, sample_image, mock_classifier
    ):
        """Test prediction when confidence is below threshold."""
        mock_extractor = MagicMock()
        mock_extractor_class.return_value = mock_extractor
        mock_extractor.extract.return_value = {
            "landmarks": np.random.rand(21, 3),
            "finger_states": np.array([True, False, True, False, True]),
        }
        # Mock low confidence (0.3)
        mock_classifier.predict_proba.return_value = np.array([[0.3, 0.35, 0.35]])

        prediction, confidence = predict_image(mock_classifier, sample_image, threshold=0.5)

        assert prediction is None
        assert confidence is None

    @patch("handmotion.predict.FeatureExtractor")
    def test_predict_image_with_pil_image(self, mock_extractor_class, mock_classifier):
        """Test prediction with PIL Image instead of path."""
        mock_extractor = MagicMock()
        mock_extractor_class.return_value = mock_extractor
        mock_extractor.extract.return_value = {
            "landmarks": np.random.rand(21, 3),
            "finger_states": np.array([True, False, True, False, True]),
        }

        image = Image.new("RGB", (100, 100), color="red")
        prediction, confidence = predict_image(mock_classifier, image)

        assert prediction == "rock"
        assert isinstance(confidence, float)
        mock_extractor.extract.assert_called_once()


class TestPredictImageWithProba:
    """Test cases for predict_image_with_proba function."""

    @patch("handmotion.predict.FeatureExtractor")
    def test_predict_image_with_proba_success(
        self, mock_extractor_class, sample_image, mock_classifier
    ):
        """Test successful prediction with full probability distribution."""
        mock_extractor = MagicMock()
        mock_extractor_class.return_value = mock_extractor
        mock_extractor.extract.return_value = {
            "landmarks": np.random.rand(21, 3),
            "finger_states": np.array([True, False, True, False, True]),
        }
        mock_classifier.predict_proba.return_value = np.array([[0.1, 0.8, 0.1]])
        mock_classifier.label_encoder.classes_ = np.array(["rock", "paper", "scissors"])

        prediction, confidence_dict = predict_image_with_proba(mock_classifier, sample_image)

        assert prediction == "rock"
        assert isinstance(confidence_dict, dict)
        assert len(confidence_dict) == 3
        assert "rock" in confidence_dict
        assert "paper" in confidence_dict
        assert "scissors" in confidence_dict
        assert confidence_dict["paper"] == 0.8

    @patch("handmotion.predict.FeatureExtractor")
    def test_predict_image_with_proba_no_hand(
        self, mock_extractor_class, sample_image, mock_classifier
    ):
        """Test prediction with proba when no hand is detected."""
        mock_extractor = MagicMock()
        mock_extractor_class.return_value = mock_extractor
        mock_extractor.extract.return_value = None

        prediction, confidence_dict = predict_image_with_proba(mock_classifier, sample_image)

        assert prediction is None
        assert confidence_dict is None
