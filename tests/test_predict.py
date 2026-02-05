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

from unittest.mock import MagicMock

import numpy as np

from handmotion.predict import predict_image


class TestPredictImage:
    """Test cases for predict_image function."""

    def test_predict_image_success(self, sample_image, mock_feature_extractor, mock_classifier):
        """Test successful prediction."""
        prediction, confidence = predict_image(
            mock_classifier, mock_feature_extractor, sample_image
        )

        assert prediction == "rock"
        assert "rock" in confidence
        assert confidence["rock"] == 0.9
        mock_feature_extractor.extract.assert_called_once()

    def test_predict_image_no_hand_detected(self, sample_image):
        """Test prediction when no hand is detected."""

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = None
        mock_classifier = MagicMock()

        prediction, confidence = predict_image(mock_classifier, mock_extractor, sample_image)

        assert prediction is None
        assert confidence is None
        mock_extractor.extract.assert_called_once()
        mock_classifier.predict.assert_not_called()

    def test_predict_image_confidence_dict(self, sample_image, mock_feature_extractor):
        """Test that confidence dict contains all classes."""

        mock_classifier = MagicMock()
        mock_classifier._prepare_features.return_value = np.random.rand(1, 68)
        mock_classifier.predict.return_value = np.array(["paper"])
        mock_classifier.predict_proba.return_value = np.array([[0.1, 0.8, 0.1]])
        mock_classifier.label_encoder.classes_ = np.array(["rock", "paper", "scissors"])

        prediction, confidence = predict_image(
            mock_classifier, mock_feature_extractor, sample_image
        )

        assert prediction == "paper"
        assert len(confidence) == 3
        assert "rock" in confidence
        assert "paper" in confidence
        assert "scissors" in confidence
        assert confidence["paper"] == 0.8
