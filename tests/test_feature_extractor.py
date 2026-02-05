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

"""Unit tests for feature extractor."""

from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from handmotion.data.feature_extractor import FeatureExtractor, HandFeatures


class TestFeatureExtractor:
    """Test cases for FeatureExtractor."""

    def test_init(self):
        """Test FeatureExtractor initialization."""
        extractor = FeatureExtractor()
        assert extractor.hand_landmarker is not None

    def test_extract_no_hand_detected(self):
        """Test extract returns None when no hand is detected."""
        extractor = FeatureExtractor()
        # Create a mock result with no hand landmarks
        mock_result = MagicMock()
        mock_result.hand_landmarks = None
        mock_result.handedness = None

        with patch.object(extractor.hand_landmarker, "detect", return_value=mock_result):
            image = Image.new("RGB", (100, 100))
            result = extractor.extract(image)
            assert result is None

    def test_extract_with_hand(self):
        """Test extract returns features when hand is detected."""
        extractor = FeatureExtractor()

        # Create mock landmarks for new API
        mock_landmark = MagicMock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.5
        mock_landmark.z = 0.0

        # New API: hand_landmarks is a list of landmarks directly
        mock_hand_landmarks = [mock_landmark] * 21

        # New API: handedness is List[List[Category]] - first list is per hand
        mock_category = MagicMock()
        mock_category.category_name = "Right"
        # First hand's categories list
        mock_hand_categories = [mock_category]
        # List of hands (each hand has a list of categories)
        mock_handedness = [mock_hand_categories]

        mock_result = MagicMock()
        mock_result.hand_landmarks = [mock_hand_landmarks]
        mock_result.handedness = mock_handedness

        with patch.object(extractor.hand_landmarker, "detect", return_value=mock_result):
            image = Image.new("RGB", (100, 100))
            result = extractor.extract(image)

            assert result is not None
            assert "landmarks" in result
            assert "finger_states" in result
            assert "handedness" in result
            assert result["handedness"] == "Right"
            assert result["landmarks"].shape == (21, 3)
            assert result["finger_states"].shape == (5,)

    def test_extract_without_handedness(self):
        """Test extract handles missing handedness."""
        extractor = FeatureExtractor()

        mock_landmark = MagicMock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.5
        mock_landmark.z = 0.0

        mock_hand_landmarks = [mock_landmark] * 21

        mock_result = MagicMock()
        mock_result.hand_landmarks = [mock_hand_landmarks]
        mock_result.handedness = None

        with patch.object(extractor.hand_landmarker, "detect", return_value=mock_result):
            image = Image.new("RGB", (100, 100))
            result = extractor.extract(image)

            assert result is not None
            assert result["handedness"] is None

    def test_normalize(self):
        """Test landmark normalization."""
        extractor = FeatureExtractor()

        # Create test landmarks (21 points, 3D)
        landmarks = np.random.rand(21, 3)
        # Set wrist at origin for easier testing
        landmarks[0] = [0.5, 0.5, 0.0]

        normalized = extractor.normalize(landmarks)

        # Wrist should be at origin after normalization
        assert np.allclose(normalized[0], [0, 0, 0], atol=1e-6)

        # Check that landmarks are scaled
        assert normalized.shape == (21, 3)

    def test_normalize_zero_scale(self):
        """Test normalize handles zero scale edge case."""
        extractor = FeatureExtractor()

        # Create landmarks where middle finger MCP is at wrist (zero scale)
        landmarks = np.zeros((21, 3))
        landmarks[0] = [0.5, 0.5, 0.0]  # Wrist
        landmarks[9] = [0.5, 0.5, 0.0]  # Middle finger MCP at same position

        normalized = extractor.normalize(landmarks)

        # Should not crash, but scale won't be applied
        assert normalized.shape == (21, 3)

    def test_get_finger_states(self):
        """Test finger state detection."""
        extractor = FeatureExtractor()

        # Create normalized landmarks with extended fingers
        landmarks = np.zeros((21, 3))
        landmarks[0] = [0, 0, 0]  # Wrist at origin

        # Set finger tips above MCPs (extended)
        landmarks[8, 1] = 0.1  # Index tip (y < mcp y means extended)
        landmarks[5, 1] = 0.2  # Index MCP
        landmarks[12, 1] = 0.1  # Middle tip
        landmarks[9, 1] = 0.2  # Middle MCP
        landmarks[16, 1] = 0.1  # Ring tip
        landmarks[13, 1] = 0.2  # Ring MCP
        landmarks[20, 1] = 0.1  # Pinky tip
        landmarks[17, 1] = 0.2  # Pinky MCP

        # Thumb: tip further from wrist than IP
        landmarks[4] = [0.3, 0.0, 0.0]  # Thumb tip
        landmarks[3] = [0.2, 0.0, 0.0]  # Thumb IP

        finger_states = extractor._get_finger_states(landmarks)

        assert finger_states.shape == (5,)
        assert finger_states.dtype == bool
        # All fingers should be extended in this setup
        assert np.all(finger_states)

    def test_format_features_none(self):
        """Test format_features with None input."""
        result = FeatureExtractor.format_features(None)
        assert result == "No hand detected"

    def test_format_features(self):
        """Test format_features with valid features."""
        features: HandFeatures = {
            "landmarks": np.random.rand(21, 3),
            "finger_states": np.array([True, True, False, False, True]),
            "handedness": "Right",
        }

        result = FeatureExtractor.format_features(features)
        assert "Right" in result
        assert "thumb" in result
        assert "extended" in result
        assert "curled" in result

    def test_format_features_no_handedness(self):
        """Test format_features with None handedness."""
        features: HandFeatures = {
            "landmarks": np.random.rand(21, 3),
            "finger_states": np.array([True, False, False, False, False]),
            "handedness": None,
        }

        result = FeatureExtractor.format_features(features)
        assert "unknown" in result
