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

"""Unit tests for data processor."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from handmotion.data.processor import DataProcessor


class TestDataProcessor:
    """Test cases for DataProcessor."""

    def test_init(self):
        """Test DataProcessor initialization."""
        # Mock FeatureExtractor to avoid slow MediaPipe initialization
        with patch("handmotion.data.processor.FeatureExtractor"):
            processor = DataProcessor()
            assert processor.extractor is not None

    def test_process_folder_nonexistent(self):
        """Test process_folder raises error for nonexistent folder."""
        with patch("handmotion.data.processor.FeatureExtractor"):
            processor = DataProcessor()
            with pytest.raises(ValueError, match="Folder does not exist"):
                processor.process_folder("/nonexistent/path", "test")

    def test_process_folder_empty(self, tmp_path):
        """Test process_folder returns empty list for folder with no images."""
        with patch("handmotion.data.processor.FeatureExtractor"):
            processor = DataProcessor()
            result = processor.process_folder(tmp_path, "test")
            assert result == []

    def test_process_folder_with_images(self, tmp_path):
        """Test process_folder processes images successfully."""
        with patch("handmotion.data.processor.FeatureExtractor"):
            processor = DataProcessor()

        # Create test images
        image1 = Image.new("RGB", (100, 100), color="red")
        image2 = Image.new("RGB", (100, 100), color="blue")
        image1.save(tmp_path / "test1.jpg")
        image2.save(tmp_path / "test2.png")

        # Mock the extractor to return features
        mock_features = {
            "landmarks": np.random.rand(21, 3),
            "finger_states": np.array([True, True, False, False, True]),
            "handedness": "Right",
        }

        with patch.object(processor.extractor, "extract", return_value=mock_features):
            result = processor.process_folder(tmp_path, "rock")

            assert len(result) == 2
            for features, label in result:
                assert label == "rock"
                assert features == mock_features

    def test_process_folder_skips_no_hand(self, tmp_path):
        """Test process_folder skips images with no hand detected."""
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = None

        with patch("handmotion.data.processor.FeatureExtractor", return_value=mock_extractor):
            processor = DataProcessor()

            image = Image.new("RGB", (100, 100))
            image.save(tmp_path / "test.jpg")

            result = processor.process_folder(tmp_path, "paper")
            assert result == []

    def test_process_folder_handles_errors(self, tmp_path):
        """Test process_folder handles processing errors gracefully."""
        # Create a mock extractor instance
        mock_extractor = MagicMock()
        mock_extractor.extract.side_effect = Exception("Test error")

        # Mock FeatureExtractor class to return our mock instance
        with patch("handmotion.data.processor.FeatureExtractor", return_value=mock_extractor):
            processor = DataProcessor()

            image = Image.new("RGB", (100, 100))
            image.save(tmp_path / "test.jpg")

            result = processor.process_folder(tmp_path, "scissors")
            assert result == []

    def test_save(self, tmp_path):
        """Test save writes npz file correctly."""
        # Mock FeatureExtractor to avoid MediaPipe initialization
        with patch("handmotion.data.processor.FeatureExtractor"):
            processor = DataProcessor()

        # Create test data
        features1 = {
            "landmarks": np.array([[0.1, 0.2, 0.3]] * 21),
            "finger_states": np.array([True, False, True, False, True]),
            "handedness": "Right",
        }
        features2 = {
            "landmarks": np.array([[0.4, 0.5, 0.6]] * 21),
            "finger_states": np.array([False, True, False, True, False]),
            "handedness": "Left",
        }

        data = [(features1, "rock"), (features2, "paper")]
        output_path = tmp_path / "output.npz"

        processor.save(data, output_path)

        assert output_path.exists()

        # Load and verify
        loaded = np.load(output_path, allow_pickle=True)
        assert "landmarks" in loaded
        assert "finger_states" in loaded
        assert "handedness" in loaded
        assert "labels" in loaded

        assert loaded["landmarks"].shape == (2, 21, 3)
        assert loaded["finger_states"].shape == (2, 5)
        assert len(loaded["handedness"]) == 2
        assert len(loaded["labels"]) == 2
        assert loaded["labels"][0] == "rock"
        assert loaded["labels"][1] == "paper"

    def test_save_empty_data(self):
        """Test save raises error for empty data."""
        # Mock FeatureExtractor to avoid MediaPipe initialization
        with patch("handmotion.data.processor.FeatureExtractor"):
            processor = DataProcessor()
            with pytest.raises(ValueError, match="Cannot save empty data"):
                processor.save([], "output.npz")

    def test_save_creates_parent_dirs(self, tmp_path):
        """Test save creates parent directories if needed."""
        # Mock FeatureExtractor to avoid MediaPipe initialization
        with patch("handmotion.data.processor.FeatureExtractor"):
            processor = DataProcessor()

        features = {
            "landmarks": np.random.rand(21, 3),
            "finger_states": np.array([True] * 5),
            "handedness": "Right",
        }
        data = [(features, "test")]

        # Create nested path
        output_path = tmp_path / "nested" / "dir" / "output.npz"
        processor.save(data, output_path)

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_process_folder_multiple_formats(self, tmp_path):
        """Test process_folder handles multiple image formats."""
        with patch("handmotion.data.processor.FeatureExtractor"):
            processor = DataProcessor()

        # Create images in different formats
        Image.new("RGB", (50, 50)).save(tmp_path / "test1.jpg")
        Image.new("RGB", (50, 50)).save(tmp_path / "test2.png")
        Image.new("RGB", (50, 50)).save(tmp_path / "test3.bmp")
        # Create a non-image file (should be ignored)
        (tmp_path / "test.txt").write_text("not an image")

        mock_features = {
            "landmarks": np.random.rand(21, 3),
            "finger_states": np.array([True] * 5),
            "handedness": "Right",
        }

        with patch.object(processor.extractor, "extract", return_value=mock_features):
            result = processor.process_folder(tmp_path, "rock")

            # Should process 3 images, ignore txt file
            assert len(result) == 3

    @pytest.mark.slow
    def test_process_folder_with_real_images(self):
        """Test process_folder with actual RPS images from test data folder."""
        processor = DataProcessor()

        # Use test images from tests/data/rps
        test_data_dir = Path(__file__).parent / "data" / "rps"

        # Test with rock images
        rock_dir = test_data_dir / "rock"
        if rock_dir.exists() and any(rock_dir.glob("*.jpg")):
            # Use available test images
            available_images = list(rock_dir.glob("*.jpg"))

            if available_images:
                result = processor.process_folder(rock_dir, "rock")

                # Should process at least some images (may skip if no hand detected)
                assert isinstance(result, list)
                # Verify structure of results
                for features, label in result:
                    assert label == "rock"
                    assert "landmarks" in features
                    assert "finger_states" in features
                    assert "handedness" in features
                    assert features["landmarks"].shape == (21, 3)
                    assert features["finger_states"].shape == (5,)

    @pytest.mark.slow
    def test_process_folder_multiple_labels(self):
        """Test processing multiple folders with different labels."""
        processor = DataProcessor()

        # Use test images from tests/data/rps
        test_data_dir = Path(__file__).parent / "data" / "rps"

        all_results = []
        labels_to_test = ["rock", "paper", "scissors"]

        for label in labels_to_test:
            label_dir = test_data_dir / label
            if label_dir.exists() and any(label_dir.glob("*.jpg")):
                result = processor.process_folder(label_dir, label)
                all_results.extend(result)

        # If we got any results, verify they have correct labels
        if all_results:
            labels_found = {label for _, label in all_results}
            assert labels_found.issubset(set(labels_to_test))
            # Verify structure of results
            for features, _label in all_results:
                assert "landmarks" in features
                assert "finger_states" in features
                assert features["landmarks"].shape == (21, 3)
                assert features["finger_states"].shape == (5,)
