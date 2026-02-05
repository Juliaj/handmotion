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

"""Shared fixtures and utilities for tests."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def sample_landmarks():
    """Create sample landmarks array (21, 3)."""
    return np.random.rand(21, 3)


@pytest.fixture
def sample_finger_states():
    """Create sample finger states array (5,)."""
    return np.array([True, False, True, False, True])


@pytest.fixture
def sample_hand_features(sample_landmarks, sample_finger_states):
    """Create sample hand features dictionary."""
    return {
        "landmarks": sample_landmarks,
        "finger_states": sample_finger_states,
        "handedness": "Right",
    }


@pytest.fixture
def sample_image(tmp_path):
    """Create a sample test image file."""
    image_path = tmp_path / "test.jpg"
    image = Image.new("RGB", (100, 100), color="red")
    image.save(image_path)
    return image_path


@pytest.fixture
def mock_feature_extractor(sample_hand_features):
    """Create a mock feature extractor."""
    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = sample_hand_features
    return mock_extractor


@pytest.fixture
def mock_classifier():
    """Create a mock classifier with default setup."""
    mock_clf = MagicMock()
    mock_clf._prepare_features.return_value = np.random.rand(1, 68)
    mock_clf.predict.return_value = np.array(["rock"])
    mock_clf.predict_proba.return_value = np.array([[0.05, 0.9, 0.05]])
    mock_clf.label_encoder.classes_ = np.array(["paper", "rock", "scissors"])
    return mock_clf


def create_test_npz(tmp_path, n_samples=10):
    """Helper to create test npz file."""
    landmarks = np.random.rand(n_samples, 21, 3)
    finger_states = np.random.rand(n_samples, 5)
    labels = (["rock", "paper", "scissors"] * ((n_samples // 3) + 1))[:n_samples]

    data_path = tmp_path / "test_data.npz"
    np.savez(data_path, landmarks=landmarks, finger_states=finger_states, labels=labels)
    return data_path, landmarks, finger_states, labels
