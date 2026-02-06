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

"""Feature extraction from hand images using MediaPipe."""

from typing import Optional, TypedDict

import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandFeatures(TypedDict):
    landmarks: np.ndarray
    finger_states: np.ndarray
    handedness: Optional[str]


class FeatureExtractor:
    """Extract hand landmarks and finger states from images."""

    # MediaPipe hand landmark indices
    WRIST = 0
    THUMB_TIP = 4
    THUMB_IP = 3
    INDEX_TIP = 8
    INDEX_MCP = 5
    MIDDLE_TIP = 12
    MIDDLE_FINGER_MCP = 9
    MIDDLE_MCP = 9
    RING_TIP = 16
    RING_MCP = 13
    PINKY_TIP = 20
    PINKY_MCP = 17

    def __init__(self, model_path=None):
        """
        Initialize MediaPipe hand landmarker.

        Args:
            model_path: Path to hand landmarker model file (.task).
                       If None, uses bundled model from mediapipe package.
        """
        from pathlib import Path

        if model_path is None:
            # Try to use bundled model or download default
            # Check if model exists in project models directory
            project_root = Path(__file__).parent.parent.parent
            model_file = project_root / "models" / "hand_landmarker.task"
            if model_file.exists():
                model_path = str(model_file)
            else:
                # Use mediapipe's bundled model path
                import mediapipe

                mediapipe_path = Path(mediapipe.__file__).parent
                bundled_model = mediapipe_path / "tasks" / "models" / "hand_landmarker.task"
                if bundled_model.exists():
                    model_path = str(bundled_model)
                else:
                    raise FileNotFoundError(
                        "Hand landmarker model not found. "
                        "Please download from: "
                        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                    )

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hand_landmarker: vision.HandLandmarker | None = (
            vision.HandLandmarker.create_from_options(options)
        )

    def close(self):
        """Close MediaPipe hand landmarker and release resources."""
        if hasattr(self, "hand_landmarker") and self.hand_landmarker is not None:
            try:
                self.hand_landmarker.close()
            except Exception:
                pass  # Ignore errors during cleanup
            self.hand_landmarker = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()

    def extract(self, image, image_format="rgb") -> Optional[HandFeatures]:
        """
        Extract features from image.

        Args:
            image: Image as numpy array or PIL Image.
            image_format: 'rgb' or 'bgr'. Only used for numpy arrays.
                         PIL Images are always treated as RGB.

        Returns:
            Dict with 'landmarks', 'finger_states', 'handedness', or None if no hand detected.
        """
        # Convert to RGB numpy array if needed
        if isinstance(image, np.ndarray) and len(image.shape) == 3:
            if image_format == "bgr":
                try:
                    import cv2

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                except ImportError as err:
                    raise ImportError(
                        "OpenCV (cv2) is required for BGR to RGB conversion. "
                        "Install with: pip install opencv-python"
                    ) from err
            # Convert numpy array to MediaPipe Image
            mp_image = Image(image_format=ImageFormat.SRGB, data=image)
        else:
            # PIL Image - convert to numpy array first
            if hasattr(image, "mode") and image.mode != "RGB":
                image = image.convert("RGB")
            image_array = np.array(image)
            mp_image = Image(image_format=ImageFormat.SRGB, data=image_array)

        if self.hand_landmarker is None:
            raise RuntimeError(
                "FeatureExtractor is closed. Create a new FeatureExtractor instance to extract "
                "features."
            )

        # Detect hand landmarks
        detection_result = self.hand_landmarker.detect(mp_image)

        if not detection_result.hand_landmarks:
            return None

        # Get first hand (we only process one hand)
        hand_landmarks = detection_result.hand_landmarks[0]
        handedness = None
        if detection_result.handedness and len(detection_result.handedness) > 0:
            # Handedness is List[List[Category]] - first list is per hand, second is categories
            hand_categories = detection_result.handedness[0]
            if hand_categories and len(hand_categories) > 0:
                handedness = hand_categories[0].category_name

        # Convert landmarks to numpy array
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])

        # Normalize landmarks
        normalized_landmarks = self.normalize(landmarks)

        # Get finger states
        finger_states = self._get_finger_states(normalized_landmarks)

        return {
            "landmarks": normalized_landmarks,
            "finger_states": finger_states,
            "handedness": handedness,
        }

    def normalize(self, landmarks):
        """
        Normalize landmarks to wrist-centered coordinates.

        Args:
            landmarks: Array of shape (21, 3) with x, y, z coordinates

        Returns:
            Normalized landmarks array
        """
        wrist = landmarks[self.WRIST].copy()

        # Translate to wrist-centered
        normalized = landmarks - wrist

        # Scale by distance from wrist to middle finger MCP joint
        # to normalize for hand size
        scale = np.linalg.norm(normalized[self.MIDDLE_FINGER_MCP])
        if scale > 0:
            normalized = normalized / scale

        return normalized

    def _get_finger_states(self, landmarks):
        """
        Encode finger extended/curled states.

        Args:
            landmarks: Normalized landmarks array of shape (21, 3)

        Returns:
            Array of 5 booleans: [thumb, index, middle, ring, pinky]
            True = extended, False = curled
        """
        finger_states = []

        # Thumb: extended if tip is further from wrist than IP joint
        thumb_tip_dist = np.linalg.norm(landmarks[self.THUMB_TIP] - landmarks[self.WRIST])
        thumb_ip_dist = np.linalg.norm(landmarks[self.THUMB_IP] - landmarks[self.WRIST])
        finger_states.append(thumb_tip_dist > thumb_ip_dist)

        # Other fingers: extended if tip is above MCP (higher y value in normalized coords)
        for tip_idx, mcp_idx in [
            (self.INDEX_TIP, self.INDEX_MCP),
            (self.MIDDLE_TIP, self.MIDDLE_MCP),
            (self.RING_TIP, self.RING_MCP),
            (self.PINKY_TIP, self.PINKY_MCP),
        ]:
            # In MediaPipe normalized coordinates, y increases downward
            # So finger is extended if tip y < mcp y (tip is "above" mcp)
            finger_states.append(landmarks[tip_idx][1] < landmarks[mcp_idx][1])

        return np.array(finger_states, dtype=bool)

    @staticmethod
    def format_features(features: Optional[HandFeatures]) -> str:
        """
        Format extracted features as a human-readable string for debugging.

        Args:
            features: HandFeatures dict or None

        Returns:
            Formatted string representation
        """
        if features is None:
            return "No hand detected"

        finger_names = ["thumb", "index", "middle", "ring", "pinky"]
        finger_states_str = ", ".join(
            f"{name}: {'extended' if state else 'curled'}"
            for name, state in zip(finger_names, features["finger_states"], strict=False)
        )

        handedness_str = features["handedness"] or "unknown"
        landmarks_shape = features["landmarks"].shape

        return (
            f"Handedness: {handedness_str}\n"
            f"Finger states: {finger_states_str}\n"
            f"Landmarks shape: {landmarks_shape}"
        )
