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

"""Process image folders and extract features."""

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from handmotion.data.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class DataProcessor:
    """Process images from folders and save extracted features."""

    def __init__(self):
        """Initialize feature extractor."""
        self.extractor = FeatureExtractor()

    def process_folder(self, folder_path, label):
        """
        Process all images in folder and extract features.

        Args:
            folder_path: Path to folder containing images
            label: Label for images in this folder

        Returns:
            List of (features, label) tuples
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder does not exist: {folder_path}")

        # Supported image extensions
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

        # Find all image files
        image_files = [
            f for f in folder.iterdir() if f.suffix.lower() in image_extensions and f.is_file()
        ]

        if not image_files:
            logger.warning(f"No image files found in {folder_path}")
            return []

        results = []
        failed_images = []

        for image_file in image_files:
            try:
                # Load image using PIL (always RGB)
                image = Image.open(image_file)
                # Convert to RGB if needed (handles RGBA, L, etc.)
                if image.mode != "RGB":
                    image = image.convert("RGB")

                # Extract features
                features = self.extractor.extract(image, image_format="rgb")

                if features is None:
                    failed_images.append(str(image_file))
                    logger.debug(f"No hand detected in {image_file}")
                    continue

                results.append((features, label))

            except Exception as e:
                failed_images.append(str(image_file))
                logger.warning(f"Failed to process {image_file}: {e}")

        if failed_images:
            logger.info(f"Failed to process {len(failed_images)} images: {failed_images}")

        logger.info(f"Processed {len(results)}/{len(image_files)} images from {folder_path}")

        return results

    def save(self, data, output_path):
        """
        Save processed data to npz file.

        Args:
            data: List of (features, label) tuples
            output_path: Path to save the npz file
        """
        if not data:
            raise ValueError("Cannot save empty data")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Separate features and labels
        features_list = [item[0] for item in data]
        labels_list = [item[1] for item in data]

        # Convert to arrays for training
        # Stack landmarks and finger_states
        landmarks_array = np.array([f["landmarks"] for f in features_list])
        finger_states_array = np.array([f["finger_states"] for f in features_list])
        handedness_list = [f["handedness"] for f in features_list]

        # Save as npz
        np.savez(
            output_path,
            # Array format for training
            landmarks=landmarks_array,
            finger_states=finger_states_array,
            handedness=handedness_list,
            labels=labels_list,
            # Keep tuple format for debugging (as metadata)
            labels_list=labels_list,
        )

        logger.info(f"Saved {len(data)} samples to {output_path}")


def main():
    """Main function to process data."""
    processor = DataProcessor()
    cwd = Path.cwd()

    results = processor.process_folder(cwd / "data" / "raw" / "rps" / "rock", "rock")
    results.extend(processor.process_folder(cwd / "data" / "raw" / "rps" / "paper", "paper"))
    results.extend(processor.process_folder(cwd / "data" / "raw" / "rps" / "scissors", "scissors"))
    processor.save(results, cwd / "data" / "processed" / "rps" / "data.npz")
    print(f"Processed {len(results)} samples")


if __name__ == "__main__":
    main()
