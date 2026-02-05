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

"""Training script for hand gesture classifier.

Examples:
    # Default: use existing data.npz, save to models/rps_classifier.joblib
    python -m handmotion.train

    # Reprocess raw data first
    python -m handmotion.train --reprocess

    # Custom paths
    python -m handmotion.train \\
        --data data/processed/rps/data.npz \\
        --model models/rps_classifier.joblib
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from handmotion.data.processor import DataProcessor
from handmotion.model import HandGestureClassifier

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate_model(classifier, x_test, y_test):
    """
    Evaluate model on test set.

    Args:
        classifier: Trained HandGestureClassifier
        x_test: Test features
        y_test: Test labels (encoded)

    Returns:
        Dictionary with evaluation metrics
    """
    # Predict on test set
    predictions_encoded = classifier.classifier.predict(x_test)
    predictions = classifier.label_encoder.inverse_transform(predictions_encoded)
    y_test_labels = classifier.label_encoder.inverse_transform(y_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions_encoded)
    cm = confusion_matrix(y_test, predictions_encoded)
    report = classification_report(y_test_labels, predictions, output_dict=True)

    metrics = {
        "accuracy": float(accuracy),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    return metrics


def print_evaluation(metrics):
    """
    Print evaluation metrics to console.

    Args:
        metrics: Dictionary with evaluation metrics
    """
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])

    print("\nClassification Report:")
    report = metrics["classification_report"]
    for label in ["rock", "paper", "scissors"]:
        if label in report:
            print(f"\n{label}:")
            print(f"  Precision: {report[label]['precision']:.4f}")
            print(f"  Recall:    {report[label]['recall']:.4f}")
            print(f"  F1-score:  {report[label]['f1-score']:.4f}")

    print("\nMacro Avg:")
    print(f"  Precision: {report['macro avg']['precision']:.4f}")
    print(f"  Recall:    {report['macro avg']['recall']:.4f}")
    print(f"  F1-score:  {report['macro avg']['f1-score']:.4f}")

    print("=" * 60 + "\n")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train hand gesture classifier")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/rps/data.npz",
        help="Path to processed data file (default: data/processed/rps/data.npz)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/rps_classifier.joblib",
        help="Path to save trained model (default: models/rps_classifier.joblib)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="models/metrics.json",
        help="Path to save evaluation metrics (default: models/metrics.json)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for testing (default: 0.2)",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Force reprocessing of raw data even if processed data exists",
    )

    args = parser.parse_args()

    # Setup paths
    data_path = Path(args.data)
    model_path = Path(args.model)
    metrics_path = Path(args.metrics)

    # Optionally reprocess data if requested or if data doesn't exist
    if args.reprocess or not data_path.exists():
        if args.reprocess:
            logger.info("Reprocessing data as requested...")
        else:
            logger.info(f"Data file not found at {data_path}, processing raw data...")

        processor = DataProcessor()
        cwd = Path.cwd()

        results = processor.process_folder(cwd / "data" / "raw" / "rps" / "rock", "rock")
        results.extend(processor.process_folder(cwd / "data" / "raw" / "rps" / "paper", "paper"))
        results.extend(
            processor.process_folder(cwd / "data" / "raw" / "rps" / "scissors", "scissors")
        )
        processor.save(results, data_path)
        logger.info(f"Processed {len(results)} samples")
    else:
        logger.info(f"Using existing data file: {data_path}")

    # Load data
    logger.info("Loading data...")
    classifier = HandGestureClassifier()
    features, labels = classifier.load_data(data_path)
    logger.info(f"Loaded {len(features)} samples with {len(np.unique(labels))} classes")

    # Train model
    logger.info("Training classifier...")
    x_train, x_test, _y_train, y_test = classifier.train(features, labels, test_size=args.test_size)
    logger.info(f"Training set: {len(x_train)} samples, Test set: {len(x_test)} samples")

    # Evaluate model
    logger.info("Evaluating model...")
    metrics = evaluate_model(classifier, x_test, y_test)
    print_evaluation(metrics)

    # Save model
    logger.info(f"Saving model to {model_path}...")
    classifier.save(model_path)

    # Save metrics
    logger.info(f"Saving metrics to {metrics_path}...")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
