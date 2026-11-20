"""Small, local classifiers trained from samples collected by the volunteer."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from object_features import extract_object_features


class LiveClassifier:
    """An SVC wrapper for multi-class object, hand, or face features."""

    def __init__(
        self,
        kind: str,
        minimum_per_class: int = 10,
        required_classes: int | None = None,
        minimum_classes: int = 2,
    ) -> None:
        self.kind = kind
        self.minimum_per_class = minimum_per_class
        self.required_classes = required_classes
        self.minimum_classes = minimum_classes
        self.samples: dict[str, list[np.ndarray]] = {}
        self.pipeline: Pipeline | None = None
        self.labels: list[str] = []

    @property
    def is_trained(self) -> bool:
        return self.pipeline is not None

    def add_feature(self, label: str, feature: np.ndarray, max_samples: int = 300) -> None:
        label = label.strip()
        if not label:
            raise ValueError("Nama kelas tidak boleh kosong.")
        values = self.samples.setdefault(label, [])
        if len(values) < max_samples:
            values.append(np.asarray(feature, dtype=np.float32).copy())

    def add_object_frame(self, label: str, frame: np.ndarray) -> None:
        self.add_feature(label, extract_object_features(frame), max_samples=250)

    def clear_samples(self) -> None:
        self.samples.clear()
        self.pipeline = None
        self.labels = []

    def counts(self) -> dict[str, int]:
        return {label: len(values) for label, values in self.samples.items()}

    def train(self) -> tuple[list[str], dict[str, int]]:
        labels = list(self.samples.keys())
        if self.required_classes is not None and len(labels) != self.required_classes:
            raise ValueError(
                f"Model ini membutuhkan tepat {self.required_classes} kelas berbeda."
            )
        if len(labels) < self.minimum_classes:
            raise ValueError(
                f"Tambahkan minimal {self.minimum_classes} kelas yang berbeda sebelum training."
            )

        counts = self.counts()
        too_small = [name for name, count in counts.items() if count < self.minimum_per_class]
        if too_small:
            details = ", ".join(f"{name}: {counts[name]} sampel" for name in too_small)
            raise ValueError(
                f"Setiap kelas membutuhkan minimal {self.minimum_per_class} sampel. "
                f"Tambahkan sampel untuk: {details}."
            )

        x_values: list[np.ndarray] = []
        y_values: list[str] = []
        for label in labels:
            for feature in self.samples[label]:
                x_values.append(feature)
                y_values.append(label)

        x = np.vstack(x_values).astype(np.float32)
        y = np.asarray(y_values)

        self.pipeline = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=4.0,
                        gamma="scale",
                        probability=True,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
        self.pipeline.fit(x, y)
        self.labels = labels
        return labels, counts

    def predict_feature(self, feature: np.ndarray) -> tuple[str, float] | None:
        if self.pipeline is None:
            return None
        x = np.asarray(feature, dtype=np.float32).reshape(1, -1)
        probabilities = self.pipeline.predict_proba(x)[0]
        best_index = int(np.argmax(probabilities))
        label = str(self.pipeline.classes_[best_index])
        confidence = float(probabilities[best_index])
        return label, confidence

    def predict_object_frame(self, frame: np.ndarray) -> tuple[str, float] | None:
        return self.predict_feature(extract_object_features(frame))

    def save(self, path: Path) -> None:
        if self.pipeline is None:
            raise ValueError("Model belum dilatih.")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"kind": self.kind, "pipeline": self.pipeline, "labels": self.labels},
            path,
        )

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        saved = joblib.load(path)
        if saved.get("kind") != self.kind:
            return False
        self.pipeline = saved["pipeline"]
        self.labels = list(saved.get("labels", []))
        return True
