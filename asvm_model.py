from io import BytesIO

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

FEATURE_COLUMNS = ("lag1", "lag2", "roll3", "trend", "hotspot", "sin_m", "cos_m")
CLASSES = np.array([0, 1], dtype=int)
MODEL_VERSION = "asvm-margin-v2"


def classification_metrics(decision_scores, labels):
    actual = np.asarray(labels, dtype=int).reshape(-1)
    scores = np.asarray(decision_scores, dtype=float).reshape(-1)
    if len(actual) != len(scores):
        raise ValueError("Decision scores and labels must have the same length.")
    classified = (scores > 0).astype(int)
    matrix = confusion_matrix(actual, classified, labels=CLASSES)
    return {
        "evaluated": int(len(actual)),
        "true_negative": int(matrix[0, 0]),
        "false_positive": int(matrix[0, 1]),
        "false_negative": int(matrix[1, 0]),
        "true_positive": int(matrix[1, 1]),
        "precision": float(precision_score(actual, classified, zero_division=0)),
        "recall": float(recall_score(actual, classified, zero_division=0)),
        "f1": float(f1_score(actual, classified, zero_division=0)),
    }


class AdaptiveSVM:
    """Incremental linear SVM with warm-up-fitted Platt calibration."""

    def __init__(self):
        self.classifier = None
        self.scaler = StandardScaler()
        self.calibrator = None
        self.class_weight_dict = None
        self.calibration_scores = []
        self.calibration_labels = []
        self.samples_seen = 0

    @staticmethod
    def _as_features(values):
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != len(FEATURE_COLUMNS):
            raise ValueError(f"Expected a 2D feature matrix with {len(FEATURE_COLUMNS)} columns.")
        if not np.isfinite(array).all():
            raise ValueError("Feature matrix contains missing or non-finite values.")
        return array

    @staticmethod
    def _balanced_weights(labels):
        observed_classes = np.unique(labels)
        weights = compute_class_weight(class_weight="balanced", classes=observed_classes, y=labels)
        return {int(label): float(weight) for label, weight in zip(observed_classes, weights)}

    def warm_up(self, X, y):
        features = self._as_features(X)
        labels = np.asarray(y, dtype=int).reshape(-1)
        if len(features) != len(labels) or len(labels) == 0:
            raise ValueError("Warm-up features and labels must have the same non-zero length.")
        if not set(np.unique(labels)).issubset({0, 1}):
            raise ValueError("Outbreak labels must be binary (0 or 1).")
        if len(np.unique(labels)) != 2:
            raise ValueError("Warm-up data must contain both outbreak classes for balanced training and Platt calibration.")

        features = self.scaler.fit_transform(features)
        self.class_weight_dict = self._balanced_weights(labels)
        self.classifier = SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=1e-4,
            average=True,
            random_state=42,
        )
        self.classifier.partial_fit(
            features,
            labels,
            classes=CLASSES,
            sample_weight=self._sample_weights(labels),
        )
        self.samples_seen += len(labels)
        warmup_scores = self.classifier.decision_function(features)
        if not self.calibrate(warmup_scores, labels):
            raise ValueError("Warm-up data did not provide enough examples to fit Platt calibration.")

    def decision_function(self, X):
        if self.classifier is None:
            raise RuntimeError("The ASVM has not completed warm-up.")
        scaled_features = self.scaler.transform(self._as_features(X))
        return np.asarray(self.classifier.decision_function(scaled_features), dtype=float).reshape(-1)

    def predict_proba(self, X):
        if self.calibrator is None:
            raise RuntimeError("Platt calibration is not ready; more prequential samples are required.")
        scores = self.decision_function(X).reshape(-1, 1)
        positive_index = list(self.calibrator.classes_).index(1)
        probabilities = self.calibrator.predict_proba(scores)
        negative_index = list(self.calibrator.classes_).index(0)
        return np.column_stack((probabilities[:, negative_index], probabilities[:, positive_index]))

    def partial_fit(self, X, y):
        if self.classifier is None:
            raise RuntimeError("Call warm_up before partial_fit.")
        features = self._as_features(X)
        labels = np.asarray(y, dtype=int).reshape(-1)
        if len(features) != len(labels):
            raise ValueError("Features and labels must have the same length.")
        if not set(np.unique(labels)).issubset({0, 1}):
            raise ValueError("Outbreak labels must be binary (0 or 1).")
        scaled_features = self.scaler.transform(features)
        self.classifier.partial_fit(scaled_features, labels, sample_weight=self._sample_weights(labels))
        self.samples_seen += len(labels)

    def _sample_weights(self, labels):
        if self.class_weight_dict is None:
            raise RuntimeError("Balanced class weights are not initialized.")
        try:
            return np.asarray([self.class_weight_dict[int(label)] for label in labels], dtype=float)
        except KeyError as error:
            raise ValueError(f"Class {error.args[0]} was not represented in warm-up data.") from error

    def calibrate(self, decision_scores, labels):
        scores = np.asarray(decision_scores, dtype=float).reshape(-1)
        targets = np.asarray(labels, dtype=int).reshape(-1)
        if len(scores) != len(targets):
            raise ValueError("Calibration scores and labels must have the same length.")
        if len(scores) < 2 or len(set(targets)) < 2:
            return False

        calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        calibrator.fit(scores.reshape(-1, 1), targets)
        self.calibration_scores = scores.tolist()
        self.calibration_labels = targets.tolist()
        self.calibrator = calibrator
        return True

    @staticmethod
    def get_risk_category(probability):
        value = float(probability)
        if not 0 <= value <= 1:
            raise ValueError("Probability must be between 0 and 1.")
        if value >= 0.50:
            return "HIGH"
        if value >= 0.20:
            return "MEDIUM"
        return "LOW"

    def to_bytes(self):
        buffer = BytesIO()
        import joblib
        joblib.dump(self, buffer)
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, payload):
        import joblib
        model = joblib.load(BytesIO(payload))
        if not isinstance(model, cls):
            raise TypeError("Persisted model checkpoint is not an AdaptiveSVM instance.")
        return model
