"""XGBoost baseline — hand-crafted cycle features → gradient-boosted classifier.

This is the "traditional ML" baseline from paper outline §6.2. Per cycle:
  features = [length, mean_amount, std_amount, time_span,
              mean_dt, value_imbalance, n_unique_nodes, max_t, min_t]
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np


def cycle_to_features(c: Dict) -> np.ndarray:
    """Convert one cycle dict to a fixed-length feature vector."""
    k = len(c["nodes"])
    amounts = np.array(c["amounts"], dtype=np.float64)
    times = np.array(c["times"], dtype=np.float64)
    mean_a = amounts.mean() if k > 0 else 0.0
    std_a = amounts.std() if k > 1 else 0.0
    imb = float(np.abs(amounts - mean_a).max() / max(mean_a, 1e-9)) if k > 0 else 0.0
    time_span = float(times.max() - times.min()) if k > 0 else 0.0
    mean_dt = float(np.diff(times).mean()) if k > 1 else 0.0
    return np.array([
        k, mean_a, std_a, time_span, mean_dt, imb,
        len(set(c["nodes"])),
        float(times.max()) if k > 0 else 0.0,
        float(times.min()) if k > 0 else 0.0,
    ], dtype=np.float32)


def batch_to_features(cycles: List[Dict]) -> np.ndarray:
    """Stack features for a batch of cycles."""
    if not cycles:
        return np.zeros((0, 9), dtype=np.float32)
    return np.stack([cycle_to_features(c) for c in cycles], axis=0)


# Feature name vector for interpretability
FEATURE_NAMES = [
    "length", "mean_amount", "std_amount", "time_span", "mean_dt",
    "value_imbalance", "n_unique_nodes", "max_time", "min_time",
]


class XGBoostCycleClassifier:
    """Wrapper around XGBoost for cycle-level classification.

    Falls back to sklearn GradientBoosting if xgboost is unavailable.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 4,
                 learning_rate: float = 0.1, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            import xgboost as xgb
            self.model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric="logloss",
                n_jobs=2,
            )
            self._backend = "xgboost"
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
            )
            self._backend = "sklearn"

    def fit(self, cycles: List[Dict], labels: np.ndarray):
        X = batch_to_features(cycles)
        self.model.fit(X, labels)
        return self

    def predict_proba(self, cycles: List[Dict]) -> np.ndarray:
        if not cycles:
            return np.zeros((0, 2))
        X = batch_to_features(cycles)
        return self.model.predict_proba(X)

    def predict(self, cycles: List[Dict]) -> np.ndarray:
        X = batch_to_features(cycles)
        return self.model.predict(X)

    @property
    def feature_importances(self) -> np.ndarray:
        return getattr(self.model, "feature_importances_", np.zeros(9))