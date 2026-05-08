"""
probe.py - Hallucination probe classifier (student-implemented).

The public API is kept compatible with solution.py and evaluate.py:
``fit``, ``fit_hyperparameters``, ``predict``, and ``predict_proba``.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


LOGREG_C = 0.01
N_BOOTSTRAP = 5
BOOTSTRAP_SEEDS = (0, 1, 7, 42, 123)
INNER_FOLDS = 5
RANDOM_STATE = 42


class HallucinationProbe(nn.Module):
    """Bootstrap ensemble of CPU logistic probes."""

    def __init__(self) -> None:
        super().__init__()
        self._scalers: list[StandardScaler] = []
        self._clfs: list[LogisticRegression] = []
        self._threshold: float = 0.5
        self.best_params_ = {
            "C": LOGREG_C,
            "class_weight": None,
            "n_bootstrap": N_BOOTSTRAP,
        }
        self.cv_score_: float | None = None
        self.oof_accuracy_: float | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward is intentionally unused; evaluation calls predict_proba()."""
        raise RuntimeError("HallucinationProbe uses sklearn; call predict_proba().")

    @staticmethod
    def _as_numpy(X: np.ndarray) -> np.ndarray:
        X_np = np.asarray(X, dtype=np.float32)
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)
        return X_np

    @staticmethod
    def _new_clf(seed: int = RANDOM_STATE) -> LogisticRegression:
        return LogisticRegression(
            C=LOGREG_C,
            solver="lbfgs",
            max_iter=5000,
            class_weight=None,
            random_state=seed,
        )

    @staticmethod
    def _fit_one(
        X: np.ndarray, y: np.ndarray, seed: int = RANDOM_STATE
    ) -> tuple[StandardScaler, LogisticRegression]:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf = HallucinationProbe._new_clf(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_scaled, y)
        return scaler, clf

    def _fit_bootstrap_ensemble(self, X: np.ndarray, y: np.ndarray) -> None:
        self._scalers = []
        self._clfs = []
        n_samples = len(y)

        for seed in BOOTSTRAP_SEEDS:
            rng = np.random.RandomState(seed)
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            if np.unique(y[idx]).size < 2:
                idx = np.arange(n_samples)
            scaler, clf = self._fit_one(X[idx], y[idx], seed=seed)
            self._scalers.append(scaler)
            self._clfs.append(clf)

    @staticmethod
    def _best_accuracy_threshold(probs: np.ndarray, y_true: np.ndarray) -> float:
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 501)]))
        best_threshold = 0.5
        best_accuracy = -1.0
        best_f1 = -1.0
        best_distance = float("inf")

        for threshold in candidates:
            y_pred = (probs >= threshold).astype(int)
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            distance = abs(float(threshold) - 0.5)

            if (
                accuracy > best_accuracy
                or (
                    np.isclose(accuracy, best_accuracy)
                    and (f1 > best_f1 or (np.isclose(f1, best_f1) and distance < best_distance))
                )
            ):
                best_accuracy = accuracy
                best_f1 = f1
                best_distance = distance
                best_threshold = float(threshold)

        return best_threshold

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X_np = self._as_numpy(X)
        y_int = np.asarray(y, dtype=int)

        class_counts = np.bincount(y_int, minlength=2)
        min_class_count = int(class_counts.min())
        n_splits = min(INNER_FOLDS, min_class_count)

        if n_splits >= 2:
            oof_probs = np.zeros(len(y_int), dtype=np.float64)
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=RANDOM_STATE,
            )
            for train_idx, val_idx in cv.split(X_np, y_int):
                scaler, clf = self._fit_one(X_np[train_idx], y_int[train_idx])
                oof_probs[val_idx] = clf.predict_proba(
                    scaler.transform(X_np[val_idx])
                )[:, 1]

            self._threshold = self._best_accuracy_threshold(oof_probs, y_int)
            oof_pred = (oof_probs >= self._threshold).astype(int)
            self.oof_accuracy_ = float(accuracy_score(y_int, oof_pred))
            self.cv_score_ = self.oof_accuracy_
        else:
            self._threshold = 0.5
            self.oof_accuracy_ = None
            self.cv_score_ = None

        self._fit_bootstrap_ensemble(X_np, y_int)
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        y_int = np.asarray(y_val, dtype=int)
        probs = self.predict_proba(X_val)[:, 1]
        self._threshold = self._best_accuracy_threshold(probs, y_int)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._clfs or not self._scalers:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        X_np = self._as_numpy(X)
        probs_pos = np.zeros(X_np.shape[0], dtype=np.float64)
        for scaler, clf in zip(self._scalers, self._clfs):
            probs_pos += clf.predict_proba(scaler.transform(X_np))[:, 1]
        probs_pos /= len(self._clfs)

        probs = np.stack([1.0 - probs_pos, probs_pos], axis=1)
        if probs.shape != (X_np.shape[0], 2):
            raise RuntimeError(
                f"Expected predict_proba shape {(X_np.shape[0], 2)}, got {probs.shape}."
            )
        return probs
