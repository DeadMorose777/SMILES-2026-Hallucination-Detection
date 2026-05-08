"""
probe.py - Hallucination probe classifier.

The probe stays deliberately small: scaled linear logistic models with an
inner C search, bootstrap probability averaging, and threshold tuning in fit().
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


CANDIDATE_C = (0.003, 0.01, 0.03, 0.1)
N_BOOTSTRAP = 5
BOOTSTRAP_SEEDS = (11, 23, 37, 53, 79)
INNER_FOLDS = 5
RANDOM_STATE = 42


class HallucinationProbe(nn.Module):
    """Bootstrap ensemble of CPU logistic probes."""

    def __init__(self) -> None:
        super().__init__()
        self._scalers: list[StandardScaler] = []
        self._clfs: list[LogisticRegression] = []
        self._threshold: float = 0.5
        self.best_C_: float = CANDIDATE_C[0]
        self.best_params_ = {
            "C": self.best_C_,
            "class_weight": None,
            "n_bootstrap": N_BOOTSTRAP,
        }
        self.cv_score_: float | None = None
        self.oof_f1_: float | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("HallucinationProbe uses sklearn; call predict_proba().")

    @staticmethod
    def _as_numpy(X: np.ndarray) -> np.ndarray:
        X_np = np.asarray(X, dtype=np.float32)
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)
        return X_np

    @staticmethod
    def _new_clf(C: float, seed: int = RANDOM_STATE) -> LogisticRegression:
        return LogisticRegression(
            C=C,
            solver="lbfgs",
            max_iter=5000,
            class_weight=None,
            random_state=seed,
        )

    @staticmethod
    def _fit_one(
        X: np.ndarray,
        y: np.ndarray,
        C: float,
        seed: int = RANDOM_STATE,
    ) -> tuple[StandardScaler, LogisticRegression]:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf = HallucinationProbe._new_clf(C=C, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_scaled, y)
        return scaler, clf

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
                    and (
                        f1 > best_f1
                        or (np.isclose(f1, best_f1) and distance < best_distance)
                    )
                )
            ):
                best_accuracy = accuracy
                best_f1 = f1
                best_distance = distance
                best_threshold = float(threshold)

        return best_threshold

    def _oof_probabilities(
        self,
        X: np.ndarray,
        y: np.ndarray,
        C: float,
        cv: StratifiedKFold,
    ) -> np.ndarray:
        oof_probs = np.zeros(len(y), dtype=np.float64)
        for train_idx, val_idx in cv.split(X, y):
            scaler, clf = self._fit_one(X[train_idx], y[train_idx], C=C)
            oof_probs[val_idx] = clf.predict_proba(scaler.transform(X[val_idx]))[:, 1]
        return oof_probs

    def _select_C(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: StratifiedKFold,
    ) -> tuple[float, float, float, float]:
        best_C = CANDIDATE_C[0]
        best_threshold = 0.5
        best_accuracy = -1.0
        best_f1 = -1.0

        for C in CANDIDATE_C:
            oof_probs = self._oof_probabilities(X, y, C=C, cv=cv)
            threshold = self._best_accuracy_threshold(oof_probs, y)
            y_pred = (oof_probs >= threshold).astype(int)
            accuracy = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred, zero_division=0)

            if (
                accuracy > best_accuracy
                or (
                    np.isclose(accuracy, best_accuracy)
                    and (f1 > best_f1 or (np.isclose(f1, best_f1) and C < best_C))
                )
            ):
                best_C = C
                best_threshold = threshold
                best_accuracy = accuracy
                best_f1 = f1

        return best_C, best_threshold, best_accuracy, best_f1

    def _fit_bootstrap_ensemble(self, X: np.ndarray, y: np.ndarray, C: float) -> None:
        self._scalers = []
        self._clfs = []
        n_samples = len(y)

        for seed in BOOTSTRAP_SEEDS:
            rng = np.random.RandomState(seed)
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            if np.unique(y[idx]).size < 2:
                idx = np.arange(n_samples)
            scaler, clf = self._fit_one(X[idx], y[idx], C=C, seed=seed)
            self._scalers.append(scaler)
            self._clfs.append(clf)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X_np = self._as_numpy(X)
        y_int = np.asarray(y, dtype=int)

        class_counts = np.bincount(y_int, minlength=2)
        min_class_count = int(class_counts.min())
        n_splits = min(INNER_FOLDS, min_class_count)

        if n_splits >= 2:
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=RANDOM_STATE,
            )
            (
                self.best_C_,
                self._threshold,
                accuracy,
                f1,
            ) = self._select_C(X_np, y_int, cv=cv)
            self.cv_score_ = float(accuracy)
            self.oof_f1_ = float(f1)
        else:
            self.best_C_ = CANDIDATE_C[0]
            self._threshold = 0.5
            self.cv_score_ = None
            self.oof_f1_ = None

        self.best_params_ = {
            "C": self.best_C_,
            "class_weight": None,
            "n_bootstrap": N_BOOTSTRAP,
            "bootstrap_seeds": BOOTSTRAP_SEEDS,
        }
        self._fit_bootstrap_ensemble(X_np, y_int, C=self.best_C_)
        return self

    def fit_hyperparameters(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "HallucinationProbe":
        y_int = np.asarray(y_val, dtype=int)
        probs = self.predict_proba(X_val)[:, 1]
        self._threshold = self._best_accuracy_threshold(probs, y_int)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._scalers or not self._clfs:
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
