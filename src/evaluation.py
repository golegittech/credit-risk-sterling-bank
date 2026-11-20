"""
Evaluation helpers: confusion-matrix metrics, probability calibration,
bootstrap confidence intervals and the DeLong test for paired ROC-AUCs.

Every classification metric is computed from the confusion matrix, so
accuracy, precision, recall, specificity and F1 always reconcile.
"""

from typing import Dict

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src import config


def metrics_from_confusion(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return {
        "accuracy": (tp + tn) / n,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
    }


class PlattCalibrator:
    """Platt scaling fitted on validation data: p_cal = sigmoid(a * logit(p_raw) + b).
    It is monotone, so it never changes the ranking (ROC-AUC, PR-AUC) or the
    direction of SHAP contributions; it only corrects the probability level that
    the cost-based cutoff depends on (SMOTE inflates predicted default rates)."""

    def fit(self, p_raw: np.ndarray, y: np.ndarray):
        self.lr = LogisticRegression(C=1e6, max_iter=1000)
        self.lr.fit(self._logit(p_raw), y)
        return self

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(self._logit(p_raw))[:, 1]


def bootstrap_ci(y, p, metric, n_boot=config.N_BOOTSTRAP, seed=config.RANDOM_SEED, alpha=0.05):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        vals.append(metric(y[idx], p[idx]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def discrimination(y, p) -> Dict[str, float]:
    roc_lo, roc_hi = bootstrap_ci(y, p, roc_auc_score)
    pr_lo, pr_hi = bootstrap_ci(y, p, average_precision_score)
    return {
        "roc_auc": float(roc_auc_score(y, p)), "roc_auc_lo": roc_lo, "roc_auc_hi": roc_hi,
        "pr_auc": float(average_precision_score(y, p)), "pr_auc_lo": pr_lo, "pr_auc_hi": pr_hi,
    }


def brier(y, p) -> float:
    return float(brier_score_loss(y, p))


# ---------------- DeLong test (Sun & Xu, 2014 fast algorithm) ---------------- #

def _midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def _fast_delong(preds_sorted, m):
    n = preds_sorted.shape[1] - m
    pos, neg = preds_sorted[:, :m], preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(preds_sorted[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    return aucs, cov


def delong_test(y, p1, p2) -> Dict[str, float]:
    """Two-sided DeLong test for the difference between two correlated ROC-AUCs."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="mergesort")
    m = int(y.sum())
    preds = np.vstack([np.asarray(p1), np.asarray(p2)])[:, order]
    aucs, cov = _fast_delong(preds, m)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = diff / np.sqrt(var) if var > 0 else 0.0
    p = 2 * stats.norm.sf(abs(z))
    return {"auc_1": float(aucs[0]), "auc_2": float(aucs[1]), "diff": float(diff), "z": float(z), "p_value": float(p)}
