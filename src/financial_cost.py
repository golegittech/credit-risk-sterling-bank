"""
Financial cost matrix and cutoff selection.

Per-loan costs (assumptions in src/config.py, reported in the cost table):
  C_FN(i) = EAD_i x LGD                      EAD_i = loan amount of applicant i
  C_FP(i) = loan amount_i x net margin x tenor

Expected loss at cutoff t = sum of C_FN over defaulters approved (p < t)
                          + sum of C_FP over good borrowers refused (p >= t)

The cutoff is chosen on the VALIDATION set and then applied unchanged to the
test set, so the test-set saving is an honest out-of-sample estimate.
"""

from typing import Dict, Any

import numpy as np

from src import config

THRESHOLD_GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)


def cost_vectors(loan_amount: np.ndarray, portfolio: str, lgd: float = None):
    a = dict(config.COST_ASSUMPTIONS[portfolio])
    if lgd is not None:
        a["lgd"] = lgd
    loan_amount = np.asarray(loan_amount, dtype=float)
    c_fn = loan_amount * a["lgd"]
    c_fp = loan_amount * a["net_margin"] * a["tenor_years"]
    return c_fn, c_fp


def bayes_cutoff(portfolio: str, lgd: float = None) -> float:
    """Theoretical cutoff for calibrated probabilities: C_FP / (C_FP + C_FN).
    With costs proportional to loan size this is the same for every applicant."""
    a = dict(config.COST_ASSUMPTIONS[portfolio])
    if lgd is not None:
        a["lgd"] = lgd
    fp = a["net_margin"] * a["tenor_years"]
    return fp / (fp + a["lgd"])


def confusion_at(y: np.ndarray, p: np.ndarray, t: float) -> Dict[str, int]:
    pred = p >= t
    y = np.asarray(y).astype(bool)
    return {
        "TP": int(np.sum(pred & y)),
        "FP": int(np.sum(pred & ~y)),
        "FN": int(np.sum(~pred & y)),
        "TN": int(np.sum(~pred & ~y)),
    }


def expected_loss(y, p, t, c_fn, c_fp) -> Dict[str, float]:
    y = np.asarray(y).astype(bool)
    pred = np.asarray(p) >= t
    fn_loss = float(np.sum(c_fn[~pred & y]))
    fp_loss = float(np.sum(c_fp[pred & ~y]))
    return {"fn_loss": fn_loss, "fp_loss": fp_loss, "total": fn_loss + fp_loss}


def loss_curve(y, p, c_fn, c_fp, grid=THRESHOLD_GRID):
    return np.array([expected_loss(y, p, t, c_fn, c_fp)["total"] for t in grid])


def choose_cutoff(y, p, c_fn, c_fp, grid=THRESHOLD_GRID) -> float:
    losses = loss_curve(y, p, c_fn, c_fp, grid)
    return float(grid[int(np.argmin(losses))])


def summarise_at_cutoffs(y, p, c_fn, c_fp, cutoff: float) -> Dict[str, Any]:
    at50 = expected_loss(y, p, 0.50, c_fn, c_fp)
    atc = expected_loss(y, p, cutoff, c_fn, c_fp)
    return {
        "cutoff": cutoff,
        **{k: v for k, v in confusion_at(y, p, cutoff).items()},
        "loss_at_050": at50["total"],
        "loss_at_cutoff": atc["total"],
        "saving": at50["total"] - atc["total"],
    }
