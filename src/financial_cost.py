"""
Financial Cost Matrix and Threshold Calibration Module.
Calculates expected portfolio losses incorporating asymmetric classification costs:
- C_FN (Cost of False Negative): Direct capital write-off (EAD * LGD)
- C_FP (Cost of False Positive): Opportunity cost of forgone interest margin
Evaluates optimal decision thresholds that minimize monetary losses for Sterling Bank Nigeria.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any

# Standard default financial parameters for Nigerian retail banking
COST_PARAMS = {
    "personal": {
        "ead": 2_500_000.0,      # Exposure at Default in NGN
        "lgd": 0.65,             # Loss Given Default (65%)
        "c_fn": 2_500_000.0 * 0.65, # NGN 1,625,000
        "c_fp": 300_000.0,       # Forgone net interest margin (NGN 300,000)
    },
    "retail": {
        "ead": 12_000_000.0,     # Exposure at Default in NGN
        "lgd": 0.60,             # Loss Given Default (60%)
        "c_fn": 12_000_000.0 * 0.60, # NGN 7,200,000
        "c_fp": 1_200_000.0,     # Forgone net interest margin (NGN 1,200,000)
    }
}

def evaluate_financial_loss(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    c_fn: float,
    c_fp: float
) -> Dict[str, float]:
    """
    Computes confusion matrix counts and total expected financial loss at a given cutoff threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1)) # Good applicant rejected (lost interest)
    fn = np.sum((y_true == 1) & (y_pred == 0)) # Defaulter approved (capital write-off)
    
    total_loss = (fn * c_fn) + (fp * c_fp)
    
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "fn_loss": float(fn * c_fn),
        "fp_loss": float(fp * c_fp),
        "total_financial_loss": float(total_loss)
    }

def optimize_decision_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    portfolio_type: str = "personal",
    threshold_range: np.ndarray = np.linspace(0.05, 0.95, 91)
) -> Dict[str, Any]:
    """
    Sweeps through candidate thresholds and identifies the financial loss-minimizing cutoff.
    """
    params = COST_PARAMS.get(portfolio_type.lower(), COST_PARAMS["personal"])
    c_fn = params["c_fn"]
    c_fp = params["c_fp"]
    
    loss_curve = []
    for th in threshold_range:
        res = evaluate_financial_loss(y_true, y_prob, th, c_fn, c_fp)
        loss_curve.append(res)
        
    df_curve = pd.DataFrame(loss_curve)
    best_idx = df_curve["total_financial_loss"].idxmin()
    best_row = df_curve.iloc[best_idx].to_dict()
    
    # Baseline comparison at conventional 0.50 threshold
    res_default_50 = evaluate_financial_loss(y_true, y_prob, 0.50, c_fn, c_fp)
    savings = res_default_50["total_financial_loss"] - best_row["total_financial_loss"]
    savings_pct = (savings / res_default_50["total_financial_loss"]) * 100 if res_default_50["total_financial_loss"] > 0 else 0
    
    return {
        "portfolio_type": portfolio_type,
        "c_fn": c_fn,
        "c_fp": c_fp,
        "optimal_threshold": float(best_row["threshold"]),
        "min_financial_loss": float(best_row["total_financial_loss"]),
        "default_50_loss": float(res_default_50["total_financial_loss"]),
        "monetary_savings_ngn": float(savings),
        "savings_percentage": float(savings_pct),
        "optimal_metrics": best_row,
        "loss_curve": df_curve.to_dict(orient="records")
    }
