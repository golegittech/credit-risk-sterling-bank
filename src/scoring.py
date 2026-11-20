"""
Scoring service shared by the FastAPI backend and the Streamlit dashboard.

Loads models/{portfolio}_deployment.joblib (written by src.experiment), which
holds the fitted pipeline, its Platt calibrator, the cost-based cutoff chosen
on the validation set, and a training background sample for SHAP.
"""

import os
from functools import lru_cache
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from src.explainability import ModelExplainer, explain_application
from src.preprocessing import feature_frame

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def bundle_path(portfolio: str) -> str:
    return os.path.join(MODELS_DIR, f"{portfolio}_deployment.joblib")


@lru_cache(maxsize=None)
def load_bundle(portfolio: str) -> Dict[str, Any]:
    return joblib.load(bundle_path(portfolio))


@lru_cache(maxsize=None)
def load_explainer(portfolio: str) -> ModelExplainer:
    b = load_bundle(portfolio)
    return ModelExplainer(b["pipeline"], b["model_name"], portfolio, b["background"])


def decide(prob: float, cutoff: float, band: float):
    if prob >= cutoff:
        return "REJECT", "HIGH RISK"
    if prob >= cutoff * band:
        return "MANUAL REVIEW", "MODERATE RISK"
    return "APPROVE", "LOW RISK"


def score_frame(portfolio: str, df: pd.DataFrame) -> pd.DataFrame:
    b = load_bundle(portfolio)
    raw = b["pipeline"].predict_proba(feature_frame(df, portfolio))[:, 1]
    cal = b["calibrator"].transform(raw)
    out = df.copy()
    out["Predicted_Default_Prob"] = np.round(cal, 4)
    out["Underwriter_Decision"] = [decide(p, b["cutoff"], b["review_band"])[0] for p in cal]
    return out


def score_application(portfolio: str, app: Dict[str, Any]) -> Dict[str, Any]:
    b = load_bundle(portfolio)
    df = pd.DataFrame([app])
    raw = float(b["pipeline"].predict_proba(feature_frame(df, portfolio))[0, 1])
    prob = float(b["calibrator"].transform(np.array([raw]))[0])
    decision, tier = decide(prob, b["cutoff"], b["review_band"])
    return {
        "model": b["model_name"],
        "default_probability": round(prob, 4),
        "decision": decision,
        "risk_tier": tier,
        "optimal_threshold_applied": round(b["cutoff"], 3),
        "manual_review_from": round(b["cutoff"] * b["review_band"], 3),
        "repayment_likelihood": round((1.0 - prob) * 100, 2),
    }


def explain(portfolio: str, app: Dict[str, Any]) -> Dict[str, Any]:
    return explain_application(load_explainer(portfolio), app)
