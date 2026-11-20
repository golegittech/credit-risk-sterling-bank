"""
Unit tests for the revised pipeline.

    python -m pytest -q
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_generator import generate_personal_loans, generate_retail_sme_loans
from src.evaluation import PlattCalibrator, delong_test, metrics_from_confusion
from src.financial_cost import bayes_cutoff, choose_cutoff, confusion_at, cost_vectors, expected_loss
from src.preprocessing import build_pipeline, feature_frame, load_portfolio, split_portfolio


def test_generators_schema():
    p = generate_personal_loans(n_samples=200, seed=1)
    r = generate_retail_sme_loans(n_samples=200, seed=1)
    assert set(config.PORTFOLIOS["personal"]["num_cols"] + config.PORTFOLIOS["personal"]["cat_cols"]) <= set(p.columns)
    assert set(config.PORTFOLIOS["retail"]["num_cols"] + config.PORTFOLIOS["retail"]["cat_cols"]) <= set(r.columns)
    assert p["loan_status"].isin([0, 1]).all() and r["loan_status"].isin([0, 1]).all()


def test_none_category_is_not_read_as_missing():
    df = load_portfolio("personal")
    assert df["existing_external_loan_status"].isna().sum() == 0
    assert "None" in set(df["existing_external_loan_status"])


def test_split_is_60_20_20_and_stratified():
    df = load_portfolio("personal")
    tr, va, te = split_portfolio(df)
    assert (len(tr), len(va), len(te)) == (3600, 1200, 1200)
    rates = [d["loan_status"].mean() for d in (tr, va, te)]
    assert max(rates) - min(rates) < 0.005


def test_smotenc_balances_training_only_and_keeps_valid_categories():
    df = generate_personal_loans(n_samples=600, seed=3)
    pipe = build_pipeline("personal", "Logistic_Regression", "smotenc")
    Xi = pipe.named_steps["impute"].fit_transform(feature_frame(df, "personal"))
    Xr, yr = pipe.named_steps["smote"].fit_resample(Xi, df["loan_status"])
    assert np.mean(yr) == 0.5
    assert set(Xr["housing_status"]) <= {"OWN", "RENT", "MORTGAGE"}  # no interpolated categories
    pipe.fit(feature_frame(df, "personal"), df["loan_status"])
    assert pipe.predict_proba(feature_frame(df, "personal")).shape == (600, 2)  # no resampling at predict


def test_metrics_reconcile_with_confusion_matrix():
    m = metrics_from_confusion(tp=111, fp=84, fn=31, tn=974)
    assert m["precision"] == pytest.approx(111 / 195)
    assert m["specificity"] == pytest.approx(974 / 1058)
    assert m["accuracy"] == pytest.approx(1085 / 1200)


def test_cost_and_cutoff():
    y = np.array([0, 0, 0, 1, 0, 1, 0, 0, 1, 0])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.2, 0.6, 0.1, 0.4, 0.9, 0.1])
    amt = np.full(10, 1_000_000.0)
    cfn, cfp = cost_vectors(amt, "personal")
    assert cfn[0] == pytest.approx(600_000) and cfp[0] == pytest.approx(100_000)
    t = choose_cutoff(y, p, cfn, cfp)
    assert expected_loss(y, p, t, cfn, cfp)["total"] <= expected_loss(y, p, 0.5, cfn, cfp)["total"]
    assert confusion_at(y, p, 0.5) == {"TP": 3, "FP": 0, "FN": 0, "TN": 7}
    assert bayes_cutoff("personal") == pytest.approx(0.1 / 0.7)


def test_platt_calibration_is_monotone():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, 500)
    y = (rng.uniform(size=500) < p * 0.3).astype(int)
    cal = PlattCalibrator().fit(p, y)
    out = cal.transform(np.sort(p))
    assert np.all(np.diff(out) >= -1e-12)


def test_delong_matches_sklearn_auc():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    a, b = y + rng.normal(0, 1, 400), y + rng.normal(0, 2, 400)
    d = delong_test(y, a, b)
    assert d["auc_1"] == pytest.approx(roc_auc_score(y, a))
    assert 0 <= d["p_value"] <= 1
