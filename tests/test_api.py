"""API tests. Require trained bundles (python -m src.experiment)."""

import os

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src import scoring

pytestmark = pytest.mark.skipif(
    not all(os.path.exists(scoring.bundle_path(p)) for p in ("personal", "retail")),
    reason="models not trained",
)
client = TestClient(app)

HIGH_RISK_PERSONAL = {"applicant_age": 26, "housing_status": "RENT", "employment_length": 0.5,
                      "annual_income": 1_500_000, "debt_to_income_ratio": 70, "credit_bureau_score": 380,
                      "existing_external_loan_status": "Defaulting", "loan_amount": 700_000}
LOW_RISK_PERSONAL = {"applicant_age": 50, "housing_status": "OWN", "employment_length": 20,
                     "annual_income": 15_000_000, "debt_to_income_ratio": 10, "credit_bureau_score": 820,
                     "existing_external_loan_status": "None", "loan_amount": 2_000_000}


def test_health_and_index():
    h = client.get("/health").json()
    assert h["status"] == "healthy"
    assert client.get("/").status_code == 200


def test_obvious_cases():
    hi = client.post("/explain/personal", json=HIGH_RISK_PERSONAL).json()
    lo = client.post("/predict/personal", json=LOW_RISK_PERSONAL).json()
    assert hi["decision"] == "REJECT" and hi["xai_explanation"]["adverse_action_reasons"]
    assert lo["decision"] == "APPROVE"
    feats = {a["feature"]: a for a in hi["xai_explanation"]["top_attributions"]}
    # a defaulting external loan must never be reported as lowering risk
    assert feats["existing_external_loan_status"]["direction"] == "INCREASES_RISK"


def test_invalid_category_rejected():
    bad = dict(HIGH_RISK_PERSONAL, housing_status="CASTLE")
    assert client.post("/predict/personal", json=bad).status_code == 422


def test_functional_suite_all_pass():
    from src.functional_test import run
    df = run()
    assert len(df) == 20
    assert not (df == "FAIL").any().any(), df[(df == "FAIL").any(axis=1)].to_string()
