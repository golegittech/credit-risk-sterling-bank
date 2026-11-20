"""
Functional test of the deployed API (supervisor review, item A7).

Sends 10 held-out test-set applications per portfolio to POST /explain/{portfolio}
through FastAPI's TestClient and checks, for each one:
  1. HTTP 200 and the expected model is used;
  2. the returned probability equals the probability computed directly from the
     saved model bundle (expected output);
  3. the decision follows the three-tier rule (approve / manual review / reject);
  4. a rejected application receives at least one plain-language reason;
  5. SHAP contributions plus the base value add up to the model's raw score
     (log-odds or probability), i.e. the explanation belongs to the scoring model.

    python -m src.functional_test     -> results/thesis/tables/functional_test.csv
"""

import os

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api.main import app
from src import config, scoring
from src.preprocessing import feature_frame, load_portfolio, split_portfolio

N_PER_PORTFOLIO = 10


def _clean(row: pd.Series, portfolio: str) -> dict:
    spec = config.PORTFOLIOS[portfolio]
    d = {}
    for c in spec["num_cols"] + spec["cat_cols"]:
        v = row[c]
        if pd.isna(v):  # API requires complete applications; use training median
            v = float(load_portfolio(portfolio)[c].median())
        d[c] = v.item() if hasattr(v, "item") else v
    if portfolio == "personal":
        d["applicant_age"] = int(d["applicant_age"])
        d["credit_bureau_score"] = int(d["credit_bureau_score"])
    return d


def run() -> pd.DataFrame:
    client = TestClient(app)
    rows = []
    for portfolio in config.PORTFOLIOS:
        _, _, test = split_portfolio(load_portfolio(portfolio))
        b = scoring.load_bundle(portfolio)
        for i in range(N_PER_PORTFOLIO):
            app_d = _clean(test.iloc[i], portfolio)
            r = client.post(f"/explain/{portfolio}", json=app_d)
            body = r.json()
            df = pd.DataFrame([app_d])
            raw = float(b["pipeline"].predict_proba(feature_frame(df, portfolio))[0, 1])
            exp_p = float(b["calibrator"].transform(np.array([raw]))[0])
            exp_dec, _ = scoring.decide(exp_p, b["cutoff"], b["review_band"])
            xai = body.get("xai_explanation", {})
            shap_sum = xai.get("base_value", 0) + sum(a["shap_value"] for a in xai.get("top_attributions", []))
            target = np.log(raw / (1 - raw)) if xai.get("units") == "log-odds" else raw
            checks = {
                "HTTP 200": r.status_code == 200,
                "Model matches": body.get("model") == b["model_name"] == xai.get("explained_model"),
                "Probability matches": abs(body.get("default_probability", -1) - round(exp_p, 4)) < 1e-4,
                "Decision matches": body.get("decision") == exp_dec,
                "Reason given if rejected": body.get("decision") != "REJECT" or len(xai.get("adverse_action_reasons", [])) > 0,
                "SHAP additivity": abs(shap_sum - target) < 1e-3,
            }
            rows.append({
                "Portfolio": config.PORTFOLIOS[portfolio]["label"], "Case": i + 1,
                "Actual outcome": "Default" if test.iloc[i][config.TARGET] == 1 else "Repaid",
                "Expected PD": round(exp_p, 4), "Returned PD": body.get("default_probability"),
                "Expected decision": exp_dec, "Returned decision": body.get("decision"),
                "Top risk-raising factor": (xai.get("adverse_action_reasons") or [""])[0],
                **{k: "Pass" if v else "FAIL" for k, v in checks.items()},
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = run()
    os.makedirs("results/thesis/tables", exist_ok=True)
    out.to_csv("results/thesis/tables/functional_test.csv", index=False)
    fails = (out == "FAIL").sum().sum()
    print(out.to_string(index=False))
    print(f"\n{len(out)} applications, {fails} failed checks")
