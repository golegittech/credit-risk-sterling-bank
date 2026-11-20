"""
SHAP explanations.

- The explainer always explains the SAME fitted model that produced the score
  (the previous version could silently fall back to a different model).
- Tree models (XGBoost, LightGBM, Random Forest): shap.TreeExplainer with
  feature_perturbation="interventional" and a background of
  config.SHAP_BACKGROUND_ROWS rows drawn from the (un-resampled) training set.
- Logistic Regression: shap.LinearExplainer on the same background.
- Units: XGBoost, LightGBM and Logistic Regression are explained in log-odds of
  default; Random Forest in probability of default.
- For a single applicant, contributions of one-hot columns are summed back to
  the original variable (e.g. all housing_status_* columns -> housing_status),
  which is what the adverse-action reasons are built from.
"""

import os
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src import config
from src.preprocessing import feature_frame, original_feature, transformed_features

PRETTY = {
    "applicant_age": "Applicant age (years)",
    "employment_length": "Employment length (years)",
    "annual_income": "Annual income (NGN)",
    "debt_to_income_ratio": "Debt-to-income ratio (%)",
    "credit_bureau_score": "Credit bureau score",
    "loan_amount": "Loan amount (NGN)",
    "housing_status": "Housing status",
    "existing_external_loan_status": "External loan status",
    "business_age_years": "Business age (years)",
    "interest_rate": "Interest rate (%)",
    "nature_of_business": "Sector",
    "business_region": "Region",
    "facility_type": "Facility type",
}


def output_units(model_name: str) -> str:
    return "probability" if model_name == "Random_Forest" else "log-odds"


class ModelExplainer:
    def __init__(self, pipe, model_name: str, portfolio: str, background_df: pd.DataFrame):
        self.pipe, self.model_name, self.portfolio = pipe, model_name, portfolio
        bg = background_df.sample(n=min(config.SHAP_BACKGROUND_ROWS, len(background_df)),
                                  random_state=config.RANDOM_SEED)
        self.bg_X, self.feature_names = transformed_features(pipe, feature_frame(bg, portfolio))
        model = pipe.named_steps["model"]
        if model_name == "Logistic_Regression":
            self.explainer = shap.LinearExplainer(model, self.bg_X)
        else:
            self.explainer = shap.TreeExplainer(model, data=self.bg_X,
                                                feature_perturbation="interventional",
                                                model_output="raw")

    def explain(self, df: pd.DataFrame) -> shap.Explanation:
        X, names = transformed_features(self.pipe, feature_frame(df, self.portfolio))
        ex = self.explainer(X, check_additivity=False) if self.model_name != "Logistic_Regression" else self.explainer(X)
        values, base = ex.values, ex.base_values
        if values.ndim == 3:  # (rows, features, classes) -> default class
            values, base = values[:, :, 1], np.asarray(base)[:, 1] if np.ndim(base) == 2 else base
        return shap.Explanation(values=values, base_values=base, data=X, feature_names=names)

    def grouped(self, ex_row: shap.Explanation) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, v in zip(ex_row.feature_names, ex_row.values):
            key = original_feature(name, self.portfolio)
            out[key] = out.get(key, 0.0) + float(v)
        return out


# ------------------------------ Global plots ------------------------------ #

def global_plots(expl: ModelExplainer, explain_df: pd.DataFrame, out_dir: str, prefix: str,
                 title_label: str) -> Dict[str, Any]:
    """Real SHAP beeswarm (shap.summary_plot) and mean |SHAP| bar chart."""
    os.makedirs(out_dir, exist_ok=True)
    ex = expl.explain(explain_df)
    units = output_units(expl.model_name)
    model_label = expl.model_name.replace("_", " ")

    shap.summary_plot(ex.values, ex.data, feature_names=ex.feature_names, max_display=12, show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 6.5)
    plt.xlabel(f"SHAP value ({units} of default; positive = higher risk)")
    plt.title(f"SHAP beeswarm: {model_label}, {title_label} portfolio\n"
              f"TreeExplainer (interventional), {len(explain_df)} test rows, "
              f"{config.SHAP_BACKGROUND_ROWS}-row training background", fontsize=10)
    plt.tight_layout()
    beeswarm = os.path.join(out_dir, f"{prefix}_shap_beeswarm.png")
    plt.savefig(beeswarm, dpi=200)
    plt.close("all")

    mean_abs = np.abs(ex.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    ranking = pd.DataFrame({
        "rank": np.arange(1, len(order) + 1),
        "feature": [ex.feature_names[i] for i in order],
        "mean_abs_shap": mean_abs[order],
    })
    top = ranking.head(12).iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"], top["mean_abs_shap"], color="#8B0000")
    plt.xlabel(f"Mean |SHAP value| ({units})")
    plt.title(f"Global feature importance (mean |SHAP|): {model_label}, {title_label} portfolio", fontsize=11)
    plt.tight_layout()
    bar = os.path.join(out_dir, f"{prefix}_shap_summary.png")
    plt.savefig(bar, dpi=200)
    plt.close("all")
    # Per-variable importance: sum the one-hot columns of each variable for every row,
    # then take mean |SHAP|. Per-column values can mislead for one-hot variables
    # (columns of one variable can carry large, offsetting contributions).
    grouped_vals = {}
    for j, name in enumerate(ex.feature_names):
        key = original_feature(name, expl.portfolio)
        grouped_vals[key] = grouped_vals.get(key, 0) + ex.values[:, j]
    g = sorted(((k, float(np.abs(v).mean())) for k, v in grouped_vals.items()), key=lambda kv: -kv[1])
    grouped = pd.DataFrame({"rank": range(1, len(g) + 1), "variable": [PRETTY.get(k, k) for k, _ in g],
                            "mean_abs_shap": [round(v, 4) for _, v in g]})
    return {"beeswarm": beeswarm, "bar": bar, "ranking": ranking, "grouped": grouped, "explanation": ex}


def waterfall_plot(expl: ModelExplainer, row_df: pd.DataFrame, out_path: str, caption: str):
    ex = expl.explain(row_df)[0]
    # Show raw values (not scaled) on the plot labels where possible.
    shap.plots.waterfall(ex, max_display=12, show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 6.5)
    plt.title(caption, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close("all")
    return ex


# --------------------------- Applicant explanation --------------------------- #

def _reason(feature: str, app: Dict[str, Any]) -> str:
    v = app.get(feature)
    if feature == "debt_to_income_ratio":
        return f"Debt-to-income ratio of {v}% is high for the requested facility."
    if feature == "credit_bureau_score":
        return f"Credit bureau score of {v} indicates elevated historical credit risk."
    if feature == "existing_external_loan_status":
        if v == "Defaulting":
            return "An existing loan with another lender is reported as defaulting."
        return f"Existing external loan status ({v}) adds to estimated risk."
    if feature == "employment_length":
        return f"Length of current employment ({v} years) is short."
    if feature == "housing_status":
        return f"Housing status ({v}) is associated with higher default rates in the portfolio."
    if feature == "annual_income":
        return f"Annual income (NGN {float(v):,.0f}) is low relative to the obligation."
    if feature == "loan_amount":
        return f"Requested amount (NGN {float(v):,.0f}) is large relative to the applicant's profile."
    if feature == "applicant_age":
        return f"Applicant age ({v}) contributed to higher estimated risk."
    if feature == "business_age_years":
        return f"Limited operating history ({v} years since incorporation)."
    if feature == "interest_rate":
        return f"Facility interest rate of {v}% is associated with higher default rates (association, not cause)."
    if feature == "nature_of_business":
        return f"Sector ({v}) has a higher observed default rate in the portfolio."
    if feature == "business_region":
        return f"Region ({v}) contributed to higher estimated risk."
    if feature == "facility_type":
        return f"Facility type ({v}) has a higher observed default rate in the portfolio."
    return f"{PRETTY.get(feature, feature)} increased estimated risk."


def explain_application(expl: ModelExplainer, app: Dict[str, Any], top_n: int = 4) -> Dict[str, Any]:
    df = pd.DataFrame([app])
    ex = expl.explain(df)[0]
    grouped = expl.grouped(ex)
    items = sorted(grouped.items(), key=lambda kv: abs(kv[1]), reverse=True)
    attributions = [{
        "feature": f,
        "label": PRETTY.get(f, f),
        "applicant_value": app.get(f),
        "shap_value": v,
        "direction": "INCREASES_RISK" if v > 0 else "DECREASES_RISK",
    } for f, v in items]
    drivers = [a for a in attributions if a["direction"] == "INCREASES_RISK"]
    return {
        "explained_model": expl.model_name,
        "units": output_units(expl.model_name),
        "base_value": float(np.ravel(ex.base_values)[0]),
        "top_attributions": attributions,
        "risk_drivers": drivers[:top_n],
        "mitigating_factors": [a for a in attributions if a["direction"] == "DECREASES_RISK"][:top_n],
        "adverse_action_reasons": [_reason(a["feature"], app) for a in drivers[:top_n]],
    }
