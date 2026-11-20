"""
Single source of truth for every setting used in the study.

Every number reported in Chapter 4 is produced by `python -m src.experiment`
using the values below. Change a value here, rerun, and every table and figure
is regenerated from that one run.

DATA NOTE: the two portfolios are SIMULATED by `src/data_generator.py`.
They are modelled on the products and field types of a Nigerian commercial bank
(Sterling Bank Nigeria) but contain no real customer records.
"""

RANDOM_SEED = 42

# 60 / 20 / 20 stratified split: train (fit models) / validation (calibrate and
# choose the cutoff) / test (touched once, for the final numbers).
TRAIN_FRAC = 0.60
VALID_FRAC = 0.20
TEST_FRAC = 0.20

CV_FOLDS = 5
SMOTE_K_NEIGHBORS = 5
N_BOOTSTRAP = 1000
SHAP_BACKGROUND_ROWS = 200

PORTFOLIOS = {
    "personal": {
        "label": "Personal (salary earner)",
        "csv": "data/personal_loans.csv",
        "num_cols": [
            "applicant_age",
            "employment_length",
            "annual_income",
            "debt_to_income_ratio",
            "credit_bureau_score",
            "loan_amount",
        ],
        "cat_cols": ["housing_status", "existing_external_loan_status"],
        "fairness_group": "age_band",
    },
    "retail": {
        "label": "SME",
        "csv": "data/retail_sme_loans.csv",
        "num_cols": ["business_age_years", "loan_amount", "interest_rate"],
        "cat_cols": [
            "nature_of_business",
            "business_region",
            "facility_type",
            "existing_external_loan_status",
        ],
        "fairness_group": "business_region",
    },
}
TARGET = "loan_status"

# Hyperparameters are fixed (no tuning search was run). They are reported in
# Table 4.0 and evaluated with 5-fold stratified cross-validation.
MODEL_PARAMS = {
    "Logistic_Regression": dict(solver="lbfgs", max_iter=1000),
    "Random_Forest": dict(n_estimators=200, max_depth=10, min_samples_split=5),
    "LightGBM": dict(n_estimators=250, learning_rate=0.05, num_leaves=31),
    "XGBoost": dict(
        n_estimators=250, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
    ),
}
MODEL_ORDER = list(MODEL_PARAMS)

# Imbalance strategies compared. "smotenc" is the main strategy and is applied
# to every model so the benchmark is like-for-like.
STRATEGIES = ["none", "class_weight", "smotenc"]
MAIN_STRATEGY = "smotenc"

# ---------------------------------------------------------------------------
# Cost assumptions (Table 3.x). These are ASSUMPTIONS, not Sterling Bank
# figures. Replace them with sourced values before relying on the Naira totals.
#   Cost of a false negative (defaulter approved)   = EAD x LGD
#       EAD = the loan amount requested (assumes full drawdown before default)
#   Cost of a false positive (good borrower refused) = loan amount x net margin x tenor
# ---------------------------------------------------------------------------
COST_ASSUMPTIONS = {
    "personal": {"lgd": 0.60, "net_margin": 0.10, "tenor_years": 1.0},
    "retail": {"lgd": 0.60, "net_margin": 0.10, "tenor_years": 1.0},
}
LGD_SENSITIVITY = [0.45, 0.60, 0.75]

# Three-tier decision: below REVIEW_BAND x cutoff -> approve; between -> manual review.
REVIEW_BAND = 0.70
