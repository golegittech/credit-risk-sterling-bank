"""
Leakage-free modelling pipelines.

Each pipeline is: impute -> (optional SMOTENC) -> encode/scale -> model.
It is an imbalanced-learn Pipeline, so the resampling step runs only inside
`fit` (training data or a training fold) and is skipped by `predict_proba`.
Validation and test data therefore always keep their real class balance.

SMOTENC is used instead of plain SMOTE because the data mixes numeric and
categorical columns: plain SMOTE would interpolate one-hot columns into
invalid fractional values, while SMOTENC assigns each synthetic record the
most common category among its nearest neighbours.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src import config


def load_portfolio(portfolio: str) -> pd.DataFrame:
    # keep_default_na=False: the category "None" (no external loan) must stay a
    # category. pandas' default would read it as missing, and it would then be
    # imputed as "Performing" (a bug in the original pipeline).
    return pd.read_csv(config.PORTFOLIOS[portfolio]["csv"], keep_default_na=False, na_values=[""])


def split_portfolio(df: pd.DataFrame, seed: int = config.RANDOM_SEED) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 60/20/20 train/validation/test split."""
    train, rest = train_test_split(
        df, train_size=config.TRAIN_FRAC, stratify=df[config.TARGET], random_state=seed
    )
    valid_share = config.VALID_FRAC / (config.VALID_FRAC + config.TEST_FRAC)
    valid, test = train_test_split(
        rest, train_size=valid_share, stratify=rest[config.TARGET], random_state=seed
    )
    return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)


def _make_model(name: str, strategy: str, pos_weight: float, seed: int):
    params = dict(config.MODEL_PARAMS[name])
    weighted = strategy == "class_weight"
    if name == "Logistic_Regression":
        return LogisticRegression(random_state=seed, class_weight="balanced" if weighted else None, **params)
    if name == "Random_Forest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, class_weight="balanced" if weighted else None, **params)
    if name == "LightGBM":
        return LGBMClassifier(random_state=seed, verbose=-1, n_jobs=4, deterministic=True, force_row_wise=True,
                              scale_pos_weight=pos_weight if weighted else 1.0, **params)
    if name == "XGBoost":
        return XGBClassifier(random_state=seed, eval_metric="logloss", n_jobs=4,
                             scale_pos_weight=pos_weight if weighted else 1.0, **params)
    raise ValueError(name)


def build_pipeline(portfolio: str, model_name: str, strategy: str, pos_weight: float = 1.0,
                   seed: int = config.RANDOM_SEED) -> Pipeline:
    spec = config.PORTFOLIOS[portfolio]
    num, cat = spec["num_cols"], spec["cat_cols"]

    impute = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), num),
            ("cat", SimpleImputer(strategy="most_frequent"), cat),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    # Scaling only matters for Logistic Regression; trees keep raw units so
    # SHAP plots show real values (Naira, %, years).
    num_step = StandardScaler() if model_name == "Logistic_Regression" else "passthrough"
    encode = ColumnTransformer(
        [
            ("num", num_step, num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ],
        verbose_feature_names_out=False,
    )

    steps = [("impute", impute)]
    if strategy == "smotenc":
        steps.append(("smote", SMOTENC(categorical_features=cat, k_neighbors=config.SMOTE_K_NEIGHBORS,
                                       random_state=seed)))
    steps += [("encode", encode), ("model", _make_model(model_name, strategy, pos_weight, seed))]
    return Pipeline(steps)


def feature_frame(df: pd.DataFrame, portfolio: str) -> pd.DataFrame:
    spec = config.PORTFOLIOS[portfolio]
    return df[spec["num_cols"] + spec["cat_cols"]]


def transformed_features(pipe: Pipeline, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Model-ready matrix and column names (resampling is skipped at transform time)."""
    pre = pipe[:-1]
    Xt = pre.transform(X)
    names = list(pipe.named_steps["encode"].get_feature_names_out())
    return np.asarray(Xt, dtype=float), names


def original_feature(encoded_name: str, portfolio: str) -> str:
    """Map a one-hot column such as 'housing_status_RENT' back to 'housing_status'."""
    spec = config.PORTFOLIOS[portfolio]
    if encoded_name in spec["num_cols"]:
        return encoded_name
    for c in sorted(spec["cat_cols"], key=len, reverse=True):
        if encoded_name.startswith(c + "_"):
            return c
    return encoded_name
