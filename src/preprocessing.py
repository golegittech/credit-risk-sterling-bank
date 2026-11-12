"""
Preprocessing and Resampling Pipeline for Credit Risk Assessment.
Handles:
- Missing value imputation (Median for continuous, Mode for categorical)
- One-hot encoding for nominal categories
- Numerical standardization
- Stratified train-test split
- SMOTE class imbalance resampling strictly on training fold (preventing data leakage)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib
import os
from typing import Tuple, Dict, Any, List

class CreditDataPreprocessor:
    def __init__(self, portfolio_type: str = "personal"):
        """
        portfolio_type: 'personal' or 'retail'
        """
        self.portfolio_type = portfolio_type.lower()
        self.num_imputer = None
        self.cat_imputer = None
        self.encoder = None
        self.scaler = None
        self.feature_names = []
        self.num_cols = []
        self.cat_cols = []
        self.target_col = 'loan_status'
        
        self._init_column_definitions()

    def _init_column_definitions(self):
        if self.portfolio_type == "personal":
            self.num_cols = [
                'applicant_age',
                'employment_length',
                'annual_income',
                'debt_to_income_ratio',
                'credit_bureau_score',
                'loan_amount'
            ]
            self.cat_cols = [
                'housing_status',
                'existing_external_loan_status'
            ]
        elif self.portfolio_type == "retail":
            self.num_cols = [
                'business_age_years',
                'loan_amount',
                'interest_rate'
            ]
            self.cat_cols = [
                'nature_of_business',
                'business_region',
                'facility_type',
                'existing_external_loan_status'
            ]
        else:
            raise ValueError(f"Unknown portfolio_type: {self.portfolio_type}")

    def fit_transform_train(self, df: pd.DataFrame, apply_smote: bool = True, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Fits transformers on raw training dataframe and optionally applies SMOTE.
        Returns: X_train_processed, y_train_resampled, feature_names
        """
        X = df[self.num_cols + self.cat_cols].copy()
        y = df[self.target_col].values.astype(int)

        # 1. Missing Value Imputation
        self.num_imputer = SimpleImputer(strategy='median')
        X_num_imp = self.num_imputer.fit_transform(X[self.num_cols])

        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        X_cat_imp = self.cat_imputer.fit_transform(X[self.cat_cols])

        # 2. Categorical One-Hot Encoding
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_cat_encoded = self.encoder.fit_transform(X_cat_imp)
        cat_feature_names = self.encoder.get_feature_names_out(self.cat_cols).tolist()

        # Combine features
        self.feature_names = self.num_cols + cat_feature_names
        X_combined = np.hstack([X_num_imp, X_cat_encoded])

        # 3. Scaler (fitted for models that need scaling, e.g. Logistic Regression)
        self.scaler = StandardScaler()
        self.scaler.fit(X_combined)

        # 4. SMOTE Resampling (Train set ONLY to avoid data leakage)
        if apply_smote:
            smote = SMOTE(random_state=random_state)
            X_resampled, y_resampled = smote.fit_resample(X_combined, y)
        else:
            X_resampled, y_resampled = X_combined, y

        return X_resampled, y_resampled, self.feature_names

    def transform_test(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms test/validation data using fitted transformers.
        Does NOT apply SMOTE (test data preserves original real-world distribution).
        """
        X = df[self.num_cols + self.cat_cols].copy()
        y = df[self.target_col].values.astype(int)

        X_num_imp = self.num_imputer.transform(X[self.num_cols])
        X_cat_imp = self.cat_imputer.transform(X[self.cat_cols])
        X_cat_encoded = self.encoder.transform(X_cat_imp)

        X_combined = np.hstack([X_num_imp, X_cat_encoded])
        return X_combined, y

    def transform_single(self, input_dict: Dict[str, Any]) -> np.ndarray:
        """
        Transforms a single application dictionary into a 2D feature array.
        """
        df_single = pd.DataFrame([input_dict])
        X_num = df_single[self.num_cols].values
        X_cat = df_single[self.cat_cols].values

        X_num_imp = self.num_imputer.transform(X_num)
        X_cat_imp = self.cat_imputer.transform(X_cat)
        X_cat_encoded = self.encoder.transform(X_cat_imp)

        return np.hstack([X_num_imp, X_cat_encoded])

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "CreditDataPreprocessor":
        return joblib.load(filepath)


def prepare_portfolio_splits(csv_path: str, portfolio_type: str, test_size: float = 0.20, random_state: int = 42):
    """
    Loads dataset, splits into train and test with stratification, fits preprocessor and returns splits.
    """
    df = pd.read_csv(csv_path)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['loan_status']
    )

    preprocessor = CreditDataPreprocessor(portfolio_type=portfolio_type)
    X_train_res, y_train_res, feature_names = preprocessor.fit_transform_train(train_df, apply_smote=True)
    X_test, y_test = preprocessor.transform_test(test_df)
    
    # Also prepare unscaled raw train for comparison
    X_train_raw, y_train_raw, _ = preprocessor.fit_transform_train(train_df, apply_smote=False)

    return {
        'train_df': train_df,
        'test_df': test_df,
        'X_train_resampled': X_train_res,
        'y_train_resampled': y_train_res,
        'X_train_raw': X_train_raw,
        'y_train_raw': y_train_raw,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_names,
        'preprocessor': preprocessor
    }
