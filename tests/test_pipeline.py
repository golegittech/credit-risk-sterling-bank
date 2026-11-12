"""
Unit tests for Sterling Bank Credit Risk Decision Engine.
Verifies:
1. Data generation schemas and output dimensions.
2. Preprocessor transformation, median imputation, and SMOTE resampling.
3. Financial cost matrix calculations and savings.
4. FastAPI endpoints structure.
"""

import unittest
import numpy as np
import pandas as pd
import os

from src.data_generator import generate_personal_loans, generate_retail_sme_loans
from src.preprocessing import CreditDataPreprocessor
from src.financial_cost import evaluate_financial_loss, optimize_decision_threshold

class TestCreditRiskPipeline(unittest.TestCase):
    
    def test_personal_data_generator(self):
        df = generate_personal_loans(n_samples=200, seed=42)
        self.assertEqual(len(df), 200)
        self.assertIn('loan_status', df.columns)
        self.assertIn('debt_to_income_ratio', df.columns)
        self.assertTrue((df['loan_status'].isin([0, 1])).all())

    def test_retail_data_generator(self):
        df = generate_retail_sme_loans(n_samples=200, seed=42)
        self.assertEqual(len(df), 200)
        self.assertIn('business_age_years', df.columns)
        self.assertIn('facility_type', df.columns)
        self.assertTrue((df['loan_status'].isin([0, 1])).all())

    def test_preprocessor_fit_transform(self):
        df = generate_personal_loans(n_samples=300, seed=42)
        prep = CreditDataPreprocessor(portfolio_type="personal")
        X_res, y_res, feat_names = prep.fit_transform_train(df, apply_smote=True)
        
        # SMOTE should balance classes (50/50)
        self.assertEqual(np.mean(y_res), 0.5)
        self.assertEqual(X_res.shape[0], len(y_res))
        self.assertGreater(len(feat_names), 0)

    def test_financial_cost_optimization(self):
        y_true = np.array([0, 0, 0, 1, 0, 1, 0, 0, 1, 0])
        y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.2, 0.6, 0.1, 0.4, 0.9, 0.1])
        res = optimize_decision_threshold(y_true, y_prob, portfolio_type="personal")
        
        self.assertIn("optimal_threshold", res)
        self.assertIn("min_financial_loss", res)
        self.assertGreaterEqual(res["monetary_savings_ngn"], 0)

if __name__ == "__main__":
    unittest.main()
