"""
Data generation module for Sterling Bank Nigeria Credit Risk Analysis.
Generates realistic, domain-grounded synthetic loan portfolios for:
1. Personal (Salary Earner) Loans (Table 3.1)
2. Retail (SME) Loans (Table 3.2)
Includes realistic financial distributions, correlated default likelihoods,
and typical banking class imbalance.
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42

def generate_personal_loans(n_samples: int = 5000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generates Personal Loan portfolio for salary earners.
    Features:
    - applicant_age: Age in years (21 - 65)
    - housing_status: RENT, MORTGAGE, OWN
    - employment_length: Years in formal employment (0 - 35)
    - annual_income: Gross annual income in NGN (e.g., 1.5M - 40M)
    - debt_to_income_ratio: Total debt service / gross income (%) (5 - 80)
    - credit_bureau_score: Credit score (300 - 850)
    - existing_external_loan_status: None, Performing, Defaulting
    - loan_amount: Principal requested in NGN (100k - 10M)
    - loan_status: 0 (Performing), 1 (Default)
    """
    np.random.seed(seed)
    
    applicant_age = np.random.randint(22, 63, size=n_samples)
    
    housing_choices = ['RENT', 'MORTGAGE', 'OWN']
    housing_probs = [0.55, 0.25, 0.20]
    housing_status = np.random.choice(housing_choices, size=n_samples, p=housing_probs)
    
    # Employment length is bounded by age - 20
    max_emp = np.maximum(1, applicant_age - 21)
    employment_length = np.round(np.random.beta(2, 5, size=n_samples) * max_emp, 1)
    
    # Annual income log-normally distributed in NGN (Mean around 4.5M NGN)
    log_income = np.random.normal(15.2, 0.75, size=n_samples)
    annual_income = np.round(np.exp(log_income), -3) # Rounded to nearest thousand
    # Clip between 1,200,000 and 45,000,000 NGN
    annual_income = np.clip(annual_income, 1_200_000, 45_000_000)
    
    # Debt to Income Ratio (%)
    # Younger/lower income tend to have higher DTI
    dti_base = np.random.beta(2.5, 4.0, size=n_samples) * 75
    # Add adjustment for low income
    income_factor = np.where(annual_income < 3_000_000, 8.0, 0.0)
    debt_to_income_ratio = np.clip(np.round(dti_base + income_factor, 1), 5.0, 85.0)
    
    # Credit Bureau Score (300 to 850) - normal around 640
    credit_score = np.random.normal(635, 95, size=n_samples)
    # Higher DTI lowers credit score
    credit_score -= (debt_to_income_ratio - 35) * 1.5
    credit_bureau_score = np.clip(np.round(credit_score), 300, 850).astype(int)
    
    # Existing external loan status
    ext_choices = ['None', 'Performing', 'Defaulting']
    ext_probs = [0.45, 0.45, 0.10]
    existing_external_loan_status = np.random.choice(ext_choices, size=n_samples, p=ext_probs)
    
    # Loan amount requested (proportional to income, 15% to 50% of annual income)
    loan_pct = np.random.uniform(0.15, 0.50, size=n_samples)
    loan_amount = np.round(annual_income * loan_pct, -4)
    loan_amount = np.clip(loan_amount, 200_000, 15_000_000)
    
    # Latent default probability formula (Logistic function with realistic non-linear risk)
    # Risk factors: high DTI, low credit score, defaulting external loans, low employment length, rent
    z = (
        - 3.2
        + 0.055 * (debt_to_income_ratio - 35)
        - 0.010 * (credit_bureau_score - 600)
        - 0.06 * employment_length
        + (housing_status == 'RENT') * 0.35
        - (housing_status == 'OWN') * 0.40
        + (existing_external_loan_status == 'Defaulting') * 1.85
        + (existing_external_loan_status == 'Performing') * 0.15
        + (loan_amount / annual_income > 0.40) * 0.45
    )
    
    prob_default = 1 / (1 + np.exp(-z))
    loan_status = (np.random.rand(n_samples) < prob_default).astype(int)
    
    # Inject ~2% realistic missing values in non-critical columns to test imputation pipeline
    missing_idx_inc = np.random.choice(n_samples, size=int(n_samples * 0.025), replace=False)
    missing_idx_emp = np.random.choice(n_samples, size=int(n_samples * 0.020), replace=False)
    
    df = pd.DataFrame({
        'applicant_age': applicant_age,
        'housing_status': housing_status,
        'employment_length': employment_length,
        'annual_income': annual_income,
        'debt_to_income_ratio': debt_to_income_ratio,
        'credit_bureau_score': credit_bureau_score,
        'existing_external_loan_status': existing_external_loan_status,
        'loan_amount': loan_amount,
        'loan_status': loan_status
    })
    
    df.loc[missing_idx_inc, 'annual_income'] = np.nan
    df.loc[missing_idx_emp, 'employment_length'] = np.nan
    
    return df


def generate_retail_sme_loans(n_samples: int = 5000, seed: int = RANDOM_SEED + 10) -> pd.DataFrame:
    """
    Generates Retail (SME) Loan portfolio for small and medium enterprises.
    Features:
    - business_age_years: Years since CAC incorporation (0.5 - 25)
    - nature_of_business: Trade, Tech, Agriculture, Manufacturing, Services, Construction
    - business_region: South-West, South-South, North-Central, North-West, South-East, North-East
    - facility_type: Term Loan, Overdraft, Asset Finance, Invoice Discounting
    - loan_amount: Requested principal (NGN 500k - 50M)
    - interest_rate: Annual lending rate (%) (18% - 34%)
    - existing_external_loan_status: None, Performing, Defaulting
    - loan_status: 0 (Performing), 1 (Default)
    """
    np.random.seed(seed)
    
    # Business age (years): log-normal with mean ~ 4.5 years
    business_age_years = np.round(np.random.exponential(scale=4.5, size=n_samples) + 0.5, 1)
    business_age_years = np.clip(business_age_years, 0.5, 28.0)
    
    sectors = ['Trade & Commerce', 'Agriculture & Agro-allied', 'Manufacturing', 'Services', 'Information Technology', 'Construction']
    sector_probs = [0.35, 0.20, 0.15, 0.15, 0.08, 0.07]
    nature_of_business = np.random.choice(sectors, size=n_samples, p=sector_probs)
    
    regions = ['South-West', 'North-Central', 'South-South', 'North-West', 'South-East', 'North-East']
    region_probs = [0.42, 0.22, 0.14, 0.10, 0.08, 0.04]
    business_region = np.random.choice(regions, size=n_samples, p=region_probs)
    
    facilities = ['Term Loan', 'Overdraft', 'Asset Finance', 'Invoice Discounting']
    facility_probs = [0.45, 0.30, 0.15, 0.10]
    facility_type = np.random.choice(facilities, size=n_samples, p=facility_probs)
    
    # Loan amount in NGN (500k to 50M)
    log_amt = np.random.normal(15.8, 0.85, size=n_samples)
    loan_amount = np.round(np.exp(log_amt), -4)
    loan_amount = np.clip(loan_amount, 500_000, 50_000_000)
    
    # Interest rate (%): in Nigeria commercial lending rates range between 20% and 32%
    base_rate = np.random.normal(25.5, 3.2, size=n_samples)
    # Higher loan amount or younger business receives slightly higher risk margin
    interest_rate = np.clip(np.round(base_rate + np.where(business_age_years < 2.0, 2.5, 0.0), 1), 18.0, 35.0)
    
    ext_choices = ['None', 'Performing', 'Defaulting']
    ext_probs = [0.40, 0.48, 0.12]
    existing_external_loan_status = np.random.choice(ext_choices, size=n_samples, p=ext_probs)
    
    # Latent SME default probability formula
    # Higher risk: young business, high interest rate, defaulting external loan, high risk sector (Construction/Agro)
    sector_risk = {
        'Trade & Commerce': 0.10,
        'Agriculture & Agro-allied': 0.50,
        'Manufacturing': 0.20,
        'Services': -0.10,
        'Information Technology': -0.25,
        'Construction': 0.45
    }
    sec_risk_vector = np.array([sector_risk[s] for s in nature_of_business])
    
    z_sme = (
        - 2.8
        - 0.12 * business_age_years
        + 0.08 * (interest_rate - 24.0)
        + (loan_amount > 20_000_000) * 0.40
        + sec_risk_vector
        + (existing_external_loan_status == 'Defaulting') * 1.95
        + (existing_external_loan_status == 'Performing') * 0.10
        + (facility_type == 'Overdraft') * 0.25
    )
    
    prob_default_sme = 1 / (1 + np.exp(-z_sme))
    loan_status = (np.random.rand(n_samples) < prob_default_sme).astype(int)
    
    # Inject missing values for business_age_years (~1.8%)
    missing_idx_age = np.random.choice(n_samples, size=int(n_samples * 0.018), replace=False)
    
    df_sme = pd.DataFrame({
        'business_age_years': business_age_years,
        'nature_of_business': nature_of_business,
        'business_region': business_region,
        'facility_type': facility_type,
        'loan_amount': loan_amount,
        'interest_rate': interest_rate,
        'existing_external_loan_status': existing_external_loan_status,
        'loan_status': loan_status
    })
    
    df_sme.loc[missing_idx_age, 'business_age_years'] = np.nan
    return df_sme


def save_datasets(output_dir: str = "data"):
    os.makedirs(output_dir, exist_ok=True)
    df_personal = generate_personal_loans(n_samples=6000)
    df_retail = generate_retail_sme_loans(n_samples=6000)
    
    personal_path = os.path.join(output_dir, "personal_loans.csv")
    retail_path = os.path.join(output_dir, "retail_sme_loans.csv")
    
    df_personal.to_csv(personal_path, index=False)
    df_retail.to_csv(retail_path, index=False)
    
    print(f"[OK] Saved Personal Loans: {personal_path} (Shape: {df_personal.shape}, Default rate: {df_personal['loan_status'].mean():.2%})")
    print(f"[OK] Saved Retail SME Loans: {retail_path} (Shape: {df_retail.shape}, Default rate: {df_retail['loan_status'].mean():.2%})")

if __name__ == "__main__":
    save_datasets()
