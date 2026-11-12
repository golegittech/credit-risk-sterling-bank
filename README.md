# Credit Risk Analysis and Loan Default Prediction Using Machine Learning Models and Data Visualization

**Candidate:** Mbaegbu Chinedum Grace  
**Matriculation Number:** 2025/A/DSC/0058  
**Degree:** Bachelor of Science (Honours) Degree in Data Science  
**Department:** Department of Data Science, School of Computing  
**Institution:** Miva Open University, Abuja, Nigeria  
**Supervisor:** Dr. A. S. Makinde  
**Date:** August, 2026  

---

## 1. Project Overview

Credit risk management is vital for the safety and solvency of commercial banking institutions. This project delivers an end-to-end, interpretable machine learning system tailored for **Sterling Bank Nigeria**, specifically separating retail credit into a **Dual-Model Architecture**:
1. **Personal Loan Portfolio (Salary Earners):** Evaluates demographic stability, verified monthly gross salary, Debt-to-Income (DTI) ratio, bureau credit score, and concurrent external debt obligations.
2. **Retail Loan Portfolio (SMEs / Commercial Enterprises):** Evaluates years since CAC incorporation, sector economic risk, geopolitical region, facility structure (overdraft, term loan), and commercial lending interest rates.

The framework resolves two primary operational hurdles in credit analytics:
- **Severe Class Imbalance:** Mitigated using strictly out-of-fold **Synthetic Minority Over-sampling Technique (SMOTE)** to prevent data leakage.
- **Model Opacity ("Black-Box" dilemma):** Resolved using **SHAP (SHapley Additive exPlanations)**, delivering global risk factor auditing and local adverse action explanations consistent with the **Central Bank of Nigeria (CBN) Consumer Protection Framework**.
- **Financial Asymmetry:** Incorporates an expected financial loss matrix ($C_{FN} = EAD \times LGD$ vs. $C_{FP} = \text{Lost Margin}$) that calibrates cutoffs below the naive 0.50 default threshold, unlocking substantial capital savings.

---

## 2. System Architecture

```text
               +-------------------------------------------+
               |  Sterling Bank Retail Credit Portfolios   |
               |     (Personal Salary & Retail SME)        |
               +---------------------+---------------------+
                                     |
                                     v
               +-------------------------------------------+
               | Preprocessing & Leakage-Free SMOTE Engine |
               +---------------------+---------------------+
                                     |
                                     v
               +-------------------------------------------+
               |   Model Benchmarking & Selection Engine   |
               | (Logistic Regression, RF, LightGBM, XGB)  |
               +---------------------+---------------------+
                                     |
                                     v
               +-------------------------------------------+
               | Financial Cost Matrix & Optimal Threshold |
               +---------------------+---------------------+
                                     |
                                     v
               +-------------------------------------------+
               |   SHAP XAI & Adverse Action Generator     |
               +----------+--------------------+-----------+
                          |                    |
                          v                    v
              +----------------------+  +---------------------+
              | FastAPI Backend REST |  | Streamlit Dashboard |
              |  (Render Deployable) |  |   for Underwriters  |
              +----------------------+  +---------------------+
```

---

## 3. Directory Layout

```
chinedum/
├── api/
│   └── main.py                     # FastAPI REST API
├── app/
│   └── streamlit_app.py            # Streamlit Underwriter & Auditor Dashboard
├── data/
│   ├── personal_loans.csv          # Personal loan portfolio
│   └── retail_sme_loans.csv        # Retail SME loan portfolio
├── dissertation/
│   ├── Chapter_1_Introduction.md
│   ├── Chapter_2_Literature_Review.md
│   ├── Chapter_3_Methodology_System_Design.md
│   ├── Chapter_4_Results_and_Discussion.md
│   ├── Chapter_5_Conclusion_and_Recommendations.md
│   └── Complete_Dissertation_Mbaegbu_Chinedum.md
├── models/                         # Saved preprocessors and trained models (.joblib)
├── results/                        # JSON performance metrics and high-res figures
├── src/
│   ├── data_generator.py           # Realistic portfolio synthesis
│   ├── preprocessing.py            # Imputation, encoding, scaling, SMOTE
│   ├── train_models.py             # 4-model benchmarking & financial matrix
│   ├── financial_cost.py           # Cost matrix & threshold optimization
│   └── explainability.py           # SHAP global/local XAI engine
├── Dockerfile                      # Container specification
├── Procfile                        # Cloud / Render deployment config
├── requirements.txt                # Python dependencies
├── run_app.bat                     # Windows 1-click execution script
└── README.md                       # Project documentation
```

---

## 4. Setup and Installation

### Prerequisites
- Python 3.10+ (Configured with pip)
- Git (Optional)

### Step 1: Install Requirements
```bash
pip install -r requirements.txt
```

### Step 2: Run Data Generation, Training, and Explainability
```bash
python -m src.data_generator
python -m src.train_models
python -m src.explainability
```

### Step 3: Launch the Interactive Dashboard
```bash
streamlit run app/streamlit_app.py
```
Access at: `http://localhost:8501`

### Step 4: Run the FastAPI Service (Optional)
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger docs at: `http://localhost:8000/docs`

---

## 5. Central Bank of Nigeria (CBN) Compliance Features
- **Adverse Action Explanations:** For any declined loan, the system automatically extracts top positive SHAP contributors to articulate defensible, plain-language rejection reasons.
- **Auditable Fair Lending:** Real-time visibility into feature weights guarantees models do not leverage prohibited demographic proxies.
- **Dual-Control Human-in-the-Loop:** Borderline risk scores trigger a `MANUAL REVIEW` flag rather than instant rejection.
