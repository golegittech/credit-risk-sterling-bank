# CHAPTER THREE: RESEARCH METHODOLOGY AND SYSTEM DESIGN

## 3.1 Research Design and Dual-Model Architecture
This study adopts a supervised machine learning approach to develop an interpretable framework for credit risk assessment and loan default prediction. The research is designed around two distinct loan portfolios: Personal loans, which primarily involve individual salary earners, and Retail loans, which include loans granted to small and medium-sized enterprises (SMEs).

The decision to separate the two portfolios is based on the differences in their financial characteristics and the factors that may influence repayment behavior. Personal loan repayment is generally associated with factors such as income stability, existing financial obligations and individual credit history. In contrast, SME loan repayment may be influenced by business performance, cash flow, industry conditions and the operating environment.

Combining both portfolios into a single modelling framework could make it more difficult to capture these differences effectively. Therefore, this study adopts a dual-model architecture in which each portfolio is processed and modelled independently.

Two separate Extreme Gradient Boosting (XGBoost) classification models are developed alongside baseline algorithms. The first model assesses the likelihood of default among personal loan borrowers, while the second evaluates default risk within the Retail (SME) portfolio. Each model undergoes its own data preprocessing, feature selection, training and evaluation procedures.

---

## 3.2 Dataset Description and Feature Selection
The study uses two separate loan portfolios modeled on historical credit operations at Sterling Bank Nigeria:

### Table 3.1: Personal (Salary Earner) Feature Schema
| Variable Category | Feature Name | Data Type | Description & Operational Metric |
| :--- | :--- | :--- | :--- |
| **Demographic** | Applicant age | Integer | Age of the primary applicant (years: 21–65) |
| | Housing status | Categorical | Home ownership status (OWN, RENT, MORTGAGE) |
| | Employment length | Continuous | Duration of current formal employment (years) |
| **Financial Risk** | Annual income | Continuous | Total verified gross yearly salary (NGN) |
| | Debt to income ratio | Continuous | Total monthly debt / Gross monthly income (%) |
| **Behavioral** | Credit bureau score | Continuous | Numerical score from national credit bureaus (300–850) |
| | Existing external loan status | Categorical | Status of concurrent external facilities (None, Performing, Defaulting) |
| **TARGET** | Loan status | Binary (0/1) | 0 = Performing, 1 = Non-Performing (Default) |

### Table 3.2: Retail (SME) Feature Schema
| Variable Category | Feature Name | Data Type | Description & Operational Metric |
| :--- | :--- | :--- | :--- |
| **Business Profile** | Business age years | Continuous | Years since official CAC incorporation |
| | Nature of business | Categorical | Economic sector (Trade, Tech, Agriculture, Manufacturing, Services, Construction) |
| | Business region | Categorical | Primary operational region (South-West, North-Central, etc.) |
| **Facility Request** | Facility type | Categorical | Product type (Term Loan, Overdraft, Asset Finance, Invoice Discounting) |
| | Loan amount | Continuous | Total requested principal amount (NGN) |
| | Interest rate | Continuous | Annual interest rate applied to the facility (%) |
| **Behavioral** | Existing external loan status | Categorical | Status of concurrent external facilities (None, Performing, Defaulting) |
| **TARGET** | Loan status | Binary (0/1) | 0 = Performing, 1 = Non-Performing (Default) |

---

## 3.3 Data Preprocessing and Feature Engineering
- **Missing Value Imputation:** Numerical variables (e.g., `annual_income`, `business_age_years`) imputed with the median to withstand financial skewness. Categorical variables imputed with the mode.
- **Categorical Encoding:** One-Hot Encoding applied to nominal variables without intrinsic ranking.
- **Numerical Scaling:** Standard Scaler ($X_{scaled} = \frac{X - \mu}{\sigma}$) fitted on the training split for scale-sensitive models like Logistic Regression.
- **Feature Selection & Multicollinearity:** Variance Inflation Factor (VIF) evaluated to detect collinearity ($VIF_j = \frac{1}{1 - R_j^2}$).

---

## 3.4 Class Imbalance Mitigation Strategy Using SMOTE
Class imbalance is addressed using SMOTE (Chawla et al., 2002).
- **Data Leakage Prevention:** SMOTE is applied strictly on the training partition. The validation and test datasets retain their native, unaltered class distribution to evaluate real-world generalization performance.

---

## 3.5 Machine Learning Model Formulations
Four supervised classification algorithms are evaluated:
1. **Logistic Regression (Baseline):** Linear log-odds estimator with balanced class weighting.
2. **Random Forest:** Bagging ensemble of 200 decision trees.
3. **Extreme Gradient Boosting (XGBoost):** Primary architecture combining second-order gradient boosting with L1/L2 shrinkage.
4. **LightGBM:** Fast leaf-wise gradient boosting classifier.

---

## 3.6 Model Evaluation and Financial Cost Matrix Framework
In commercial lending, classification errors have asymmetric consequences:
- **False Negative ($C_{FN}$):** Defaulter misclassified as creditworthy $\implies$ Direct capital write-off ($C_{FN} = EAD \times LGD$).
- **False Positive ($C_{FP}$):** Good applicant rejected $\implies$ Forgone interest opportunity margin.

$$\text{Total Expected Financial Loss} = (C_{FN} \times \text{FN}) + (C_{FP} \times \text{FP})$$

Decision cutoff thresholds are calibrated across a grid to minimize this monetary loss rather than assuming the arbitrary 0.50 cutoff.

---

## 3.7 Explainable AI (XAI) and API Deployment Architecture
- **SHAP Engine:** Calculates additive game-theoretic attributions ($f(x) = \phi_0 + \sum_{j=1}^M \phi_j$) globally (beeswarm / summary) and locally (waterfall plots).
- **FastAPI Backend:** Exposes RESTful endpoints (`/predict`, `/explain`, `/metrics`).
- **Streamlit Interactive Dashboard:** Delivers a web interface for credit underwriters and auditors.
- **Render Deployment:** Cloud deployment configuration via Docker and Procfile.
