# CHAPTER FIVE: SUMMARY, CONCLUSION, AND RECOMMENDATIONS

## 5.1 Summary of Findings
This study set out to develop and evaluate an interpretable machine learning framework for credit risk assessment and loan default prediction tailored for Sterling Bank Nigeria. Confronted with the industry-wide challenges of severe class imbalance, model opacity ("black-box" dilemma), and asymmetric classification costs, the research designed, trained, evaluated, and operationalized a **Dual-Model Architecture** separating the **Personal Loan Portfolio (Salary Earners)** from the **Retail Loan Portfolio (Small and Medium-Sized Enterprises - SMEs)**.

The empirical findings directly address the three core research questions established in Chapter One:

### 1. Findings on Research Question 1 (Data Preprocessing & Resampling via SMOTE)
*RQ1: How effectively can data preprocessing and synthetic resampling techniques, such as SMOTE, address class imbalance and help identify important risk factors in Sterling Bank’s personal and retail loan data?*
- In their native state, loan portfolios exhibited significant class imbalance: defaults accounted for only **11.8%** of Personal loans and **15.2%** of Retail SME loans.
- Median imputation successfully preserved the non-normal, skewed distribution of financial variables (gross annual salary and requested loan amounts) without inducing synthetic outliers.
- An empirical ablation benchmark revealed that training tree-based models on raw imbalanced data resulted in unacceptably low default detection rates (Recall of only 45.77% on Personal loans and 51.10% on SME loans).
- The deployment of SMOTE strictly on the training partitions resolved class imbalance, boosting XGBoost Recall to **78.17% (Personal)** and **80.77% (Retail SME)** on unseen holdout test sets.
- Strict isolation of resampling to the training set ensured zero data leakage, confirming that SMOTE substantively improves default detection without sacrificing real-world generalizability.

### 2. Findings on Research Question 2 (Ensemble Machine Learning vs. Traditional Statistical Models)
*RQ2: How does the performance of an ensemble machine learning model, such as XGBoost, compare with traditional statistical models in predicting loan defaults when assessed using appropriate credit risk evaluation metrics?*
- The primary ensemble architecture, **Extreme Gradient Boosting (XGBoost)**, achieved superior performance across all primary discrimination and precision-recall metrics on both credit portfolios:
  - **Personal Loans:** XGBoost attained an **ROC-AUC of 0.8924**, a **PR-AUC of 0.6582**, an **Accuracy of 89.08%**, and an **F1-Score of 0.6588**.
  - **Retail SME Loans:** XGBoost attained an **ROC-AUC of 0.8768**, a **PR-AUC of 0.6174**, an **Accuracy of 87.25%**, and an **F1-Score of 0.6622**.
- LightGBM demonstrated competitive performance, closely tracking XGBoost, while Random Forest proved robust in specificity.
- The traditional baseline, **Logistic Regression**, exhibited the lowest discriminatory capability (ROC-AUC of 0.8241 and 0.8123) and produced low precision (32.89% and 34.69%). Its linear assumptions failed to capture critical non-linear borrower interactions (e.g., the compounded default risk when high Debt-to-Income ratios coincide with low credit scores), leading to excessive false-positive loan rejections.
- Incorporating an **Asymmetric Financial Cost Matrix** demonstrated that the default 0.50 cutoff was financially suboptimal. Calibrating the decision threshold to **$\tau^* = 0.35$ for Personal loans** and **$\tau^* = 0.33$ for Retail SME loans** minimized Expected Financial Loss, achieving monetary savings of **NGN 11.18 Million** and **NGN 64.80 Million** respectively on the test cohorts.

### 3. Findings on Research Question 3 (Explainable AI via SHAP & Regulatory Compliance)
*RQ3: How can a SHAP-based Explainable Artificial Intelligence framework help explain loan default predictions and support transparent and auditable credit risk assessments in the context of relevant CBN guidelines?*
- Integrating SHAP (SHapley Additive exPlanations) successfully demystified the internal mechanics of the XGBoost ensembles at both global and local operational levels.
- **Global Interpretability:** Identified the principal systemic drivers of default:
  - *Personal Portfolio:* Debt-to-Income (DTI) ratio, Credit Bureau Score, and concurrent External Loan Delinquencies dominated predictive attributions.
  - *Retail SME Portfolio:* Concurrent External Facility Delinquency, Facility Interest Rate, and Business Age (years since CAC incorporation) were the decisive determinants of commercial credit risk.
- **Local Interpretability & Adverse Action Notices:** For individual applicants, SHAP waterfall plots decomposed risk scores into additive, mathematically defensible feature contributions. For rejected applicants, the framework automatically formulated clear, plain-language rejection reasons compliant with the **Central Bank of Nigeria (CBN) Consumer Protection Framework (Section 3.2)**.
- **Operational Delivery:** The framework was successfully engineered into a production-grade **FastAPI REST API** and an interactive **Streamlit Underwriting Dashboard**, empowering risk analysts and compliance auditors to inspect individual applicant attributions in real time.

---

## 5.2 Conclusion
This research proves that the traditional dichotomy between high predictive performance and model interpretability in retail banking is a false compromise. By integrating Extreme Gradient Boosting (XGBoost) with out-of-fold SMOTE resampling and game-theoretic SHAP explainability, this study developed an auditable, high-accuracy decisioning engine tailored for Sterling Bank Nigeria.

The research demonstrates that segregating credit underwriting into distinct Personal (Salary Earner) and Retail (SME) portfolios is vital: individual borrowers and commercial enterprises operate under fundamentally distinct cash flow mechanics, risk factors, and default trajectories. Furthermore, replacing arbitrary 0.50 classification cutoffs with loss-minimizing financial thresholds aligns machine learning models directly with the commercial banking objective: minimizing capital write-offs while preserving profitable lending growth.

Ultimately, this study bridges academic data science and commercial banking operations, delivering a deployable system that enhances credit underwriting efficiency, protects bank capital reserves, and fulfills regulatory transparency mandates within the Nigerian financial ecosystem.

---

## 5.3 Practical Recommendations for Sterling Bank Nigeria
Based on the empirical findings, the following actionable recommendations are proposed for Sterling Bank Nigeria:

1. **Adopt the Dual-Model Credit Architecture:**
   Sterling Bank should retire consolidated retail scorecards in favor of segregated Personal and SME underwriting engines. Distinct risk scorecards ensure that commercial attributes (CAC incorporation age, facility interest rate, economic sector) are not conflated with personal salary dynamics.
2. **Implement Financial Loss-Calibrated Cutoff Thresholds:**
   The bank's credit risk committee should replace standard 0.50 decision thresholds with loss-calibrated cutoffs ($\tau^* \approx 0.35$ for Personal and $\tau^* \approx 0.33$ for SME lending). Because capital write-offs on defaulted facilities cost over five times more than forgone interest margins, adopting these thresholds will directly reduce Non-Performing Loans (NPLs) and save tens of millions of Naira annually.
3. **Automate Adverse Action Disclosures Using SHAP:**
   Sterling Bank should integrate the automated SHAP attribution module into its customer notification pipeline. When a credit application is declined, the system should generate a statutory adverse action statement citing the top 3 contributing factors (e.g., excessive DTI or credit bureau delinquencies). This protects the bank against regulatory censure, builds customer trust, and guides borrowers on steps needed to improve their creditworthiness.
4. **Deploy Dynamic Risk-Adjusted Facility Pricing:**
   Given that high interest rates significantly increase default risk among SMEs (Mean $|\text{SHAP}| = 1.05$), the bank should structure facilities with tiered pricing: commercial borrowers exhibiting strong CAC stability and clean bureau histories should receive interest rate discounts, lowering their debt service burden and reducing default likelihood.
5. **Establish a Human-in-the-Loop Tri-Tier Decision System:**
   Underwriting decisions should be structured into three operational tiers:
   - **Automated Approval:** Default Probability $< 0.20$ (immediate disbursement).
   - **Manual Senior Review:** Default Probability between $0.20$ and $0.35$ (requires human underwriter inspection using the SHAP dashboard).
   - **Automated Adverse Action:** Default Probability $\ge 0.35$ (system rejection with clear explanations).

---

## 5.4 Recommendations for Regulatory Oversight (Central Bank of Nigeria & NDIC)
To promote financial system stability and foster responsible financial technology adoption across Nigeria's commercial banking sector, the Central Bank of Nigeria (CBN) and Nigeria Deposit Insurance Corporation (NDIC) should consider the following policy directions:

1. **Mandate Algorithmic Explainability Standards (XAI Guidelines):**
   The CBN should issue explicit regulatory directives requiring all deposit money banks utilizing machine learning or artificial intelligence in credit scoring to employ mathematically auditable explainability frameworks (such as SHAP or Integrated Gradients). Pure "black-box" decision models should be disallowed for credit underwriting.
2. **Promote Asymmetric Cost Matrix Benchmarks:**
   Regulatory stress-testing and model validation guidelines (such as Basel II/III internal ratings-based approaches) should encourage banks to evaluate credit models using Expected Financial Loss matrices rather than relying exclusively on overall classification accuracy or standard ROC-AUC.
3. **Conduct Regular Fairness and Algorithmic Bias Audits:**
   Regulatory supervisors should require commercial banks to submit annual fairness and disparate impact audits. SHAP global feature importances should be inspected to ensure that models do not inadvertently utilize demographic proxies that could disadvantage protected classes or underserved geographic regions.

---

## 5.5 Limitations and Future Research Directions
While this research achieved its primary objectives, certain limitations suggest productive avenues for future research:

1. **Macroeconomic Covariate Integration:**
   This study utilized historical retail facility data. Future research should integrate macroeconomic time-series indicators—such as the Monetary Policy Rate (MPR), headline inflation, and foreign exchange rates—as dynamic covariates to evaluate model robustness across economic shocks.
2. **Incorporation of Alternative Non-Traditional Data:**
   Many prospective borrowers and micro-enterprises in Nigeria lack comprehensive credit bureau files ("thin-file" applicants). Future studies should investigate the fusion of alternative data streams (e.g., mobile money transaction velocity, utility bill payments, and POS terminal cash flow) into the ensemble feature pipeline.
3. **Concept Drift and Continuous Online Learning:**
   Borrower repayment behavior changes over time due to seasonal business cycles and policy changes. Future research should implement continuous model monitoring frameworks that evaluate population stability index (PSI) and concept drift, triggering automated retraining when feature distributions shift.

---

## 5.6 Operational Implementation Roadmap
To facilitate the transition from academic prototype to live enterprise operation within Sterling Bank Nigeria, a three-phase deployment roadmap is outlined in Table 5.1:

#### Table 5.1: Enterprise Operational Implementation Roadmap
| Phase | Duration | Core Milestones & Deliverables | Primary Stakeholders |
| :--- | :--- | :--- | :--- |
| **Phase 1: Validation & Shadow Testing** | Months 1 – 3 | - Integrate FastAPI backend into staging data warehouse.<br>- Run models in passive "shadow mode" parallel to existing scorecards.<br>- Validate predictions against 90-day delinquency outcomes.<br>- Conduct security, data privacy, and NDPR compliance audits. | Data Science Team, IT Infrastructure, Risk Management |
| **Phase 2: Champion-Challenger Pilot** | Months 4 – 6 | - Route 25% of retail loan applications through the ML decision engine.<br>- Train branch credit officers and underwriters on the Streamlit dashboard.<br>- Audit automated adverse action notices for CBN compliance.<br>- Evaluate false positive rates and underwriter overrides. | Credit Underwriting, Compliance Officers, Branch Operations |
| **Phase 3: Enterprise Rollout & Continuous Monitoring** | Months 7 – 12 | - Full enterprise rollout across all Sterling Bank retail channels.<br>- Implement automated CI/CD retraining pipelines triggered by model drift.<br>- Establish executive KPI dashboard tracking portfolio NPL ratios and capital savings. | Chief Risk Officer (CRO), Board Risk Committee, DevOps Team |

---

## 5.7 Final Remarks
The combination of advanced gradient boosting algorithms, synthetic minority resampling, financial loss calibration, and explainable artificial intelligence establishes a powerful paradigm for contemporary banking analytics. By empowering credit underwriters with transparent risk insights and minimizing expected financial write-offs, the framework developed in this study demonstrates that modern data science can simultaneously safeguard commercial bank solvency and promote equitable, transparent credit access across Nigeria.
