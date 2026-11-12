# CREDIT RISK ANALYSIS AND LOAN DEFAULT PREDICTION USING MACHINE LEARNING MODEL AND DATA VISUALIZATION

**BY:**  
**MBAEGBU CHINEDUM GRACE**  
**MATRICULATION NUMBER: 2025/A/DSC/0058**  

A PROJECT SUBMITTED TO THE DEPARTMENT OF DATA SCIENCE, SCHOOL OF COMPUTING, MIVA OPEN UNIVERSITY ABUJA, NIGERIA.  
IN PARTIAL FULFILLMENT OF THE REQUIREMENTS FOR THE AWARD OF THE BACHELOR OF SCIENCE (HONOURS) DEGREE IN DATA SCIENCE  

**AUGUST, 2026**  

---

## CERTIFICATION

This is to certify that I am responsible for the work submitted in this Project, that the original work is mine, except as specified in the acknowledgement and references, and that neither the project nor the original work contained therein has been submitted to this University or any other institution for the award of a degree.

\
______________________________  
**MBAEGBU CHINEDUM GRACE**  
2025/A/DSC/0058  

---

## APPROVAL

This project has been approved for the Department of Data Science, School of Computing, Miva Open University, Abuja, Nigeria.

\
**Dr. A. S. Makinde**  
Supervisor  
Signature and Date: ……………………………  

\
**Dr. Chinonso G. Alaebo**  
Head of Department  
Signature and Date: …………………………..  

\
**Emeka Ogbuju, PhD**  
Dean, School of Computing  
Signature and Date: …………………………..  

---

## DEDICATION

This project is dedicated to the Almighty God for the abundant grace, strength, and divine wisdom bestowed upon me throughout this academic journey; to my beloved mother and siblings for their constant emotional support, sacrifices, and unconditional love; and to my lecturers for the knowledge, technical discipline, and data science skills imparted during my tenure at Miva Open University.

---

## ACKNOWLEDGMENTS

I would like to express my deepest gratitude to my supervisor, **Dr. A. S. Makinde**, for his clear guidance, constructive critiques, academic patience, and unwavering support throughout the conception, analysis, and drafting of this research.

I also sincerely thank the lecturers and administrative staff of the Department of Data Science, School of Computing at Miva Open University for fostering an intellectually rigorous academic environment and providing the institutional resources that enabled the completion of this research.

Most importantly, I am deeply grateful to my elder sister **Angela**, and my dear friends **Samuel** and **Eyo** for their constant encouragement, technical discussions, moral advice, and motivation, which provided the emotional resilience needed to overcome complex research challenges.

Finally, I am profoundly thankful for my own persistence, focus, and dedication in seeing this work through to successful completion.

---

## ABSTRACT

Credit risk assessment is foundational to commercial banking stability and asset quality. Traditional statistical scorecards (such as Logistic Regression) frequently struggle to capture non-linear borrower interactions, remain susceptible to severe class imbalance (where default events represent a small fraction of loan portfolios), and lack dynamic loss-minimizing threshold calibrations. Furthermore, high-performing ensemble machine learning models often suffer from opacity ("black-box" dilemma), creating regulatory barriers under Central Bank of Nigeria (CBN) transparency mandates. 

This study develops, evaluates, and deploys an interpretable machine learning framework for credit risk analysis and loan default prediction tailored for **Sterling Bank Nigeria**. To address the heterogeneous financial mechanics of different borrower cohorts, the study adopts a **Dual-Model Architecture** independently processing: (1) a **Personal Loan Portfolio (Salary Earners)**, and (2) a **Retail Loan Portfolio (Small and Medium-Sized Enterprises - SMEs)** ($N = 6,000$ historical records each).

Missing values were imputed using distribution-preserving medians, categorical predictors encoded, and class imbalance remediated using the **Synthetic Minority Over-sampling Technique (SMOTE)** applied strictly within training folds to eliminate data leakage. Four classification architectures were trained and benchmarked: Logistic Regression, Random Forest, LightGBM, and Extreme Gradient Boosting (XGBoost).

Empirical findings reveal that SMOTE achieved a transformative uplift in default detection sensitivity, elevating XGBoost Recall from 45.77% to **78.17%** on Personal loans, and from 51.10% to **80.77%** on Retail SME loans on untouched holdout test sets. XGBoost proved to be the superior predictive architecture, attaining an **ROC-AUC of 0.8924** and **PR-AUC of 0.6582** on Personal loans, and an **ROC-AUC of 0.8768** and **PR-AUC of 0.6174** on Retail SME loans, significantly outperforming Logistic Regression (ROC-AUC of 0.8241 and 0.8123).

Evaluating models under an **Asymmetric Financial Cost Matrix** ($C_{FN} = EAD \times LGD$ vs. $C_{FP} = \text{Lost Margin}$) established that naive 0.50 cutoff thresholds were financially suboptimal. Calibrating decision thresholds to $\tau^* = 0.35$ for Personal loans and $\tau^* = 0.33$ for SME facilities minimized Expected Financial Loss, securing capital savings of **NGN 11.18 Million** and **NGN 64.80 Million** respectively on the test samples.

Finally, **SHAP (SHapley Additive exPlanations)** was integrated to provide global systemic feature auditing and local waterfall attributions. The framework demystified model logic—identifying Debt-to-Income (DTI), bureau credit score, and external facility status as primary personal risk factors, and interest rates, commercial history, and external delinquencies as primary SME risk drivers. The local explainability engine automatically synthesizes statutory Adverse Action Notices, satisfying the CBN Consumer Protection Framework. The complete system was operationalized as a production-ready **FastAPI REST backend** paired with an interactive **Streamlit Underwriting Dashboard**, establishing a modern, auditable, and loss-minimizing credit analytics standard for Nigerian commercial banking.

**Keywords:** Credit Risk Assessment, Loan Default Prediction, Extreme Gradient Boosting (XGBoost), Class Imbalance, SMOTE, Explainable AI (XAI), SHAP, Central Bank of Nigeria (CBN), Financial Cost Matrix, Streamlit.

---

## TABLE OF CONTENTS

- Title Page
- Certification
- Approval
- Dedication
- Acknowledgments
- Abstract
- Table of Contents
- List of Tables
- List of Figures
- **CHAPTER ONE: INTRODUCTION**
  - 1.1 Background of the Study
  - 1.2 Problem Statement
  - 1.3 Research Aim and Objectives
  - 1.4 Research Questions
  - 1.5 Significance of the Study
  - 1.6 Scope of the Study
  - 1.7 Limitations of the Study
  - 1.8 Definition of Key Terms
- **CHAPTER TWO: LITERATURE REVIEW**
  - 2.1 Historical Perspectives of Credit Risk Analytics
  - 2.2 Theoretical Framework
  - 2.3 Conceptual Framework
  - 2.4 Review of Related Work
  - 2.5 Gaps in Existing Research
  - 2.6 Summary of the Literature Review
- **CHAPTER THREE: RESEARCH METHODOLOGY AND SYSTEM DESIGN**
  - 3.1 Research Design and Dual-Model Architecture
  - 3.2 Dataset Description and Feature Selection
  - 3.3 Data Preprocessing and Feature Engineering
  - 3.4 Class Imbalance Mitigation Strategy Using SMOTE
  - 3.5 Machine Learning Model Formulations
  - 3.6 Model Evaluation and Financial Cost Matrix Framework
  - 3.7 Explainable AI (XAI) and API Deployment Architecture
- **CHAPTER FOUR: RESULTS, MODEL ANALYSIS, AND DISCUSSION**
  - 4.1 Overview and Experimental Setup
  - 4.2 Exploratory Data Analysis & Preprocessing Findings
  - 4.3 Impact of Class Imbalance Mitigation via SMOTE (RQ1 Analysis)
  - 4.4 Model Performance & Comparative Benchmark (RQ2 Analysis)
  - 4.5 Financial Cost Matrix & Decision Threshold Calibration
  - 4.6 Explainable AI (XAI) & SHAP Interpretability Findings (RQ3 Analysis)
  - 4.7 Operational Deployment & Interactive Underwriting Prototype
  - 4.8 Chapter Summary
- **CHAPTER FIVE: SUMMARY, CONCLUSION, AND RECOMMENDATIONS**
  - 5.1 Summary of Findings
  - 5.2 Conclusion
  - 5.3 Practical Recommendations for Sterling Bank Nigeria
  - 5.4 Recommendations for Regulatory Oversight (CBN & NDIC)
  - 5.5 Limitations and Future Research Directions
  - 5.6 Operational Implementation Roadmap
  - 5.7 Final Remarks
- **REFERENCES**

---

## LIST OF TABLES
- Table 3.1: Personal (Salary Earner) Feature Schema
- Table 3.2: Retail (SME) Feature Schema
- Table 4.0: Hyperparameter Architecture Specifications
- Table 4.1: Descriptive Summary of Personal Loan Portfolio
- Table 4.2: Descriptive Summary of Retail (SME) Loan Portfolio
- Table 4.3: Resampling Partitions Before and After SMOTE
- Table 4.4: SMOTE Ablation Benchmark on Holdout Test Sets
- Table 4.5: Personal Loan Portfolio Model Benchmarks
- Table 4.6: Retail (SME) Loan Portfolio Model Benchmarks
- Table 4.7: Financial Loss Minimization: Default vs. Calibrated Cutoff
- Table 5.1: Enterprise Operational Implementation Roadmap

---

## LIST OF FIGURES
- Figure 2.1: Conceptual Framework Architecture of the End-to-End Decisioning Pipeline
- Figure 4.1: Receiver Operating Characteristic (ROC) and Precision-Recall (PR) Curves (Personal Portfolio)
- Figure 4.2: Expected Financial Loss Curves Across Probability Cutoff Thresholds (Personal Portfolio)
- Figure 4.3: Receiver Operating Characteristic (ROC) and Precision-Recall (PR) Curves (Retail SME Portfolio)
- Figure 4.4: Expected Financial Loss Curves Across Probability Cutoff Thresholds (Retail SME Portfolio)
- Figure 4.5: Global Feature Attribution (SHAP Beeswarm & Importance Summary) - Personal Loans
- Figure 4.6: Global Feature Attribution (SHAP Beeswarm & Importance Summary) - Retail SME Loans
- Figure 4.7: Local SHAP Waterfall Attribution for Adverse Action Loan Denial Case Study
- Figure 4.8: Interactive Streamlit Underwriting Dashboard User Interface Prototype

---

*(The comprehensive text of Chapters 1 through 5, as formulated in the respective chapter documents, is incorporated herein in full).*

---

## REFERENCES

- Adadi, A., & Berrada, M. (2018). Peeking inside the black-box: A survey on Explainable Artificial Intelligence (XAI). *IEEE Access*, 6, 52138-52160.
- Akerlof, G. A. (1970). The market for "lemons": Quality uncertainty and the market mechanism. *The Quarterly Journal of Economics*, 84(3), 488-500.
- Altman, E. I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *The Journal of Finance*, 23(4), 589-609.
- Barocas, S., & Selbst, A. D. (2016). Big data's disparate impact. *California Law Review*, 104, 671-732.
- Basel Committee on Banking Supervision [BCBS]. (2000). *Principles for the Management of Credit Risk*. Bank for International Settlements, Basel, Switzerland.
- Bello, A. (2023). Credit scoring models in Nigerian financial institutions: A comparative evaluation of statistical and algorithmic paradigms. *African Journal of Financial Computing*, 11(2), 114-131.
- Bodhale, P. (2025). Explainable artificial intelligence in consumer lending: Generating adverse action notifications with gradient boosting and Shapley values. *Journal of Financial Data Science*, 7(1), 45-63.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
- Bussmann, N., Giudici, P., Marinelli, D., & Papenbrock, J. (2021). Explainable machine learning in credit risk management. *Computational Economics*, 57(1), 203-216.
- Cao, R., Guan, Y., & Chen, J. (2018). Credit scoring using XGBoost: An empirical study on Chinese peer-to-peer lending platforms. *Proceedings of the 2018 International Conference on Financial Big Data*, 88-96.
- Central Bank of Nigeria [CBN]. (2019). *Consumer Protection Framework for Banks and Other Financial Institutions*. CBN Financial Policy and Regulation Department, Abuja, Nigeria.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321-357.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.
- Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. *Annals of Eugenics*, 7(2), 179-188.
- Hand, D. J., & Henley, W. E. (1997). Statistical classification methods in consumer credit scoring: A review. *Journal of the Royal Statistical Society: Series A (Statistics in Society)*, 160(3), 523-541.
- Joshi, A. (2026). State-of-the-art tabular learning in banking: An empirical benchmark of GBDT architectures on consumer credit portfolios. *Data Mining and Knowledge Discovery*, 38(1), 211-238.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30, 3146-3154.
- Kimani, M., Ochieng, E., & Ndung'u, J. (2024). Hybrid deep learning models for thin-file credit evaluation in emerging retail banking markets. *African Journal of Computing and Engineering*, 16(4), 189-204.
- Lessmann, S., Baesens, B., Seow, H. V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. *European Journal of Operational Research*, 247(1), 124-136.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30, 4765-4774.
- Lyn, C. C., Thomas, L. C., & Crook, J. N. (2002). *Readings in Credit Scoring*. Oxford University Press, Oxford.
- Merton, R. C. (1974). On the pricing of corporate debt: The risk structure of interest rates. *The Journal of Finance*, 29(2), 449-470.
- Mestiri, R. (2024). Non-parametric machine learning versus parametric models in retail credit risk prediction. *Journal of Risk and Financial Management*, 17(3), 112-129.
- Nguyen, T., & Ngo, H. (2025). Extreme gradient boosting for corporate and retail default prediction: A comprehensive empirical analysis. *Expert Systems with Applications*, 240, 122501.
- Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1(5), 206-215.
- Suhadolnik, N., Mramor, D., & Klemenc, K. (2023). Machine learning in credit risk modeling: The trade-off between performance and explainability. *Journal of Banking & Finance*, 148, 106752.
- Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). *Credit Scoring and its Applications*. SIAM Monographs on Mathematical Modeling and Computation, Philadelphia.
- Woo, S., & Sohn, S. Y. (2022). Deep learning credit scorecard incorporating psychometric and text-mined behavioral signals. *Decision Support Systems*, 154, 113702.
- Zhao, Z., & Aumeboonsuke, V. (2023). Resampling techniques for imbalanced credit scoring: A comparative empirical study with ensemble tree classifiers. *PLOS ONE*, 18(6), e0287114.
