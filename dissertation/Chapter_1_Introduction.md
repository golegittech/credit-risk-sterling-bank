# CHAPTER ONE: INTRODUCTION

## 1.1 Background of the Study
Credit risk refers to the possibility that a borrower may fail to meet agreed financial obligations. It remains one of the major sources of risk for retail and commercial lending institutions worldwide (Basel Committee on Banking Supervision [BCBS], 2000).

Traditionally, financial institutions have assessed credit risk through manual underwriting, expert judgment, and statistical models. These methods include the 5 Cs of Credit, namely Character, Capacity, Capital, Collateral, and Conditions, as well as statistical techniques such as Logistic Regression, Discriminant Analysis, and Altman’s Z-Score (Altman, 1968; Thomas et al., 2002; Mestiri, 2024; Bello, 2023).

Although these traditional methods are relatively easy to understand and interpret, they have some limitations. For example, conventional linear scoring models may struggle to capture complex relationships between variables, patterns in borrower behaviour, and changes in economic conditions. These limitations can affect their ability to assess credit risk accurately, particularly when borrower characteristics and lending conditions become more complex (Mestiri, 2024).

### Relevance in Modern Computing and Data Science
Advances in computing, big data, and machine learning have changed how financial institutions assess credit risk. These technologies make it possible to process large amounts of data, identify patterns in borrower behaviour, and make lending decisions more efficiently.

- **Processing Large Volumes of Data:** Modern computing infrastructure allows financial institutions to process large amounts of transactional records, credit histories, and other borrower information. This helps them analyse a wider range of factors when assessing creditworthiness (Suhadolnik et al., 2023; Bello, 2023).
- **Identifying Complex Patterns:** Machine learning models such as XGBoost, LightGBM, and Random Forest can identify complex relationships between borrower characteristics and loan repayment outcomes. Unlike traditional linear models, these algorithms can capture non-linear relationships without requiring all such relationships to be specified manually (Chen & Guestrin, 2016; Suhadolnik et al., 2023).
- **Faster Loan Decisions:** Machine learning can automate parts of the loan assessment process and help financial institutions make decisions more quickly. This is particularly useful for digital banks and financial technology companies that process large numbers of loan applications (Bello, 2023).
- **Understanding Model Predictions:** Although machine learning models can make accurate predictions, some are difficult to interpret. This can be a concern in credit risk assessment, where financial institutions need to understand the factors influencing lending decisions. Methods such as SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model Agnostic Explanations) help explain individual predictions by showing how different features contribute to a model's output (Lundberg & Lee, 2017; Bussmann et al., 2021).

---

## 1.2 Problem Statement
Despite improvements in credit scoring models, financial institutions continue to face the challenge of predicting which borrowers are likely to default on their loans. Loan defaults can lead to financial losses and affect the stability of lending institutions. Two major challenges make credit risk prediction difficult.

The first challenge is **class imbalance**. In many credit datasets, borrowers who default make up a smaller proportion of the total number of borrowers. When a model is trained on such data, it may favour the majority class, which consists of borrowers who repay their loans. As a result, the model may fail to identify applicants who are likely to default, leading to high-risk loans being approved (Chawla et al., 2002).

The second challenge is **model interpretability**. Machine learning models such as XGBoost and neural networks can identify complex patterns in credit data, but their predictions may be difficult to explain. Financial institutions need to understand the factors behind lending decisions, particularly when an application is rejected. This creates a challenge when using complex models, as their predictive performance must be balanced with the need to explain their decisions. Methods such as SHAP can help explain model predictions and provide insight into the factors influencing credit risk assessments (Bussmann et al., 2021).

Changes in economic conditions and borrower behaviour make credit risk prediction even more challenging. These issues highlight the need for models that can identify potential loan defaults while providing reliable performance measures and explanations that are easy to understand. This study therefore focuses on addressing class imbalance and model interpretability in credit risk prediction.

---

## 1.3 Research Aim and Objectives

### Research Aim
The aim of this study is to develop and evaluate an interpretable machine learning framework for credit risk assessment and loan default prediction using credit data from Sterling Bank Nigeria. The study will focus on addressing class imbalance and using Explainable Artificial Intelligence (XAI) to make model predictions easier to understand.

### Research Objectives
The specific objectives of the study are:
1. To explore and preprocess Sterling Bank’s personal and retail loan data and apply synthetic resampling techniques to address class imbalance.
2. To develop an ensemble machine learning model, such as XGBoost, and compare its performance with traditional statistical models using suitable credit risk evaluation metrics.
3. To apply a SHAP-based Explainable Artificial Intelligence framework to explain model predictions and examine how these explanations can support transparent and auditable credit risk assessments in line with relevant Central Bank of Nigeria (CBN) guidelines.

---

## 1.4 Research Questions
To achieve the stated objectives, this study seeks to answer the following research questions:
- **RQ1:** How effectively can data preprocessing and synthetic resampling techniques, such as SMOTE, address class imbalance and help identify important risk factors in Sterling Bank’s personal and retail loan data?
- **RQ2:** How does the performance of an ensemble machine learning model, such as XGBoost, compare with traditional statistical models in predicting loan defaults when assessed using appropriate credit risk evaluation metrics?
- **RQ3:** How can a SHAP-based Explainable Artificial Intelligence framework help explain loan default predictions and support transparent and auditable credit risk assessments in the context of relevant CBN guidelines?

---

## 1.5 Significance of the Study

### Significance to Sterling Bank Nigeria and Commercial Lenders
The study may help Sterling Bank Nigeria understand how machine learning can be used to improve credit risk assessment. By comparing an ensemble model with traditional statistical models, the research will examine whether machine learning can identify borrowers who are more likely to default. This could support loan assessment and risk management by providing additional information for lending decisions. The findings may also help the bank assess whether these methods could improve the efficiency of its credit assessment process (Suhadolnik et al., 2023).

### Significance to Regulatory Oversight and Policy
The use of complex machine learning models in credit assessment can make it difficult to explain why a borrower has been classified as high risk. This can create challenges for transparency, auditing, and the review of lending decisions. This study will examine how SHAP (SHapley Additive exPlanations) can be used to explain the factors that influence model predictions. These explanations may help risk analysts, compliance officers, and auditors better understand the model's outputs. The study will also consider the relevance of these explanations to applicable Central Bank of Nigeria (CBN) guidelines and the need for transparent credit assessment (Lundberg & Lee, 2017; Bussmann et al., 2021).

### Significance to Applied Data Science and Financial Computing
This study will contribute to the application of machine learning in credit risk prediction by examining two key challenges: class imbalance and model interpretability. It will explore the use of synthetic resampling techniques, such as SMOTE, and evaluate an ensemble model, such as XGBoost, against traditional statistical approaches. The results may provide useful information for data scientists and researchers working on credit risk models, particularly when dealing with imbalanced financial datasets. The study may also serve as a reference for future research on the application of machine learning to credit assessment in Nigeria and other African banking markets (Chawla et al., 2002; Chen & Guestrin, 2016).

---

## 1.6 Scope of the Study

### Institutional Focus
The study focuses on Sterling Bank Nigeria, specifically its personal and retail lending operations. The analysis will use available historical loan data to examine factors associated with loan default and assess the performance of selected credit risk prediction models.

### Data Analysis and Model Development
The study covers the main stages of the data science process, including exploratory data analysis, data cleaning, feature engineering, and the treatment of class imbalance. Synthetic resampling techniques, such as SMOTE, will be considered alongside cost sensitive methods where appropriate. The study will compare Logistic Regression with machine learning models such as Random Forest, XGBoost, and LightGBM to evaluate their performance in predicting loan defaults (Chawla et al., 2002; Chen & Guestrin, 2016; Suhadolnik et al., 2023).

### Model Interpretability and Visualization
The study will use SHAP (SHapley Additive exPlanations) to examine the factors influencing model predictions at both the overall and individual borrower levels. It will also include an interactive dashboard prototype to present model performance, important risk factors, and prediction explanations. The explanations will be considered in relation to the need for transparent and auditable credit risk assessments and relevant Central Bank of Nigeria (CBN) guidelines (Lundberg & Lee, 2017; Bussmann et al., 2021).

---

## 1.7 Limitations of the Study
- **Historical Data and Economic Changes:** The study relies on historical loan data, which may not fully reflect sudden changes in Nigeria’s economic conditions, such as exchange rate fluctuations or increases in inflation. Unless relevant macroeconomic variables are available and included in the analysis, the models may not account for the effects of these changes on borrowers’ ability to repay their loans.
- **Limited Real-Time Testing:** The models will be evaluated using historical data, appropriate training and testing procedures, and cross-validation where applicable. Testing the models in Sterling Bank’s live production environment, integrating them with real-time data streams, and conducting A/B testing are outside the scope of this study.
- **Limitations of Synthetic Resampling:** Synthetic resampling techniques, such as SMOTE, can help address class imbalance by generating additional examples of the minority class. However, these synthetic examples are created from patterns in the available data and may not fully represent the behaviour of actual borrowers. Their use therefore requires careful evaluation to avoid overfitting or misleading performance estimates (Chawla et al., 2002).

---

## 1.8 Definition of Key Terms
- **Credit Risk:** The potential that a bank borrower or counterparty will fail to meet agreed financial obligations according to contract terms, resulting in direct financial loss to the lending institution (Basel Committee on Banking Supervision [BCBS], 2000).
- **Loan Default:** A binary target condition representing a failure by a borrower to make required debt repayments for a specified period (typically 90 days or more past due), triggering legal or collection procedures.
- **Class Imbalance:** A machine learning dataset condition where the target classes are severely disproportionate. In credit scoring, non-defaulting borrowers constitute the majority class (e.g., 85 to 95 percent), while defaulting borrowers constitute the minority class (e.g., 5 to 15 percent) (Chawla et al., 2002).
- **Ensemble Learning:** A machine learning strategy that combines multiple weak or base predictive models (such as individual decision trees) to construct a single, highly accurate meta-estimator. Examples include Random Forest, XGBoost, and LightGBM (Chen & Guestrin, 2016).
- **Gradient Boosting (XGBoost / LightGBM):** An ensemble technique that builds decision trees sequentially, where each new tree is optimized to minimize the residual errors produced by the previous combination of trees using gradient descent principles (Chen & Guestrin, 2016).
- **Synthetic Minority Over-sampling Technique (SMOTE):** An algorithmic resampling technique that mitigates class imbalance by synthesizing new, artificial minority class samples along line segments connecting existing k-nearest neighbors in the feature space, rather than simply duplicating existing records (Chawla et al., 2002).
- **Explainable AI (XAI):** A suite of computational frameworks and algorithms designed to make the internal decision logic of complex, high-performing "black-box" machine learning models human-interpretable, auditable, and regulatory-compliant (Bussmann et al., 2021).
- **SHAP (SHapley Additive exPlanations):** A game-theoretic framework for explaining machine learning model predictions that computes Shapley values to assign precise mathematical attribution scores to each input feature for individual and global decisions (Lundberg & Lee, 2017).
- **Non-Performing Loans (NPLs):** Bank loans that are subject to default or in close risk of default due to non-payment of principal or interest over a specified duration, serving as a primary measure of commercial bank asset quality.
