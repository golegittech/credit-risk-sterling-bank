# CHAPTER TWO: LITERATURE REVIEW

## 2.1 Historical Perspectives of Credit Risk Analytics and Overview of Data Science in Credit Scoring
The methods used by financial institutions to assess credit risk have changed over time. In the early years of lending, decisions were largely based on personal relationships, manual assessment, and the judgment of loan officers. Lenders often used the 5 Cs of Credit, which are Character, Capacity, Capital, Collateral, and Conditions, to assess whether a borrower was likely to repay a loan. Although this approach allowed lenders to consider individual circumstances, decisions could be subjective and the process was difficult to apply consistently when handling large numbers of applications. These principles are still used in credit assessment today.

### The Rise of Statistical Credit Scoring (1950s to 1990s)
The development of statistical methods introduced a more systematic approach to credit risk assessment. Altman (1968), for example, developed the Z-Score model to assess the likelihood of corporate bankruptcy. During this period, standardized credit scoring also became increasingly important in consumer lending, with companies such as Fair, Isaac and Company contributing to the development of credit scoring systems.

For several decades, statistical models, particularly Logistic Regression (LR) and Linear Discriminant Analysis (LDA), played an important role in credit scoring (Hand & Henley, 1997). Logistic regression estimates the probability of an outcome, such as whether a borrower will default on a loan, based on a set of predictor variables. Its mathematical representation is:
$$\ln\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_k X_k$$

Financial institutions adopted statistical scoring models for several reasons. First, the models provided a consistent way to assess loan applications and could be used to develop credit scorecards. Second, their mathematical structure made it easier to interpret the relationship between borrower characteristics and the estimated risk of default. This was particularly useful when lenders needed to explain the factors influencing credit decisions.

Despite these advantages, traditional statistical models have limitations. Logistic regression assumes a linear relationship between the predictor variables and the log-odds of default. Although additional features and transformations can help capture more complex relationships, this often requires careful manual design. Statistical models can also be affected by multicollinearity, outliers, and incorrectly specified relationships between variables (Hand & Henley, 1997).

### The Machine Learning Era (2000s to Present)
The growth of digital banking has increased the amount and variety of data available for credit risk assessment. Financial institutions may now use information from transaction records, credit bureau histories, and other customer data to evaluate borrowers. These sources can contain complex patterns that may be difficult to capture using traditional statistical models alone (Mestiri, 2024).

Machine learning has therefore become an important approach to credit risk prediction. Unlike traditional models that may require explicit assumptions about relationships between variables, some machine learning algorithms can identify more complex patterns directly from the data.

Two important approaches are Random Forest and gradient boosting:
- **Random Forest (RF):** Random Forest combines the predictions of multiple decision trees. Each tree is trained using a sample of the training data and a subset of features. The combined predictions can reduce the tendency of an individual decision tree to overfit the training data (Breiman, 2001).
- **Gradient Boosted Decision Trees (GBDT):** Gradient boosting builds decision trees sequentially, with each new tree contributing to the correction of errors in the existing model. XGBoost and LightGBM are examples of gradient boosting frameworks used for predictive modelling. They can capture complex relationships between borrower characteristics and loan default outcomes (Chen & Guestrin, 2016; Ke et al., 2017).

These machine learning approaches provide useful alternatives to traditional statistical models. However, their performance depends on the quality of the data, the choice of model, and the evaluation method. Some models are also more difficult to interpret than traditional scorecards. For this reason, assessing predictive performance alongside model interpretability is important when applying machine learning to credit risk assessment.

---

## 2.2 Theoretical Framework
The theoretical foundation of this study draws on financial risk theories and computational machine learning concepts. Financial theories provide an understanding of the factors that influence borrowers' repayment behavior and the conditions that may lead to loan default. Statistical and machine learning theories, on the other hand, provide the methods required to analyze borrower information, predict default risk and explain the factors influencing model predictions.

The study is therefore guided by four major theoretical areas: financial risk theories, supervised machine learning, class imbalance and resampling techniques, and Explainable Artificial Intelligence.

### 2.2.1 Financial Risk Theories
#### Credit Risk Theory and Asymmetric Information
One of the major challenges in credit markets is the unequal distribution of information between lenders and borrowers. Borrowers generally have more information about their financial circumstances, repayment intentions and existing financial obligations than the lending institution. This situation is described as asymmetric information in economics.

Akerlof (1970) examined the effects of information asymmetry in markets, demonstrating how differences in the information available to market participants can influence economic decisions and market outcomes. In credit markets, asymmetric information creates two major problems:
- **Adverse Selection:** This occurs before a loan is granted. Because lenders cannot perfectly assess the financial condition and repayment intentions of every applicant, borrowers with higher default risks may be more likely to obtain credit under conditions that do not accurately reflect their actual risk.
- **Moral Hazard:** This occurs after a loan has been granted. A borrower may engage in financial activities that increase the likelihood of default because the lender cannot continuously observe every aspect of the borrower's behavior.

These problems are relevant to credit risk assessment because they demonstrate why lenders need reliable information about borrowers before and after extending credit. Machine learning models can help address some of the information limitations by examining historical repayment records, financial characteristics and other available borrower information to identify patterns associated with default risk.

#### Merton's Structural Default Theory
Merton (1974) developed a structural approach to modelling corporate default using principles from option pricing theory. The model considers a company's assets and debt obligations and explains default in terms of the relationship between the market value of the firm's assets and its outstanding debt. Under this framework, a company is considered to be in default at the maturity of its debt when the value of its assets is insufficient to meet its debt obligations.

The theory provides an important foundation for understanding the relationship between financial strength, debt obligations and default risk. Although the original model was developed for corporate finance, its underlying principle is relevant to credit risk assessment more broadly. For individual borrowers, related financial indicators such as income stability, existing debt obligations and available cash flow can provide useful information about repayment capacity. Similarly, for SME borrowers, business assets, cash flow, outstanding liabilities and operating performance influence the ability to meet loan obligations.

### 2.2.2 Statistical and Machine Learning Foundations
#### Supervised Binary Classification and Empirical Risk Minimization
Loan default prediction can be formulated as a supervised binary classification problem. In supervised learning, a model is trained using historical observations for which the actual outcomes are known:
- $Y = 1$ represents a default outcome.
- $Y = 0$ represents a non-default outcome.

The general objective of supervised model training can be represented as:
$$\min_{\theta} \frac{1}{N} \sum_{i=1}^{N} L\left(y_i, f(x_i; \theta)\right) + \lambda \Omega(\theta)$$

For binary classification, binary cross-entropy is commonly used as a loss function:
$$L(y, \hat{y}) = - \left[ y \log \hat{y} + (1 - y) \log (1 - \hat{y}) \right]$$

#### Decision Tree and Ensemble Boosting Theory
Ensemble learning combines multiple models to improve predictive performance. Random Forest uses bagging, training multiple trees on bootstrap samples. Gradient Boosted Decision Trees (GBDT) build trees sequentially:
$$F_t(x) = F_{t-1}(x) + \eta h_t(x)$$
XGBoost and LightGBM incorporate second-order Taylor approximations and regularisation to deliver scalable tabular modeling (Chen & Guestrin, 2016; Ke et al., 2017).

### 2.2.3 Class Imbalance and Resampling Theory
In credit risk datasets, default cases represent a smaller proportion of available records compared with non-default cases (e.g., 10% defaults vs 90% non-defaults). To address class imbalance, the Synthetic Minority Over-sampling Technique (SMOTE; Chawla et al., 2002) synthesizes new observations between an existing minority observation $x_i$ and its $k$-nearest neighbour $x_{zi}$:
$$x_{new} = x_i + \lambda (x_{zi} - x_i) \quad \text{where } \lambda \sim U(0, 1)$$

**Data Leakage Prevention:** SMOTE must be strictly applied only within the training set, leaving validation and test sets untouched to reflect real-world distributions.

### 2.2.4 Explainable AI (XAI) Theory: Cooperative Game Theory & SHAP
To resolve the "black-box" dilemma of tree ensembles, this study leverages SHAP (SHapley Additive exPlanations), rooted in cooperative game theory (Lundberg & Lee, 2017):
$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$
This guarantees that local feature attributions are additive, consistent, and locally accurate, providing a defensible audit trail for every credit decision.

---

## 2.3 Conceptual Framework
The conceptual framework maps the flow of data from ingestion through preprocessing, benchmarking, evaluation, and XAI deployment:
1. **Phase 1: Data Ingestion:** Personal and Retail loan portfolios from Sterling Bank Nigeria.
2. **Phase 2: Preprocessing & Resampling:** Missing value imputation, one-hot encoding, scaling, and train-only SMOTE.
3. **Phase 3: Model Training & Benchmarking:** Logistic Regression, Random Forest, LightGBM, and XGBoost.
4. **Phase 4: Model Evaluation & Comparative Analysis:** Statistical metrics (ROC-AUC, PR-AUC, F1, Recall) and Expected Financial Loss Matrix minimization.
5. **Phase 5: Explainable AI (XAI) & Deployment:** SHAP global summary/beeswarm, local waterfall plots, FastAPI REST endpoints, and Streamlit underwriter dashboard.

---

## 2.4 Review of Related Work
1. **Traditional Statistical Models in Banking:** Altman (1968), Hand & Henley (1997), Lyn et al. (2002), Lessmann et al. (2015).
2. **The Shift to Ensemble Machine Learning:** Breiman (2001), Chen & Guestrin (2016), Cao et al. (2018), Mestiri (2024), Joshi (2026).
3. **Deep Learning in Credit Scoring:** Kimani et al. (2024).
4. **Advanced Resampling for Class Imbalance:** Chawla et al. (2002), Zhao & Aumeboonsuke (2023).
5. **Explainable AI (XAI) and Regulatory Compliance:** Rudin (2019), Bussmann et al. (2021), Bodhale (2025).

---

## 2.5 Gaps in Existing Research
1. Separation of High-Performance Modeling and Explainable AI (XAI).
2. Over-Reliance on Standard Classification Accuracy over Financial Loss Minimization.
3. Limited Contextual Focus on Developing Banking Markets like Nigeria.
4. Lack of Operational Visualizations and Underwriter Tooling.

---

## 2.6 Summary of the Literature Review
This study bridges these identified research gaps by developing a dual-model ensemble pipeline integrated directly with SHAP attributions, calibrated via an asymmetric financial loss matrix, and operationalized through an interactive underwriting dashboard conforming to Central Bank of Nigeria (CBN) regulatory mandates.
