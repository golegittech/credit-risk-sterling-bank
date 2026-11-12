"""
Interactive Credit Risk Underwriting & Portfolio Analytics Dashboard.
Built with Streamlit for Sterling Bank Nigeria Credit Risk Officers, Underwriters,
and Central Bank of Nigeria (CBN) Regulatory Auditors.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import json
import joblib

# Set Page Config
st.set_page_config(
    page_title="Sterling Bank Credit Risk Decision Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #8B0000; /* Sterling Crimson/Wine */
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4A4A4A;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #8B0000;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .status-badge-approved {
        background-color: #d4edda;
        color: #155724;
        font-weight: bold;
        padding: 6px 12px;
        border-radius: 4px;
        display: inline-block;
    }
    .status-badge-review {
        background-color: #fff3cd;
        color: #856404;
        font-weight: bold;
        padding: 6px 12px;
        border-radius: 4px;
        display: inline-block;
    }
    .status-badge-rejected {
        background-color: #f8d7da;
        color: #721c24;
        font-weight: bold;
        padding: 6px 12px;
        border-radius: 4px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Helper Functions ----------------- #

@st.cache_resource
def load_assets(portfolio_type: str):
    m_path = f"models/{portfolio_type}_xgboost.joblib"
    p_path = f"models/{portfolio_type}_preprocessor.joblib"
    if os.path.exists(m_path) and os.path.exists(p_path):
        model = joblib.load(m_path)
        prep = joblib.load(p_path)
        return model, prep
    return None, None

@st.cache_data
def load_metrics(portfolio_type: str):
    met_path = f"results/{portfolio_type}_metrics.json"
    if os.path.exists(met_path):
        with open(met_path, 'r') as f:
            return json.load(f)
    return None

# ----------------- Header Banner ----------------- #

col_logo, col_title = st.columns([1, 6])
with col_title:
    st.markdown('<p class="main-header">STERLING BANK NIGERIA — CREDIT RISK DECISION ENGINE</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Dual-Model Machine Learning Underwriting System & Explainable AI (XAI) Compliance Dashboard</p>', unsafe_allow_html=True)

# ----------------- Sidebar Navigation ----------------- #

st.sidebar.image("https://via.placeholder.com/260x80/8B0000/FFFFFF?text=STERLING+BANK")
st.sidebar.title("Operational Controls")

app_mode = st.sidebar.radio(
    "Navigation Mode",
    [
        "Loan Application Underwriting",
        "Portfolio Analytics & Benchmarks",
        "Global XAI & Risk Drivers",
        "Batch Portfolio Audit"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project Information**")
st.sidebar.info(
    """
    **Researcher:** Mbaegbu Chinedum Grace  
    **Matric No:** 2025/A/DSC/0058  
    **Supervisor:** Dr. A. S. Makinde  
    **Institution:** Miva Open University, Abuja  
    **Degree:** B.Sc. Data Science  
    """
)

# ----------------- MODE 1: Underwriting Decisioning ----------------- #

if app_mode == "Loan Application Underwriting":
    st.subheader("Automated Loan Application Underwriting")
    
    portfolio_choice = st.selectbox(
        "Select Lending Portfolio Segment",
        ["Personal Loan (Salary Earner)", "Retail SME Loan"]
    )
    
    p_key = "personal" if "Personal" in portfolio_choice else "retail"
    model, prep = load_assets(p_key)
    metrics_data = load_metrics(p_key)
    
    if model is None or prep is None:
        st.warning(f"⚠️ {portfolio_choice} models are not yet trained. Please run `python -m src.train_models` to generate models.")
    else:
        opt_th = metrics_data.get("XGBoost", {}).get("financial_optimization", {}).get("optimal_threshold", 0.35)
        
        st.markdown(f"**Applied Financial Loss-Minimizing Cutoff Threshold:** `{opt_th:.2f}`")
        
        with st.form("underwrite_form"):
            if p_key == "personal":
                c1, c2, c3 = st.columns(3)
                with c1:
                    age = st.slider("Applicant Age", 21, 65, 34)
                    housing = st.selectbox("Housing Status", ["RENT", "MORTGAGE", "OWN"])
                    emp_len = st.number_input("Employment Length (Years)", 0.0, 40.0, 5.5, step=0.5)
                with c2:
                    annual_inc = st.number_input("Gross Annual Salary (NGN)", 1_000_000, 60_000_000, 5_500_000, step=250_000)
                    loan_amt = st.number_input("Requested Facility Amount (NGN)", 100_000, 20_000_000, 1_800_000, step=100_000)
                    dti = st.slider("Debt-to-Income (DTI) Ratio (%)", 5.0, 85.0, 28.5, step=0.5)
                with c3:
                    credit_score = st.slider("Credit Bureau Score", 300, 850, 680)
                    ext_loan = st.selectbox("Concurrent External Facilities", ["None", "Performing", "Defaulting"])
                    
                submitted = st.form_submit_button("Evaluate Credit Risk & Explain Prediction")
                
                input_payload = {
                    'applicant_age': age,
                    'housing_status': housing,
                    'employment_length': emp_len,
                    'annual_income': annual_inc,
                    'debt_to_income_ratio': dti,
                    'credit_bureau_score': credit_score,
                    'existing_external_loan_status': ext_loan,
                    'loan_amount': loan_amt
                }

            else: # Retail SME
                c1, c2, c3 = st.columns(3)
                with c1:
                    biz_age = st.number_input("Business Age (Years incorporated)", 0.5, 30.0, 6.0, step=0.5)
                    nature_biz = st.selectbox("Nature of Business Sector", [
                        'Trade & Commerce', 'Agriculture & Agro-allied', 'Manufacturing',
                        'Services', 'Information Technology', 'Construction'
                    ])
                    region = st.selectbox("Operational Geopolitical Zone", [
                        'South-West', 'North-Central', 'South-South', 'North-West', 'South-East', 'North-East'
                    ])
                with c2:
                    facility = st.selectbox("Facility Type", ['Term Loan', 'Overdraft', 'Asset Finance', 'Invoice Discounting'])
                    loan_amt = st.number_input("Requested Facility Amount (NGN)", 500_000, 50_000_000, 8_500_000, step=500_000)
                    rate = st.slider("Facility Annual Interest Rate (%)", 18.0, 35.0, 24.5, step=0.5)
                with c3:
                    ext_loan = st.selectbox("Concurrent External Facilities", ["None", "Performing", "Defaulting"])
                    
                submitted = st.form_submit_button("Evaluate Credit Risk & Explain Prediction")
                
                input_payload = {
                    'business_age_years': biz_age,
                    'nature_of_business': nature_biz,
                    'business_region': region,
                    'facility_type': facility,
                    'loan_amount': loan_amt,
                    'interest_rate': rate,
                    'existing_external_loan_status': ext_loan
                }

        if submitted:
            from src.explainability import CreditRiskExplainer
            explainer = CreditRiskExplainer(p_key)
            explanation = explainer.explain_instance(input_payload)
            prob_def = explanation["predicted_default_probability"]
            repayment_prob = (1.0 - prob_def) * 100
            
            st.markdown("---")
            st.subheader("Credit Decisioning Output")
            
            d_col1, d_col2 = st.columns([2, 3])
            
            with d_col1:
                # Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(prob_def * 100, 1),
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Default Probability (%)", 'font': {'size': 20, 'color': '#333'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "#8B0000" if prob_def >= opt_th else "#28a745"},
                        'steps': [
                            {'range': [0, opt_th * 70], 'color': "#e8f5e9"},
                            {'range': [opt_th * 70, opt_th * 100], 'color': "#fffde7"},
                            {'range': [opt_th * 100, 100], 'color': "#ffebee"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': opt_th * 100
                        }
                    }
                ))
                fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge)
                
            with d_col2:
                st.markdown("### Underwriter Assessment")
                if prob_def < (opt_th * 0.70):
                    st.markdown('<p class="status-badge-approved">STATUS: AUTOMATED APPROVAL RECOMMENDED</p>', unsafe_allow_html=True)
                    st.write("Borrower risk profile is comfortably within the bank's risk appetite.")
                elif prob_def < opt_th:
                    st.markdown('<p class="status-badge-review">STATUS: SECONDARY MANUAL REVIEW REQUIRED</p>', unsafe_allow_html=True)
                    st.write("Borrower exhibits borderline metrics requiring review by a Senior Credit Underwriter.")
                else:
                    st.markdown('<p class="status-badge-rejected">STATUS: APPLICATION DECLINED (HIGH DEFAULT RISK)</p>', unsafe_allow_html=True)
                    st.write("Predicted default likelihood breaches the loss-minimizing hurdle rate.")
                    
                st.write(f"**Estimated Repayment Likelihood:** `{repayment_prob:.1f}%`")
                st.write(f"**Applied Loss-Minimizing Threshold:** `{opt_th:.3f}`")
                
                # Adverse Action Notice per CBN Guidelines
                if prob_def >= opt_th:
                    st.markdown("#### 📜 Central Bank of Nigeria (CBN) Adverse Action Summary")
                    st.info("In compliance with the CBN Consumer Protection Framework, the primary factors determining this adverse decision are:")
                    for idx, reason in enumerate(explanation["adverse_action_reasons"][:3], 1):
                        st.markdown(f"**{idx}.** {reason}")

            # Local SHAP Attribution Bar Chart
            st.markdown("### Explainable AI (XAI) — Local Feature Attribution")
            st.write("Shapley values illustrate how individual borrower attributes push risk upward (red) or downward (green).")
            
            top_attrs = pd.DataFrame(explanation["top_attributions"])
            top_attrs["color"] = np.where(top_attrs["shap_value"] > 0, "#d9534f", "#5cb85c")
            
            fig_bar = px.bar(
                top_attrs[::-1],
                x="shap_value",
                y="feature",
                orientation='h',
                color="direction",
                color_discrete_map={"INCREASES_RISK": "#d9534f", "DECREASES_RISK": "#5cb85c"},
                labels={"shap_value": "SHAP Contribution to Default Risk", "feature": "Application Feature"},
                title=f"Feature Impact Breakdown for Applicant ({portfolio_choice})"
            )
            fig_bar.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_bar)

# ----------------- MODE 2: Portfolio Analytics & Benchmarks ----------------- #

elif app_mode == "Portfolio Analytics & Benchmarks":
    st.subheader("Model Performance Benchmarking & Financial Loss Minimization")
    
    portfolio_choice = st.selectbox(
        "Select Lending Portfolio Segment",
        ["Personal Loan (Salary Earner)", "Retail SME Loan"],
        key="bench_port"
    )
    p_key = "personal" if "Personal" in portfolio_choice else "retail"
    metrics_data = load_metrics(p_key)
    
    if metrics_data is None:
        st.warning("Metrics not yet generated. Please execute model training script.")
    else:
        # Build comparison dataframe
        rows = []
        for m_name, vals in metrics_data.items():
            fin = vals.get("financial_optimization", {})
            rows.append({
                "Model Architecture": m_name.replace("_", " "),
                "ROC-AUC": f"{vals['roc_auc']:.4f}",
                "PR-AUC": f"{vals['pr_auc']:.4f}",
                "F1-Score": f"{vals['f1_score']:.4f}",
                "Recall (Sensitivity)": f"{vals['recall']:.4f}",
                "Precision": f"{vals['precision']:.4f}",
                "Accuracy": f"{vals['accuracy']:.4f}",
                "Optimal Cutoff": f"{fin.get('optimal_threshold', 0.5):.2f}",
                "Expected Loss (NGN M)": f"{fin.get('min_financial_loss', 0)/1e6:.2f}",
                "Monetary Savings (NGN M)": f"{fin.get('monetary_savings_ngn', 0)/1e6:.2f}"
            })
            
        df_bench = pd.DataFrame(rows)
        st.dataframe(df_bench)
        
        c_fig1, c_fig2 = st.columns(2)
        roc_path = f"results/{p_key}_roc_pr_curves.png"
        cost_path = f"results/{p_key}_cost_curve.png"
        
        with c_fig1:
            if os.path.exists(roc_path):
                st.image(roc_path, caption=f"ROC and PR Evaluation Curves ({portfolio_choice})")
            else:
                st.info("ROC/PR curve image pending generation.")
                
        with c_fig2:
            if os.path.exists(cost_path):
                st.image(cost_path, caption=f"Expected Financial Loss Curves ({portfolio_choice})")
            else:
                st.info("Cost curve image pending generation.")

# ----------------- MODE 3: Global XAI & Risk Drivers ----------------- #

elif app_mode == "Global XAI & Risk Drivers":
    st.subheader("Global Explainable AI (XAI) & Factor Attribution")
    st.markdown("""
    Global SHAP explanations quantify the systemic importance of each financial and demographic feature
    across the entire lending portfolio, satisfying regulatory transparency standards.
    """)
    
    portfolio_choice = st.selectbox(
        "Select Portfolio Segment",
        ["Personal Loan (Salary Earner)", "Retail SME Loan"],
        key="xai_port"
    )
    p_key = "personal" if "Personal" in portfolio_choice else "retail"
    
    col1, col2 = st.columns(2)
    beeswarm_path = f"results/{p_key}_shap_beeswarm.png"
    summary_path = f"results/{p_key}_shap_summary.png"
    
    with col1:
        if os.path.exists(beeswarm_path):
            st.image(beeswarm_path, caption=f"SHAP Beeswarm Plot — {portfolio_choice}")
        else:
            st.info("Global beeswarm plot pending.")
            
    with col2:
        if os.path.exists(summary_path):
            st.image(summary_path, caption=f"Mean |SHAP| Importance Ranking — {portfolio_choice}")
        else:
            st.info("Global summary plot pending.")
            
    st.markdown("### Risk Interpretation & Regulatory Compliance")
    if p_key == "personal":
        st.write("""
        - **Debt-to-Income (DTI) Ratio** exhibits the highest systemic SHAP attribution. Applicants exceeding 45% DTI demonstrate exponential default probability increases.
        - **Credit Bureau Score** displays a strong inverse relationship: higher scores substantially lower default odds.
        - **External Delinquencies** immediately shift the base log-odds into the high-risk bracket.
        """)
    else:
        st.write("""
        - **Existing External Delinquency** is the single greatest risk driver in the Retail SME portfolio.
        - **Facility Interest Rate** heavily increases default risk when exceeding 26%, reflecting repayment burdens in high inflation environments.
        - **Business Age (Years incorporated)** provides strong protection against default, verifying commercial stability over time.
        """)

# ----------------- MODE 4: Batch Portfolio Audit ----------------- #

elif app_mode == "Batch Portfolio Audit":
    st.subheader("Batch Portfolio Credit Scoring & Compliance Audit")
    portfolio_choice = st.selectbox("Portfolio Segment", ["Personal Loan", "Retail SME"], key="batch_port")
    p_key = "personal" if "Personal" in portfolio_choice else "retail"
    
    model, prep = load_assets(p_key)
    metrics_data = load_metrics(p_key)
    
    if model is None or prep is None:
        st.warning("Models not yet available.")
    else:
        opt_th = metrics_data.get("XGBoost", {}).get("financial_optimization", {}).get("optimal_threshold", 0.35)
        csv_file = f"data/{p_key}_loans.csv" if p_key == "personal" else "data/retail_sme_loans.csv"
        
        if os.path.exists(csv_file):
            df_full = pd.read_csv(csv_file).head(50)
            X_batch, _ = prep.transform_test(df_full)
            probs = model.predict_proba(X_batch)[:, 1]
            
            df_full["Predicted_Default_Prob"] = np.round(probs, 4)
            df_full["Underwriter_Decision"] = np.where(probs < (opt_th * 0.70), "APPROVE", np.where(probs < opt_th, "MANUAL REVIEW", "REJECT"))
            
            st.write(f"Scored sample of 50 accounts against optimal threshold `{opt_th:.2f}`:")
            st.dataframe(df_full)
            
            csv_down = df_full.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Scored Audit Report (CSV)",
                csv_down,
                f"sterling_bank_{p_key}_scored_audit.csv",
                "text/csv"
            )
