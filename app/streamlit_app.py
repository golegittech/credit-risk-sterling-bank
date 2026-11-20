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

from src import scoring
from src.preprocessing import load_portfolio


@st.cache_resource
def load_assets(portfolio_type: str):
    if os.path.exists(scoring.bundle_path(portfolio_type)):
        return scoring.load_bundle(portfolio_type)
    return None

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
    st.markdown('<p class="main-header">CREDIT RISK DECISION ENGINE (ACADEMIC PROTOTYPE)</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Dual-model (personal and SME) default prediction with SHAP explanations</p>', unsafe_allow_html=True)
st.warning("Academic prototype. All loan records are simulated; no real customer data is used. Not affiliated with or operated by Sterling Bank.")

# ----------------- Sidebar Navigation ----------------- #

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
        ["Personal Loan (Salary Earner)", "SME Loan"]
    )
    
    p_key = "personal" if "Personal" in portfolio_choice else "retail"
    bundle = load_assets(p_key)
    
    if bundle is None:
        st.warning(f"⚠️ {portfolio_choice} model is not yet trained. Please run `python -m src.experiment`.")
    else:
        opt_th = bundle["cutoff"]
        band = bundle["review_band"]
        
        st.markdown(f"**Model:** `{bundle['model_name'].replace('_', ' ')}` · **Cost-based cutoff (chosen on validation set):** `{opt_th:.2f}`")
        
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
            scored = scoring.score_application(p_key, input_payload)
            explanation = scoring.explain(p_key, input_payload)
            prob_def = scored["default_probability"]
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
                            {'range': [0, opt_th * band * 100], 'color': "#e8f5e9"},
                            {'range': [opt_th * band * 100, opt_th * 100], 'color': "#fffde7"},
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
                if prob_def < (opt_th * band):
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
                    st.markdown("#### 📜 Reasons for decline (sample adverse-action statement)")
                    st.info("Designed to support the CBN Consumer Protection Regulations (2019), under which a declined customer can ask for the reasons. The main factors that raised this applicant's estimated risk were:")
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
                y="label",
                orientation='h',
                color="direction",
                color_discrete_map={"INCREASES_RISK": "#d9534f", "DECREASES_RISK": "#5cb85c"},
                labels={"shap_value": f"SHAP contribution ({explanation['units']} of default)", "label": "Application feature"},
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
    across the test set, supporting transparency of the model's behaviour.
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
            
    st.markdown("### Feature ranking (mean |SHAP| on the test set)")
    rank_path = f"results/thesis/tables/{p_key}_shap_ranking.csv"
    if os.path.exists(rank_path):
        st.dataframe(pd.read_csv(rank_path))
        st.caption("SHAP shows association with predicted risk, not cause.")

# ----------------- MODE 4: Batch Portfolio Audit ----------------- #

elif app_mode == "Batch Portfolio Audit":
    st.subheader("Batch Portfolio Credit Scoring & Compliance Audit")
    portfolio_choice = st.selectbox("Portfolio Segment", ["Personal Loan", "Retail SME"], key="batch_port")
    p_key = "personal" if "Personal" in portfolio_choice else "retail"
    
    bundle = load_assets(p_key)
    
    if bundle is None:
        st.warning("Models not yet available.")
    else:
        opt_th = bundle["cutoff"]
        if True:
            df_full = scoring.score_frame(p_key, load_portfolio(p_key).head(50))
            
            st.write(f"Scored sample of 50 accounts against optimal threshold `{opt_th:.2f}`:")
            st.dataframe(df_full)
            
            csv_down = df_full.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Scored Audit Report (CSV)",
                csv_down,
                f"{p_key}_scored_audit_simulated.csv",
                "text/csv"
            )
