"""
FastAPI Backend & Web Interface for Sterling Bank Nigeria Credit Risk Assessment.
Provides:
- Web Underwriting Portal at GET /
- RESTful endpoints for Loan default prediction & SHAP Explainable AI
- Model performance metrics & financial loss optimization lookup
"""

import os
import json
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(
    title="Sterling Bank Credit Risk & Loan Default Prediction API",
    description="Dual-Model Machine Learning API (Personal Salary Earner & Retail SME) with SHAP Explainability",
    version="1.0.0"
)

# Enable CORS for frontend and deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = "models"
RESULTS_DIR = "results"

# ----------------- Pydantic Request Models ----------------- #

class PersonalLoanRequest(BaseModel):
    applicant_age: int = Field(..., ge=18, le=75, description="Age in years")
    housing_status: str = Field(..., description="OWN, RENT, or MORTGAGE")
    employment_length: float = Field(..., ge=0, le=45, description="Years in formal employment")
    annual_income: float = Field(..., ge=100_000, description="Gross annual income in NGN")
    debt_to_income_ratio: float = Field(..., ge=0, le=100, description="DTI percentage")
    credit_bureau_score: int = Field(..., ge=300, le=850, description="Credit Bureau Score")
    existing_external_loan_status: str = Field(..., description="None, Performing, or Defaulting")
    loan_amount: float = Field(..., ge=50_000, description="Principal requested in NGN")

class RetailLoanRequest(BaseModel):
    business_age_years: float = Field(..., ge=0, le=60, description="Years since CAC incorporation")
    nature_of_business: str = Field(..., description="Trade, Agriculture, Manufacturing, Tech, etc.")
    business_region: str = Field(..., description="Geopolitical zone / region")
    facility_type: str = Field(..., description="Term Loan, Overdraft, Asset Finance, etc.")
    loan_amount: float = Field(..., ge=100_000, description="Principal requested in NGN")
    interest_rate: float = Field(..., ge=5, le=50, description="Lending interest rate (%)")
    existing_external_loan_status: str = Field(..., description="None, Performing, or Defaulting")

# ----------------- Helper Functions ----------------- #

def load_model_and_prep(portfolio_type: str):
    m_path = os.path.join(MODELS_DIR, f"{portfolio_type}_xgboost.joblib")
    p_path = os.path.join(MODELS_DIR, f"{portfolio_type}_preprocessor.joblib")
    if not os.path.exists(m_path) or not os.path.exists(p_path):
        raise HTTPException(status_code=503, detail=f"Models for {portfolio_type} not trained or missing.")
    model = joblib.load(m_path)
    prep = joblib.load(p_path)
    return model, prep

def get_optimal_threshold(portfolio_type: str) -> float:
    metrics_path = os.path.join(RESULTS_DIR, f"{portfolio_type}_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            data = json.load(f)
            return float(data.get("XGBoost", {}).get("financial_optimization", {}).get("optimal_threshold", 0.35))
    return 0.35

# ----------------- Endpoints ----------------- #

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sterling Bank — Credit Risk Decision Engine</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <style>
            :root { --primary-crimson: #8B0000; --dark-crimson: #5c0000; --gold: #c59b27; }
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; color: #333; }
            .header-banner { background: linear-gradient(135deg, var(--primary-crimson) 0%, #2b0000 100%); color: white; padding: 30px 20px; border-bottom: 4px solid var(--gold); }
            .badge-gold { background-color: var(--gold); color: white; }
            .card { border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.06); border: none; margin-bottom: 25px; }
            .card-header { background-color: white; border-bottom: 2px solid #f0f0f0; font-weight: 700; padding: 18px 25px; }
            .nav-pills .nav-link.active { background-color: var(--primary-crimson); font-weight: 600; }
            .nav-pills .nav-link { color: #555; }
            .btn-sterling { background-color: var(--primary-crimson); color: white; font-weight: 600; padding: 12px 25px; border-radius: 8px; border: none; }
            .btn-sterling:hover { background-color: var(--dark-crimson); color: white; }
            .score-box { border-radius: 12px; padding: 25px; text-align: center; }
            .score-approved { background-color: #e8f5e9; border: 2px solid #28a745; color: #155724; }
            .score-review { background-color: #fffde7; border: 2px solid #ffc107; color: #856404; }
            .score-rejected { background-color: #ffebee; border: 2px solid #dc3545; color: #721c24; }
            .adverse-notice { background-color: #f8d7da; border-left: 5px solid #dc3545; padding: 15px; border-radius: 4px; margin-top: 15px; }
        </style>
    </head>
    <body>
        <div class="header-banner">
            <div class="container">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <h2 class="mb-1"><i class="fa-solid fa-building-columns me-2"></i>STERLING BANK NIGERIA</h2>
                        <p class="mb-0 text-white-50">Interpretable Credit Risk Analytics & Loan Default Prediction Engine</p>
                    </div>
                    <div>
                        <a href="/docs" target="_blank" class="btn btn-outline-light btn-sm"><i class="fa-solid fa-code me-1"></i> Swagger API Docs</a>
                    </div>
                </div>
            </div>
        </div>

        <div class="container my-4">
            <div class="row">
                <div class="col-lg-3">
                    <div class="card p-3">
                        <h6 class="text-uppercase text-muted fw-bold mb-3" style="font-size: 0.8rem;">Project Information</h6>
                        <p class="small mb-2"><strong>Researcher:</strong> Mbaegbu Chinedum Grace</p>
                        <p class="small mb-2"><strong>Matric No:</strong> 2025/A/DSC/0058</p>
                        <p class="small mb-2"><strong>Supervisor:</strong> Dr. A. S. Makinde</p>
                        <p class="small mb-2"><strong>Degree:</strong> B.Sc. Data Science</p>
                        <p class="small mb-3"><strong>Institution:</strong> Miva Open University, Abuja</p>
                        <hr>
                        <h6 class="text-uppercase text-muted fw-bold mb-2" style="font-size: 0.8rem;">Regulatory Standard</h6>
                        <span class="badge bg-dark mb-2">CBN Consumer Protection</span>
                        <span class="badge badge-gold">SHAP Explainable AI</span>
                        <span class="badge bg-danger">Loss Matrix Optimization</span>
                    </div>
                </div>

                <div class="col-lg-9">
                    <div class="card">
                        <div class="card-header">
                            <ul class="nav nav-pills card-header-pills" id="loanTabs" role="tablist">
                                <li class="nav-item">
                                    <button class="nav-link active" id="personal-tab" data-bs-toggle="pill" data-bs-target="#personal-panel" type="button">Personal Loan (Salary Earner)</button>
                                </li>
                                <li class="nav-item">
                                    <button class="nav-link" id="retail-tab" data-bs-toggle="pill" data-bs-target="#retail-panel" type="button">Retail SME Loan</button>
                                </li>
                            </ul>
                        </div>
                        <div class="card-body p-4">
                            <div class="tab-content" id="loanTabsContent">
                                <!-- PERSONAL LOAN PANEL -->
                                <div class="tab-pane fade show active" id="personal-panel" role="tabpanel">
                                    <form id="personalForm">
                                        <div class="row g-3">
                                            <div class="col-md-4">
                                                <label class="form-label fw-bold">Applicant Age</label>
                                                <input type="number" class="form-control" id="p_age" value="34" min="21" max="65" required>
                                            </div>
                                            <div class="col-md-4">
                                                <label class="form-label fw-bold">Housing Status</label>
                                                <select class="form-select" id="p_housing">
                                                    <option value="RENT">RENT</option>
                                                    <option value="MORTGAGE">MORTGAGE</option>
                                                    <option value="OWN">OWN</option>
                                                </select>
                                            </div>
                                            <div class="col-md-4">
                                                <label class="form-label fw-bold">Employment Length (Years)</label>
                                                <input type="number" step="0.5" class="form-control" id="p_emp" value="5.5" required>
                                            </div>
                                            <div class="col-md-6">
                                                <label class="form-label fw-bold">Gross Annual Salary (NGN)</label>
                                                <input type="number" class="form-control" id="p_income" value="5500000" step="100000" required>
                                            </div>
                                            <div class="col-md-6">
                                                <label class="form-label fw-bold">Requested Loan Amount (NGN)</label>
                                                <input type="number" class="form-control" id="p_loan" value="1800000" step="50000" required>
                                            </div>
                                            <div class="col-md-4">
                                                <label class="form-label fw-bold">Debt-to-Income (DTI) %</label>
                                                <input type="number" step="0.1" class="form-control" id="p_dti" value="28.5" required>
                                            </div>
                                            <div class="col-md-4">
                                                <label class="form-label fw-bold">Credit Bureau Score</label>
                                                <input type="number" class="form-control" id="p_score" value="680" min="300" max="850" required>
                                            </div>
                                            <div class="col-md-4">
                                                <label class="form-label fw-bold">External Loan Status</label>
                                                <select class="form-select" id="p_ext">
                                                    <option value="None">None</option>
                                                    <option value="Performing" selected>Performing</option>
                                                    <option value="Defaulting">Defaulting</option>
                                                </select>
                                            </div>
                                            <div class="col-12 mt-4">
                                                <button type="button" class="btn btn-sterling w-100" onclick="submitPersonal()">
                                                    <i class="fa-solid fa-calculator me-2"></i>Evaluate Credit Risk & Generate Adverse Notice
                                                </button>
                                            </div>
                                        </div>
                                    </form>
                                </div>

                                <!-- RETAIL SME PANEL -->
                                <div class="tab-pane fade" id="retail-panel" role="tabpanel">
                                    <form id="retailForm">
                                        <div class="row g-3">
                                            <div class="col-md-4">
                                                <label class="form-label fw-bold">Business Age (CAC Years)</label>
                                                <input type="number" step="0.5" class="form-control" id="r_age" value="6.0" required>
                                            </div>
                                            <div class="col-md-4">
                                                <label class="form-label fw-bold">Nature of Business</label>
                                                <select class="form-select" id="r_nature">
                                                    <option value="Trade & Commerce">Trade & Commerce</option>
                                                    <option value="Agriculture & Agro-allied">Agriculture & Agro-allied</option>
                                                    <option value="Manufacturing">Manufacturing</option>
                                                    <option value="Services">Services</option>
                                                    <option value="Information Technology">Information Technology</option>
                                                    <option value="Construction">Construction</option>
                                                </select>
                                            </div>
                                            <div class="col-md-4">
                                                <label class="form-label fw-bold">Operational Region</label>
                                                <select class="form-select" id="r_region">
                                                    <option value="South-West">South-West</option>
                                                    <option value="North-Central">North-Central</option>
                                                    <option value="South-South">South-South</option>
                                                    <option value="North-West">North-West</option>
                                                    <option value="South-East">South-East</option>
                                                    <option value="North-East">North-East</option>
                                                </select>
                                            </div>
                                            <div class="col-md-4">
                                                <label class="form-label fw-bold">Facility Type</label>
                                                <select class="form-select" id="r_facility">
                                                    <option value="Term Loan">Term Loan</option>
                                                    <option value="Overdraft">Overdraft</option>
                                                    <option value="Asset Finance">Asset Finance</option>
                                                    <option value="Invoice Discounting">Invoice Discounting</option>
                                                </select>
                                            </div>
                                            <div class="col-md-4">
                                                <label class="form-label fw-bold">Requested Facility (NGN)</label>
                                                <input type="number" class="form-control" id="r_loan" value="8500000" step="250000" required>
                                            </div>
                                            <div class="col-md-4">
                                                <label class="form-label fw-bold">Interest Rate (%)</label>
                                                <input type="number" step="0.1" class="form-control" id="r_rate" value="24.5" required>
                                            </div>
                                            <div class="col-md-12">
                                                <label class="form-label fw-bold">Concurrent External Loan Status</label>
                                                <select class="form-select" id="r_ext">
                                                    <option value="None">None</option>
                                                    <option value="Performing" selected>Performing</option>
                                                    <option value="Defaulting">Defaulting</option>
                                                </select>
                                            </div>
                                            <div class="col-12 mt-4">
                                                <button type="button" class="btn btn-sterling w-100" onclick="submitRetail()">
                                                    <i class="fa-solid fa-briefcase me-2"></i>Evaluate Commercial SME Risk
                                                </button>
                                            </div>
                                        </div>
                                    </form>
                                </div>
                            </div>

                            <!-- DECISION RESULT DISPLAY -->
                            <div id="resultContainer" class="mt-4" style="display:none;">
                                <hr>
                                <h5 class="fw-bold mb-3"><i class="fa-solid fa-clipboard-check me-2"></i>Automated Underwriting Decision</h5>
                                <div id="decisionCard" class="score-box"></div>
                                <div id="adverseBox" style="display:none;"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            async function submitPersonal() {
                const payload = {
                    applicant_age: parseInt(document.getElementById('p_age').value),
                    housing_status: document.getElementById('p_housing').value,
                    employment_length: parseFloat(document.getElementById('p_emp').value),
                    annual_income: parseFloat(document.getElementById('p_income').value),
                    debt_to_income_ratio: parseFloat(document.getElementById('p_dti').value),
                    credit_bureau_score: parseInt(document.getElementById('p_score').value),
                    existing_external_loan_status: document.getElementById('p_ext').value,
                    loan_amount: parseFloat(document.getElementById('p_loan').value)
                };
                evaluateLoan('/explain/personal', payload);
            }

            async function submitRetail() {
                const payload = {
                    business_age_years: parseFloat(document.getElementById('r_age').value),
                    nature_of_business: document.getElementById('r_nature').value,
                    business_region: document.getElementById('r_region').value,
                    facility_type: document.getElementById('r_facility').value,
                    loan_amount: parseFloat(document.getElementById('r_loan').value),
                    interest_rate: parseFloat(document.getElementById('r_rate').value),
                    existing_external_loan_status: document.getElementById('r_ext').value
                };
                evaluateLoan('/explain/retail', payload);
            }

            async function evaluateLoan(endpoint, payload) {
                const resDiv = document.getElementById('resultContainer');
                const cardDiv = document.getElementById('decisionCard');
                const adverseDiv = document.getElementById('adverseBox');
                resDiv.style.display = 'block';
                cardDiv.innerHTML = '<div class="spinner-border text-danger" role="status"></div><p class="mt-2">Computing XGBoost Risk Scores and SHAP Attributions...</p>';
                adverseDiv.style.display = 'none';

                try {
                    const res = await fetch(endpoint, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    const data = await res.json();
                    
                    let cls = 'score-approved';
                    let icon = 'fa-check-circle';
                    if (data.decision === 'MANUAL REVIEW') { cls = 'score-review'; icon = 'fa-triangle-exclamation'; }
                    else if (data.decision === 'REJECT') { cls = 'score-rejected'; icon = 'fa-ban'; }

                    cardDiv.className = 'score-box ' + cls;
                    cardDiv.innerHTML = `
                        <h3><i class="fa-solid ${icon} me-2"></i>DECISION: ${data.decision}</h3>
                        <p class="mb-1"><strong>Predicted Probability of Default:</strong> ${(data.default_probability * 100).toFixed(1)}%</p>
                        <p class="mb-1"><strong>Repayment Likelihood:</strong> ${data.repayment_likelihood}%</p>
                        <p class="mb-0"><strong>Applied Financial Optimal Cutoff:</strong> ${data.optimal_threshold_applied}</p>
                    `;

                    if (data.decision === 'REJECT' && data.xai_explanation && data.xai_explanation.adverse_action_reasons) {
                        let reasonsHtml = '<ul>';
                        data.xai_explanation.adverse_action_reasons.forEach(r => { reasonsHtml += `<li>${r}</li>`; });
                        reasonsHtml += '</ul>';

                        adverseDiv.className = 'adverse-notice';
                        adverseDiv.innerHTML = `
                            <h6 class="fw-bold mb-2"><i class="fa-solid fa-scale-balanced me-2"></i>Central Bank of Nigeria (CBN) Adverse Action Statement</h6>
                            <p class="small mb-1">In compliance with CBN consumer credit regulations, this decline was driven by:</p>
                            ${reasonsHtml}
                        `;
                        adverseDiv.style.display = 'block';
                    }
                } catch (err) {
                    cardDiv.className = 'score-box score-rejected';
                    cardDiv.innerHTML = `<p class="text-danger">Error: ${err.message}</p>`;
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/api")
def root_api():
    return {
        "institution": "Sterling Bank Nigeria",
        "system": "Credit Risk Analysis & Loan Default Prediction API",
        "dual_architecture": ["personal_loans", "retail_sme_loans"],
        "status": "operational",
        "endpoints": ["/predict/personal", "/predict/retail", "/explain/personal", "/explain/retail", "/metrics/{portfolio_type}"]
    }

@app.get("/health")
def health_check():
    personal_ready = os.path.exists(os.path.join(MODELS_DIR, "personal_xgboost.joblib"))
    retail_ready = os.path.exists(os.path.join(MODELS_DIR, "retail_xgboost.joblib"))
    return {
        "status": "healthy" if (personal_ready and retail_ready) else "models_pending",
        "personal_model_loaded": personal_ready,
        "retail_model_loaded": retail_ready
    }

@app.post("/predict/personal")
def predict_personal(application: PersonalLoanRequest):
    model, prep = load_model_and_prep("personal")
    features = prep.transform_single(application.dict())
    prob = float(model.predict_proba(features)[0, 1])
    opt_threshold = get_optimal_threshold("personal")
    
    if prob < (opt_threshold * 0.70):
        decision = "APPROVE"
        tier = "LOW RISK"
    elif prob < opt_threshold:
        decision = "MANUAL REVIEW"
        tier = "MODERATE RISK"
    else:
        decision = "REJECT"
        tier = "HIGH RISK"
        
    return {
        "portfolio": "Personal Loan (Salary Earner)",
        "applicant": application.dict(),
        "default_probability": round(prob, 4),
        "decision": decision,
        "risk_tier": tier,
        "optimal_threshold_applied": round(opt_threshold, 3),
        "repayment_likelihood": round((1.0 - prob) * 100, 2)
    }

@app.post("/predict/retail")
def predict_retail(application: RetailLoanRequest):
    model, prep = load_model_and_prep("retail")
    features = prep.transform_single(application.dict())
    prob = float(model.predict_proba(features)[0, 1])
    opt_threshold = get_optimal_threshold("retail")
    
    if prob < (opt_threshold * 0.70):
        decision = "APPROVE"
        tier = "LOW RISK"
    elif prob < opt_threshold:
        decision = "MANUAL REVIEW"
        tier = "MODERATE RISK"
    else:
        decision = "REJECT"
        tier = "HIGH RISK"
        
    return {
        "portfolio": "Retail SME Loan",
        "applicant": application.dict(),
        "default_probability": round(prob, 4),
        "decision": decision,
        "risk_tier": tier,
        "optimal_threshold_applied": round(opt_threshold, 3),
        "repayment_likelihood": round((1.0 - prob) * 100, 2)
    }

@app.post("/explain/personal")
def explain_personal(application: PersonalLoanRequest):
    from src.explainability import CreditRiskExplainer
    explainer = CreditRiskExplainer("personal")
    pred_res = predict_personal(application)
    explanation = explainer.explain_instance(application.dict())
    return {
        **pred_res,
        "xai_explanation": explanation
    }

@app.post("/explain/retail")
def explain_retail(application: RetailLoanRequest):
    from src.explainability import CreditRiskExplainer
    explainer = CreditRiskExplainer("retail")
    pred_res = predict_retail(application)
    explanation = explainer.explain_instance(application.dict())
    return {
        **pred_res,
        "xai_explanation": explanation
    }

@app.get("/metrics/{portfolio_type}")
def get_metrics(portfolio_type: str):
    p_type = portfolio_type.lower()
    metrics_path = os.path.join(RESULTS_DIR, f"{p_type}_metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail=f"Metrics for {p_type} not found.")
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    return data
