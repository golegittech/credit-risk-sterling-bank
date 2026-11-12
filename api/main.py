"""
FastAPI Backend & Interactive Underwriting Dashboard for Sterling Bank Nigeria.
Full-featured cloud application containing:
- 4 Operational Modes (Underwriting, Benchmarks, Global XAI, Batch Audit)
- Plotly Gauge Chart & SHAP Local Feature Attribution Bar Chart
- High-res ROC/PR curves & Financial Cost curves
- Interactive CBN regulatory framework modals
- RESTful JSON endpoints at /predict, /explain, /metrics and /docs
"""

import os
import json
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd

app = FastAPI(
    title="Sterling Bank Credit Risk & Loan Default Prediction API",
    description="Dual-Model Machine Learning API (Personal Salary Earner & Retail SME) with SHAP Explainability",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Mount results directory to serve generated charts & curves directly
if os.path.exists(RESULTS_DIR):
    app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

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

# ----------------- Main Dashboard UI ----------------- #

@app.get("/", response_class=HTMLResponse)
def dashboard_html():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sterling Bank Nigeria — Credit Risk Decision Engine</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
            :root { --primary-crimson: #8B0000; --dark-crimson: #5c0000; --gold: #c59b27; }
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; color: #212529; }
            .header-banner { background: linear-gradient(135deg, var(--primary-crimson) 0%, #2b0000 100%); color: white; padding: 22px 20px; border-bottom: 4px solid var(--gold); }
            .sidebar-card { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 20px; }
            .nav-sidebar .nav-link { color: #495057; font-weight: 600; padding: 12px 16px; border-radius: 8px; margin-bottom: 6px; transition: all 0.2s; }
            .nav-sidebar .nav-link:hover { background-color: #f1f3f5; color: var(--primary-crimson); }
            .nav-sidebar .nav-link.active { background-color: var(--primary-crimson); color: white; }
            .main-content { background: white; border-radius: 12px; padding: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
            .btn-sterling { background-color: var(--primary-crimson); color: white; font-weight: 600; padding: 12px 20px; border-radius: 8px; border: none; }
            .btn-sterling:hover { background-color: var(--dark-crimson); color: white; }
            .score-box { border-radius: 10px; padding: 20px; text-align: center; margin-top: 20px; }
            .score-approved { background-color: #e8f5e9; border: 2px solid #28a745; color: #155724; }
            .score-review { background-color: #fffde7; border: 2px solid #ffc107; color: #856404; }
            .score-rejected { background-color: #ffebee; border: 2px solid #dc3545; color: #721c24; }
            .adverse-notice { background-color: #f8d7da; border-left: 5px solid #dc3545; padding: 15px; border-radius: 6px; margin-top: 15px; }
            .reg-badge { cursor: pointer; display: block; padding: 8px 12px; border-radius: 6px; font-size: 0.82rem; font-weight: 600; text-align: center; margin-bottom: 8px; text-decoration: none; transition: transform 0.15s; }
            .reg-badge:hover { transform: translateY(-2px); }
            .badge-cbn { background-color: #212529; color: white; }
            .badge-shap { background-color: var(--gold); color: white; }
            .badge-loss { background-color: var(--primary-crimson); color: white; }
            .chart-img { width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 15px; }
            .table-custom th { background-color: var(--primary-crimson); color: white; }
        </style>
    </head>
    <body>
        <div class="header-banner">
            <div class="container-fluid px-4">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <h3 class="mb-0 fw-bold"><i class="fa-solid fa-building-columns me-2"></i>STERLING BANK NIGERIA</h3>
                        <p class="mb-0 text-white-50 small">Interpretable Credit Risk Analytics & Loan Default Prediction Engine</p>
                    </div>
                    <div>
                        <a href="/docs" target="_blank" class="btn btn-outline-light btn-sm"><i class="fa-solid fa-code me-1"></i> Swagger API Docs</a>
                    </div>
                </div>
            </div>
        </div>

        <div class="container-fluid px-4 my-4">
            <div class="row g-4">
                <!-- SIDEBAR -->
                <div class="col-lg-3">
                    <div class="sidebar-card">
                        <h6 class="text-uppercase text-muted fw-bold mb-3 small"><i class="fa-solid fa-sliders me-2"></i>Operational Controls</h6>
                        <ul class="nav flex-column nav-sidebar" id="dashboardTabs" role="tablist">
                            <li class="nav-item">
                                <a class="nav-link active" data-bs-toggle="pill" href="#tab-underwriting"><i class="fa-solid fa-user-check me-2"></i>Loan Underwriting</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" data-bs-toggle="pill" href="#tab-benchmarks" onclick="loadBenchmarks('personal')"><i class="fa-solid fa-chart-line me-2"></i>Portfolio Benchmarks</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" data-bs-toggle="pill" href="#tab-xai" onclick="loadXAI('personal')"><i class="fa-solid fa-brain me-2"></i>Global XAI & Risk Drivers</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" data-bs-toggle="pill" href="#tab-audit" onclick="loadBatchAudit('personal')"><i class="fa-solid fa-table-list me-2"></i>Batch Portfolio Audit</a>
                            </li>
                        </ul>
                    </div>

                    <div class="sidebar-card">
                        <h6 class="text-uppercase text-muted fw-bold mb-3 small"><i class="fa-solid fa-id-card me-2"></i>Project Information</h6>
                        <p class="small mb-1"><strong>Researcher:</strong> Mbaegbu Chinedum Grace</p>
                        <p class="small mb-1"><strong>Matric No:</strong> 2025/A/DSC/0058</p>
                        <p class="small mb-1"><strong>Supervisor:</strong> Dr. A. S. Makinde</p>
                        <p class="small mb-1"><strong>Degree:</strong> B.Sc. (Hons) Data Science</p>
                        <p class="small mb-3"><strong>Institution:</strong> Miva Open University, Abuja</p>
                        <hr>
                        <h6 class="text-uppercase text-muted fw-bold mb-2 small"><i class="fa-solid fa-scale-balanced me-2"></i>Regulatory Standards</h6>
                        <a href="#" class="reg-badge badge-cbn" data-bs-toggle="modal" data-bs-target="#cbnModal">
                            <i class="fa-solid fa-shield-halved me-1"></i> CBN Consumer Protection
                        </a>
                        <a href="#" class="reg-badge badge-shap" data-bs-toggle="modal" data-bs-target="#shapModal">
                            <i class="fa-solid fa-magnifying-glass-chart me-1"></i> SHAP Explainable AI
                        </a>
                        <a href="#" class="reg-badge badge-loss" data-bs-toggle="modal" data-bs-target="#lossModal">
                            <i class="fa-solid fa-money-bill-trend-up me-1"></i> Loss Matrix Optimization
                        </a>
                    </div>
                </div>

                <!-- MAIN CONTENT AREA -->
                <div class="col-lg-9">
                    <div class="main-content">
                        <div class="tab-content" id="dashboardTabsContent">
                            
                            <!-- TAB 1: UNDERWRITING -->
                            <div class="tab-pane fade show active" id="tab-underwriting">
                                <div class="d-flex justify-content-between align-items-center mb-3">
                                    <h4 class="fw-bold mb-0">Automated Loan Application Underwriting</h4>
                                    <div class="btn-group" role="group">
                                        <input type="radio" class="btn-check" name="portfolio_type" id="port_personal" checked onchange="togglePortfolioForm('personal')">
                                        <label class="btn btn-outline-danger btn-sm" for="port_personal">Personal (Salary Earner)</label>
                                        <input type="radio" class="btn-check" name="portfolio_type" id="port_retail" onchange="togglePortfolioForm('retail')">
                                        <label class="btn btn-outline-danger btn-sm" for="port_retail">Retail SME Loan</label>
                                    </div>
                                </div>
                                <p class="text-muted small">Real-time credit risk evaluation calibrated against Sterling Bank's financial loss-minimizing threshold.</p>

                                <!-- PERSONAL FORM -->
                                <div id="form-personal-container">
                                    <div class="row g-3">
                                        <div class="col-md-4">
                                            <label class="form-label small fw-bold">Applicant Age (Years)</label>
                                            <input type="number" class="form-control" id="p_age" value="34" min="21" max="65">
                                        </div>
                                        <div class="col-md-4">
                                            <label class="form-label small fw-bold">Housing Status</label>
                                            <select class="form-select" id="p_housing">
                                                <option value="RENT">RENT</option>
                                                <option value="MORTGAGE">MORTGAGE</option>
                                                <option value="OWN">OWN</option>
                                            </select>
                                        </div>
                                        <div class="col-md-4">
                                            <label class="form-label small fw-bold">Employment Length (Years)</label>
                                            <input type="number" step="0.5" class="form-control" id="p_emp" value="5.5">
                                        </div>
                                        <div class="col-md-6">
                                            <label class="form-label small fw-bold">Gross Annual Salary (NGN)</label>
                                            <input type="number" class="form-control" id="p_income" value="5500000" step="100000">
                                        </div>
                                        <div class="col-md-6">
                                            <label class="form-label small fw-bold">Requested Loan (NGN)</label>
                                            <input type="number" class="form-control" id="p_loan" value="1800000" step="50000">
                                        </div>
                                        <div class="col-md-4">
                                            <label class="form-label small fw-bold">Debt-to-Income (DTI) %</label>
                                            <input type="number" step="0.1" class="form-control" id="p_dti" value="28.5">
                                        </div>
                                        <div class="col-md-4">
                                            <label class="form-label small fw-bold">Credit Bureau Score</label>
                                            <input type="number" class="form-control" id="p_score" value="680" min="300" max="850">
                                        </div>
                                        <div class="col-md-4">
                                            <label class="form-label small fw-bold">External Loan Status</label>
                                            <select class="form-select" id="p_ext">
                                                <option value="None">None</option>
                                                <option value="Performing" selected>Performing</option>
                                                <option value="Defaulting">Defaulting</option>
                                            </select>
                                        </div>
                                        <div class="col-12 mt-3">
                                            <button class="btn btn-sterling w-100" onclick="evaluatePersonal()">
                                                <i class="fa-solid fa-calculator me-2"></i>Evaluate Personal Credit Risk & Explain Decision
                                            </button>
                                        </div>
                                    </div>
                                </div>

                                <!-- RETAIL SME FORM -->
                                <div id="form-retail-container" style="display:none;">
                                    <div class="row g-3">
                                        <div class="col-md-4">
                                            <label class="form-label small fw-bold">Business Age (CAC Years)</label>
                                            <input type="number" step="0.5" class="form-control" id="r_age" value="6.0">
                                        </div>
                                        <div class="col-md-4">
                                            <label class="form-label small fw-bold">Nature of Business</label>
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
                                            <label class="form-label small fw-bold">Operational Region</label>
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
                                            <label class="form-label small fw-bold">Facility Type</label>
                                            <select class="form-select" id="r_facility">
                                                <option value="Term Loan">Term Loan</option>
                                                <option value="Overdraft">Overdraft</option>
                                                <option value="Asset Finance">Asset Finance</option>
                                                <option value="Invoice Discounting">Invoice Discounting</option>
                                            </select>
                                        </div>
                                        <div class="col-md-4">
                                            <label class="form-label small fw-bold">Facility Amount (NGN)</label>
                                            <input type="number" class="form-control" id="r_loan" value="8500000" step="250000">
                                        </div>
                                        <div class="col-md-4">
                                            <label class="form-label small fw-bold">Lending Interest Rate (%)</label>
                                            <input type="number" step="0.1" class="form-control" id="r_rate" value="24.5">
                                        </div>
                                        <div class="col-md-12">
                                            <label class="form-label small fw-bold">Concurrent External Loan Status</label>
                                            <select class="form-select" id="r_ext">
                                                <option value="None">None</option>
                                                <option value="Performing" selected>Performing</option>
                                                <option value="Defaulting">Defaulting</option>
                                            </select>
                                        </div>
                                        <div class="col-12 mt-3">
                                            <button class="btn btn-sterling w-100" onclick="evaluateRetail()">
                                                <i class="fa-solid fa-briefcase me-2"></i>Evaluate SME Commercial Risk & Explain Decision
                                            </button>
                                        </div>
                                    </div>
                                </div>

                                <!-- UNDERWRITING OUTPUT DISPLAY -->
                                <div id="underwritingResult" class="mt-4" style="display:none;">
                                    <hr>
                                    <h5 class="fw-bold mb-3"><i class="fa-solid fa-chart-pie me-2"></i>Underwriter Assessment & Risk Breakdown</h5>
                                    
                                    <div class="row align-items-center">
                                        <div class="col-md-6">
                                            <div id="gaugePlot" style="height: 280px;"></div>
                                        </div>
                                        <div class="col-md-6">
                                            <div id="decisionCard" class="score-box"></div>
                                        </div>
                                    </div>

                                    <div id="adverseContainer" style="display:none;"></div>

                                    <h6 class="fw-bold mt-4 mb-2"><i class="fa-solid fa-magnifying-glass-arrow-right me-2"></i>Local SHAP Feature Attribution (Risk Impact)</h6>
                                    <p class="text-muted small">Quantifies the positive (risk-accelerating, red) and negative (risk-mitigating, green) contribution of each feature to the borrower's default score.</p>
                                    <div id="shapBarPlot" style="height: 350px;"></div>
                                </div>
                            </div>

                            <!-- TAB 2: BENCHMARKS -->
                            <div class="tab-pane fade" id="tab-benchmarks">
                                <div class="d-flex justify-content-between align-items-center mb-3">
                                    <h4 class="fw-bold mb-0">Model Benchmarks & Financial Loss Minimization</h4>
                                    <select class="form-select w-auto" id="benchPortfolioSelect" onchange="loadBenchmarks(this.value)">
                                        <option value="personal">Personal Loan Portfolio</option>
                                        <option value="retail">Retail SME Portfolio</option>
                                    </select>
                                </div>
                                <div class="table-responsive mb-4">
                                    <table class="table table-bordered table-hover table-custom" id="benchmarkTable">
                                        <thead>
                                            <tr>
                                                <th>Model Architecture</th>
                                                <th>ROC-AUC</th>
                                                <th>PR-AUC</th>
                                                <th>F1-Score</th>
                                                <th>Recall</th>
                                                <th>Precision</th>
                                                <th>Accuracy</th>
                                                <th>Optimal Cutoff</th>
                                                <th>Savings (NGN M)</th>
                                            </tr>
                                        </thead>
                                        <tbody id="benchmarkTableBody">
                                            <tr><td colspan="9" class="text-center">Loading model benchmarks...</td></tr>
                                        </tbody>
                                    </table>
                                </div>
                                <div class="row">
                                    <div class="col-md-6">
                                        <h6 class="fw-bold text-center">ROC and Precision-Recall Evaluation Curves</h6>
                                        <img id="benchRocImg" src="/results/personal_roc_pr_curves.png" class="chart-img" alt="ROC Curves">
                                    </div>
                                    <div class="col-md-6">
                                        <h6 class="fw-bold text-center">Expected Financial Loss vs Decision Cutoff</h6>
                                        <img id="benchCostImg" src="/results/personal_cost_curve.png" class="chart-img" alt="Financial Cost Curves">
                                    </div>
                                </div>
                            </div>

                            <!-- TAB 3: GLOBAL XAI -->
                            <div class="tab-pane fade" id="tab-xai">
                                <div class="d-flex justify-content-between align-items-center mb-3">
                                    <h4 class="fw-bold mb-0">Global Explainable AI (XAI) & Factor Attribution</h4>
                                    <select class="form-select w-auto" id="xaiPortfolioSelect" onchange="loadXAI(this.value)">
                                        <option value="personal">Personal Loan Portfolio</option>
                                        <option value="retail">Retail SME Portfolio</option>
                                    </select>
                                </div>
                                <p class="text-muted small">Systemic credit risk direction and feature sensitivity across the portfolio. <strong>Red bars</strong> indicate factors that accelerate default risk; <strong>Green bars</strong> indicate protective factors that reduce default likelihood and support approval.</p>
                                <div class="row">
                                    <div class="col-md-6">
                                        <h6 class="fw-bold text-center text-danger"><i class="fa-solid fa-arrows-split-up-and-left me-1"></i>Risk Accelerators vs. Protective Mitigators</h6>
                                        <img id="xaiBeeswarmImg" src="/results/personal_shap_beeswarm.png" class="chart-img" alt="Directional Risk Drivers">
                                    </div>
                                    <div class="col-md-6">
                                        <h6 class="fw-bold text-center text-dark"><i class="fa-solid fa-ranking-star me-1"></i>Overall Feature Importance Magnitude</h6>
                                        <img id="xaiSummaryImg" src="/results/personal_shap_summary.png" class="chart-img" alt="SHAP Importance Summary">
                                    </div>
                                </div>
                                <div class="card bg-light p-3 mt-3 border-0">
                                    <h6 class="fw-bold text-dark"><i class="fa-solid fa-circle-info me-2"></i>Key Regulatory Risk Drivers</h6>
                                    <div id="xaiInsightText" class="small text-muted"></div>
                                </div>
                            </div>

                            <!-- TAB 4: BATCH AUDIT -->
                            <div class="tab-pane fade" id="tab-audit">
                                <div class="d-flex justify-content-between align-items-center mb-3">
                                    <h4 class="fw-bold mb-0">Batch Portfolio Credit Scoring & Compliance Audit</h4>
                                    <div>
                                        <select class="form-select d-inline-block w-auto me-2" id="auditPortfolioSelect" onchange="loadBatchAudit(this.value)">
                                            <option value="personal">Personal Loan</option>
                                            <option value="retail">Retail SME</option>
                                        </select>
                                        <button class="btn btn-sm btn-outline-danger" onclick="downloadBatchAudit()"><i class="fa-solid fa-download me-1"></i> Download Scored Audit CSV</button>
                                    </div>
                                </div>
                                <p class="text-muted small">Representative sample of 50 accounts scored against calibrated loss-minimizing threshold.</p>
                                <div class="table-responsive" style="max-height: 480px; overflow-y: auto;">
                                    <table class="table table-sm table-striped table-hover small" id="auditTable">
                                        <thead class="table-dark" id="auditTableHead"></thead>
                                        <tbody id="auditTableBody"></tbody>
                                    </table>
                                </div>
                            </div>

                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- MODAL 1: CBN CONSUMER PROTECTION -->
        <div class="modal fade" id="cbnModal" tabindex="-1">
            <div class="modal-dialog modal-dialog-centered">
                <div class="modal-content">
                    <div class="modal-header bg-dark text-white">
                        <h5 class="modal-title"><i class="fa-solid fa-shield-halved me-2"></i>CBN Consumer Protection Framework</h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body small">
                        <p>Under <strong>Section 3.2 of the Central Bank of Nigeria (CBN) Consumer Protection Framework</strong> for Banks and Financial Institutions:</p>
                        <ul>
                            <li>Lending institutions must ensure automated credit scoring algorithms are free of disparate demographic bias.</li>
                            <li>Consumers have the statutory right to understand the precise financial and behavioral reasons behind adverse credit actions (loan denials).</li>
                            <li>This system enforces full compliance by automatically synthesizing statutory adverse action statements directly from the top SHAP mathematical contributors.</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>

        <!-- MODAL 2: SHAP XAI -->
        <div class="modal fade" id="shapModal" tabindex="-1">
            <div class="modal-dialog modal-dialog-centered">
                <div class="modal-content">
                    <div class="modal-header bg-warning text-dark">
                        <h5 class="modal-title"><i class="fa-solid fa-brain me-2"></i>SHAP (SHapley Additive exPlanations)</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body small">
                        <p><strong>SHAP</strong> is rooted in cooperative game theory (Lundberg & Lee, 2017):</p>
                        <ul>
                            <li>Each applicant feature acts as a player in a collaborative game estimating default risk.</li>
                            <li>Shapley values allocate fair additive payoffs based on marginal contributions across all feature coalitions.</li>
                            <li>Guarantees local accuracy, consistency, and eliminate the "black-box" dilemma of tree-boosting ensembles.</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>

        <!-- MODAL 3: LOSS MATRIX -->
        <div class="modal fade" id="lossModal" tabindex="-1">
            <div class="modal-dialog modal-dialog-centered">
                <div class="modal-content">
                    <div class="modal-header bg-danger text-white">
                        <h5 class="modal-title"><i class="fa-solid fa-money-bill-trend-up me-2"></i>Financial Loss Matrix Optimization</h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body small">
                        <p>Standard 0.50 classification thresholds fail because credit risk errors are asymmetric:</p>
                        <ul>
                            <li><strong>False Negative (Defaulter approved):</strong> Costs <em>EAD × LGD</em> (direct capital write-off).</li>
                            <li><strong>False Positive (Good applicant rejected):</strong> Costs only the forgone net interest margin.</li>
                            <li>Our threshold calibration sweeps from 0.05 to 0.95, selecting cutoffs that minimize total expected financial loss and save tens of millions of Naira for Sterling Bank.</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            let currentAuditData = [];

            function togglePortfolioForm(type) {
                if (type === 'personal') {
                    document.getElementById('form-personal-container').style.display = 'block';
                    document.getElementById('form-retail-container').style.display = 'none';
                } else {
                    document.getElementById('form-personal-container').style.display = 'none';
                    document.getElementById('form-retail-container').style.display = 'block';
                }
                document.getElementById('underwritingResult').style.display = 'none';
            }

            async function evaluatePersonal() {
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
                runEvaluation('/explain/personal', payload);
            }

            async function evaluateRetail() {
                const payload = {
                    business_age_years: parseFloat(document.getElementById('r_age').value),
                    nature_of_business: document.getElementById('r_nature').value,
                    business_region: document.getElementById('r_region').value,
                    facility_type: document.getElementById('r_facility').value,
                    loan_amount: parseFloat(document.getElementById('r_loan').value),
                    interest_rate: parseFloat(document.getElementById('r_rate').value),
                    existing_external_loan_status: document.getElementById('r_ext').value
                };
                runEvaluation('/explain/retail', payload);
            }

            async function runEvaluation(endpoint, payload) {
                const outDiv = document.getElementById('underwritingResult');
                const cardDiv = document.getElementById('decisionCard');
                const adverseDiv = document.getElementById('adverseContainer');
                outDiv.style.display = 'block';
                cardDiv.className = 'score-box';
                cardDiv.innerHTML = '<div class="spinner-border text-danger" role="status"></div><p class="mt-2 mb-0">Computing XGBoost Risk Scores & SHAP attributions...</p>';
                adverseDiv.style.display = 'none';

                try {
                    const res = await fetch(endpoint, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    const data = await res.json();
                    const prob = data.default_probability;
                    const optTh = data.optimal_threshold_applied;

                    // Decision Box
                    let cls = 'score-approved';
                    let icon = 'fa-circle-check';
                    if (data.decision === 'MANUAL REVIEW') { cls = 'score-review'; icon = 'fa-triangle-exclamation'; }
                    else if (data.decision === 'REJECT') { cls = 'score-rejected'; icon = 'fa-ban'; }

                    cardDiv.className = 'score-box ' + cls;
                    cardDiv.innerHTML = `
                        <h4 class="fw-bold mb-2"><i class="fa-solid ${icon} me-2"></i>DECISION: ${data.decision}</h4>
                        <p class="mb-1"><strong>Predicted Probability of Default:</strong> ${(prob * 100).toFixed(1)}%</p>
                        <p class="mb-1"><strong>Repayment Likelihood:</strong> ${data.repayment_likelihood}%</p>
                        <p class="mb-0"><strong>Applied Financial Optimal Cutoff:</strong> ${optTh}</p>
                    `;

                    // Plotly Gauge Chart
                    const gaugeData = [{
                        type: "indicator",
                        mode: "gauge+number",
                        value: prob * 100,
                        title: { text: "Default Probability (%)", font: { size: 16 } },
                        gauge: {
                            axis: { range: [0, 100] },
                            bar: { color: prob >= optTh ? "#8B0000" : "#28a745" },
                            steps: [
                                { range: [0, optTh * 70], color: "#e8f5e9" },
                                { range: [optTh * 70, optTh * 100], color: "#fffde7" },
                                { range: [optTh * 100, 100], color: "#ffebee" }
                            ],
                            threshold: {
                                line: { color: "red", width: 4 },
                                thickness: 0.75,
                                value: optTh * 100
                            }
                        }
                    }];
                    const gaugeLayout = { height: 260, margin: { t: 30, b: 20, l: 30, r: 30 } };
                    Plotly.newPlot('gaugePlot', gaugeData, gaugeLayout, { responsive: true, displayModeBar: false });

                    // CBN Adverse Statement
                    if (data.decision === 'REJECT' && data.xai_explanation && data.xai_explanation.adverse_action_reasons) {
                        let listHtml = '';
                        data.xai_explanation.adverse_action_reasons.forEach(r => { listHtml += `<li>${r}</li>`; });
                        adverseDiv.className = 'adverse-notice';
                        adverseDiv.innerHTML = `
                            <h6 class="fw-bold mb-2"><i class="fa-solid fa-scale-balanced me-2"></i>Central Bank of Nigeria (CBN) Adverse Action Statement</h6>
                            <p class="small mb-1">In compliance with Section 3.2 of the CBN Consumer Protection Framework, this loan denial was driven by:</p>
                            <ul class="mb-0 small">${listHtml}</ul>
                        `;
                        adverseDiv.style.display = 'block';
                    }

                    // Local SHAP Feature Impact Bar Chart
                    if (data.xai_explanation && data.xai_explanation.top_attributions) {
                        const attrs = data.xai_explanation.top_attributions.slice().reverse();
                        const yNames = attrs.map(a => a.feature.replace(/_/g, ' '));
                        const xVals = attrs.map(a => a.shap_value);
                        const colors = attrs.map(a => a.shap_value > 0 ? '#d9534f' : '#5cb85c');

                        const barData = [{
                            type: 'bar',
                            x: xVals,
                            y: yNames,
                            orientation: 'h',
                            marker: { color: colors }
                        }];
                        const barLayout = {
                            title: 'Feature Impact on Default Risk (SHAP Attributions)',
                            xaxis: { title: 'SHAP Value (Left: Reduces Risk | Right: Increases Risk)' },
                            margin: { l: 150, r: 30, t: 40, b: 40 },
                            height: 330
                        };
                        Plotly.newPlot('shapBarPlot', barData, barLayout, { responsive: true, displayModeBar: false });
                    }

                } catch (err) {
                    cardDiv.className = 'score-box score-rejected';
                    cardDiv.innerHTML = `<p class="text-danger mb-0">Error communicating with API: ${err.message}</p>`;
                }
            }

            async function loadBenchmarks(portfolio) {
                const tbody = document.getElementById('benchmarkTableBody');
                tbody.innerHTML = '<tr><td colspan="9" class="text-center"><div class="spinner-border spinner-border-sm text-danger"></div> Loading...</td></tr>';
                document.getElementById('benchRocImg').src = `/results/${portfolio}_roc_pr_curves.png`;
                document.getElementById('benchCostImg').src = `/results/${portfolio}_cost_curve.png`;

                try {
                    const res = await fetch(`/metrics/${portfolio}`);
                    const data = await res.json();
                    let rowsHtml = '';
                    for (const [mName, mVals] of Object.entries(data)) {
                        const fin = mVals.financial_optimization || {};
                        const savings = (fin.monetary_savings_ngn || 0) / 1e6;
                        rowsHtml += `
                            <tr>
                                <td><strong>${mName.replace(/_/g, ' ')}</strong></td>
                                <td>${mVals.roc_auc.toFixed(4)}</td>
                                <td>${mVals.pr_auc.toFixed(4)}</td>
                                <td>${mVals.f1_score.toFixed(4)}</td>
                                <td>${(mVals.recall * 100).toFixed(1)}%</td>
                                <td>${(mVals.precision * 100).toFixed(1)}%</td>
                                <td>${(mVals.accuracy * 100).toFixed(1)}%</td>
                                <td>${(fin.optimal_threshold || 0.5).toFixed(2)}</td>
                                <td class="text-success fw-bold">NGN ${savings.toFixed(2)}M</td>
                            </tr>
                        `;
                    }
                    tbody.innerHTML = rowsHtml;
                } catch (e) {
                    tbody.innerHTML = `<tr><td colspan="9" class="text-danger text-center">Failed to load metrics.</td></tr>`;
                }
            }

            function loadXAI(portfolio) {
                document.getElementById('xaiBeeswarmImg').src = `/results/${portfolio}_shap_beeswarm.png`;
                document.getElementById('xaiSummaryImg').src = `/results/${portfolio}_shap_summary.png`;
                const insightDiv = document.getElementById('xaiInsightText');
                if (portfolio === 'personal') {
                    insightDiv.innerHTML = `
                        <ul>
                            <li><strong>Debt-to-Income (DTI) Ratio:</strong> Dominates systemic attribution. Applicants with DTI &gt; 45% exhibit steep positive SHAP pushes towards default.</li>
                            <li><strong>Credit Bureau Score:</strong> High negative correlation; scores &gt; 700 provide massive protection against default odds.</li>
                            <li><strong>External Delinquency:</strong> Active default on outside facilities immediately elevates log-odds into high-risk tiers.</li>
                        </ul>
                    `;
                } else {
                    insightDiv.innerHTML = `
                        <ul>
                            <li><strong>Concurrent External Delinquency:</strong> The single greatest risk driver in the Retail SME portfolio.</li>
                            <li><strong>Facility Interest Rate:</strong> Rates exceeding 26% create severe cashflow debt servicing strain.</li>
                            <li><strong>Business Age (CAC Years):</strong> Operating history &gt; 5 years provides significant shielding against loan default.</li>
                        </ul>
                    `;
                }
            }

            async function loadBatchAudit(portfolio) {
                if (document.getElementById('auditPortfolioSelect')) {
                    document.getElementById('auditPortfolioSelect').value = portfolio;
                }
                const thead = document.getElementById('auditTableHead');
                const tbody = document.getElementById('auditTableBody');
                tbody.innerHTML = '<tr><td colspan="10" class="text-center"><div class="spinner-border spinner-border-sm text-danger"></div> Loading 50 accounts...</td></tr>';

                try {
                    const res = await fetch(`/audit-sample/${portfolio}`);
                    if (!res.ok) {
                        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
                    }
                    const json = await res.json();
                    currentAuditData = json.data;
                    
                    if (currentAuditData && currentAuditData.length > 0) {
                        let headHtml = '<tr>';
                        Object.keys(currentAuditData[0]).forEach(k => { headHtml += `<th>${k.replace(/_/g, ' ')}</th>`; });
                        headHtml += '</tr>';
                        thead.innerHTML = headHtml;

                        let bodyHtml = '';
                        currentAuditData.forEach(row => {
                            bodyHtml += '<tr>';
                            Object.entries(row).forEach(([k, val]) => {
                                if (k === 'Underwriter_Decision') {
                                    let badge = val === 'APPROVE' ? 'bg-success' : (val === 'MANUAL REVIEW' ? 'bg-warning text-dark' : 'bg-danger');
                                    bodyHtml += `<td><span class="badge ${badge}">${val}</span></td>`;
                                } else {
                                    bodyHtml += `<td>${val}</td>`;
                                }
                            });
                            bodyHtml += '</tr>';
                        });
                        tbody.innerHTML = bodyHtml;
                    } else {
                        tbody.innerHTML = '<tr><td colspan="10" class="text-muted text-center">No records available.</td></tr>';
                    }
                } catch (e) {
                    console.error('Audit sample error:', e);
                    tbody.innerHTML = `<tr><td colspan="10" class="text-danger text-center">Failed to load audit sample (${e.message}).</td></tr>`;
                }
            }

            function downloadBatchAudit() {
                if (!currentAuditData || currentAuditData.length === 0) return;
                const keys = Object.keys(currentAuditData[0]);
                let csv = keys.join(',') + '\\n';
                currentAuditData.forEach(row => {
                    csv += keys.map(k => JSON.stringify(row[k] || '')).join(',') + '\\n';
                });
                const blob = new Blob([csv], { type: 'text/csv' });
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.setAttribute('href', url);
                a.setAttribute('download', `sterling_bank_batch_audit.csv`);
                a.click();
            }
        </script>
    </body>
    </html>
    """

# ----------------- Endpoints ----------------- #

@app.get("/audit-sample/{portfolio_type}")
def get_audit_sample(portfolio_type: str):
    p_type = "personal" if "personal" in portfolio_type.lower() else "retail"
    csv_file = os.path.join(DATA_DIR, f"{p_type}_loans.csv" if p_type == "personal" else "retail_sme_loans.csv")
    
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file).head(50)
    else:
        from src.data_generator import generate_personal_loans, generate_retail_sme_loans
        df = generate_personal_loans(n_samples=50) if p_type == "personal" else generate_retail_sme_loans(n_samples=50)
    
    model, prep = load_model_and_prep(p_type)
    opt_th = get_optimal_threshold(p_type)
    
    X_batch, _ = prep.transform_test(df)
    probs = model.predict_proba(X_batch)[:, 1]
    
    df_res = df.copy()
    df_res["Predicted_Default_Prob"] = np.round(probs, 4)
    df_res["Underwriter_Decision"] = np.where(probs < (opt_th * 0.70), "APPROVE", np.where(probs < opt_th, "MANUAL REVIEW", "REJECT"))
    df_res = df_res.replace({np.nan: "N/A"}).fillna("N/A")
    
    return {"data": df_res.to_dict(orient="records")}

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
