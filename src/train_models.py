"""
Model Training and Benchmarking Pipeline.
Trains and benchmarks 4 classification architectures for both Personal and Retail (SME) portfolios:
1. Logistic Regression (Baseline)
2. Random Forest
3. XGBoost (Primary architecture)
4. LightGBM

Computes comprehensive statistical metrics (ROC-AUC, PR-AUC, F1, Recall, Precision, Accuracy)
and financial cost matrices, saves trained artifacts and generated figures.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
import joblib

from src.data_generator import save_datasets
from src.preprocessing import prepare_portfolio_splits, CreditDataPreprocessor
from src.financial_cost import optimize_decision_threshold, evaluate_financial_loss, COST_PARAMS

# Style configuration
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'Arial', 'font.size': 11})

def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

def train_portfolio(portfolio_type: str, csv_path: str, models_dir: str = "models", results_dir: str = "results"):
    print(f"\n=======================================================")
    print(f" TRAINING PORTFOLIO: {portfolio_type.upper()}")
    print(f"=======================================================")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Prepare splits & preprocessor
    splits = prepare_portfolio_splits(csv_path, portfolio_type=portfolio_type)
    X_train_res = splits['X_train_resampled']
    y_train_res = splits['y_train_resampled']
    X_train_raw = splits['X_train_raw']
    y_train_raw = splits['y_train_raw']
    X_test = splits['X_test']
    y_test = splits['y_test']
    feature_names = splits['feature_names']
    preprocessor = splits['preprocessor']
    
    # Save preprocessor
    prep_path = os.path.join(models_dir, f"{portfolio_type}_preprocessor.joblib")
    preprocessor.save(prep_path)
    print(f"[OK] Preprocessor saved to {prep_path}")
    print(f"    Train shape (raw): {X_train_raw.shape}, Default rate: {y_train_raw.mean():.2%}")
    print(f"    Train shape (SMOTE): {X_train_res.shape}, Default rate: {y_train_res.mean():.2%}")
    print(f"    Test shape (Untouched): {X_test.shape}, Default rate: {y_test.mean():.2%}")
    
    # Standardize data for Logistic Regression
    X_train_res_scaled = preprocessor.scaler.transform(X_train_res)
    X_test_scaled = preprocessor.scaler.transform(X_test)
    X_train_raw_scaled = preprocessor.scaler.transform(X_train_raw)

    # 2. Instantiate Models
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "Random_Forest": RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42),
        "LightGBM": lgb.LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=250, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss')
    }
    
    results = {}
    test_probs = {}
    
    for name, model in models.items():
        print(f" -> Training {name} with SMOTE...")
        if name == "Logistic_Regression":
            model.fit(X_train_res_scaled, y_train_res)
            probs = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train_res, y_train_res)
            probs = model.predict_proba(X_test)[:, 1]
            
        test_probs[name] = probs
        metrics = evaluate_predictions(y_test, probs)
        
        # Financial optimization
        fin_opt = optimize_decision_threshold(y_test, probs, portfolio_type=portfolio_type)
        metrics["financial_optimization"] = fin_opt
        results[name] = metrics
        
        # Save model artifact
        m_path = os.path.join(models_dir, f"{portfolio_type}_{name.lower()}.joblib")
        joblib.dump(model, m_path)
        print(f"    ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1_score']:.4f}")
        print(f"    Optimal Cutoff: {fin_opt['optimal_threshold']:.2f} | Loss Reduction: NGN {fin_opt['monetary_savings_ngn']:,.2f} ({fin_opt['savings_percentage']:.1f}%)")

    # Ablation: Train XGBoost WITHOUT SMOTE to measure exact SMOTE uplift (RQ1)
    print(" -> Running Ablation: XGBoost WITHOUT SMOTE...")
    xgb_no_smote = xgb.XGBClassifier(n_estimators=250, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss')
    xgb_no_smote.fit(X_train_raw, y_train_raw)
    probs_no_smote = xgb_no_smote.predict_proba(X_test)[:, 1]
    metrics_no_smote = evaluate_predictions(y_test, probs_no_smote)
    fin_opt_no_smote = optimize_decision_threshold(y_test, probs_no_smote, portfolio_type=portfolio_type)
    metrics_no_smote["financial_optimization"] = fin_opt_no_smote
    results["XGBoost_No_SMOTE"] = metrics_no_smote
    test_probs["XGBoost_No_SMOTE"] = probs_no_smote
    print(f"    [No SMOTE] ROC-AUC: {metrics_no_smote['roc_auc']:.4f} | Recall: {metrics_no_smote['recall']:.4f} | PR-AUC: {metrics_no_smote['pr_auc']:.4f}")

    # 3. Save JSON Metrics
    metrics_json_path = os.path.join(results_dir, f"{portfolio_type}_metrics.json")
    with open(metrics_json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"[OK] Metrics saved to {metrics_json_path}")
    
    # 4. Generate Figures
    _plot_roc_pr_curves(y_test, test_probs, portfolio_type, results_dir)
    _plot_financial_loss_curves(y_test, test_probs, portfolio_type, results_dir)
    
    return results

def _plot_roc_pr_curves(y_test: np.ndarray, test_probs: dict, portfolio_type: str, results_dir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for name, probs in test_probs.items():
        if name == "XGBoost_No_SMOTE":
            continue
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = roc_auc_score(y_test, probs)
        ax1.plot(fpr, tpr, label=f"{name.replace('_', ' ')} (AUC = {roc_auc:.3f})", lw=2)
        
        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        ax2.plot(rec, prec, label=f"{name.replace('_', ' ')} (PR-AUC = {pr_auc:.3f})", lw=2)
        
    ax1.plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random Chance (0.50)")
    ax1.set_title(f"ROC Curves - {portfolio_type.capitalize()} Portfolio", fontsize=13, fontweight='bold')
    ax1.set_xlabel("False Positive Rate (1 - Specificity)")
    ax1.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    baseline_pr = y_test.mean()
    ax2.plot([0, 1], [baseline_pr, baseline_pr], 'k--', lw=1.5, label=f"Baseline Default Rate ({baseline_pr:.2f})")
    ax2.set_title(f"Precision-Recall Curves - {portfolio_type.capitalize()} Portfolio", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Recall (Defaults Captured)")
    ax2.set_ylabel("Precision (Accuracy of Default Predictions)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"{portfolio_type}_roc_pr_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[OK] Saved ROC/PR Curves: {plot_path}")

def _plot_financial_loss_curves(y_test: np.ndarray, test_probs: dict, portfolio_type: str, results_dir: str):
    params = COST_PARAMS[portfolio_type]
    c_fn = params["c_fn"]
    c_fp = params["c_fp"]
    thresholds = np.linspace(0.05, 0.90, 86)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name in ["Logistic_Regression", "Random_Forest", "LightGBM", "XGBoost"]:
        probs = test_probs[name]
        losses = [evaluate_financial_loss(y_test, probs, th, c_fn, c_fp)["total_financial_loss"] / 1_000_000 for th in thresholds]
        min_loss = min(losses)
        opt_th = thresholds[np.argmin(losses)]
        ax.plot(thresholds, losses, label=f"{name.replace('_', ' ')} (Min: NGN {min_loss:.1f}M @ {opt_th:.2f})", lw=2)
        
    ax.axvline(0.50, color='gray', linestyle=':', lw=1.5, label="Standard Cutoff (0.50)")
    ax.set_title(f"Expected Financial Loss vs. Decision Cutoff ({portfolio_type.capitalize()} Loan Portfolio)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Classification Cutoff Threshold (Probability of Default)")
    ax.set_ylabel("Expected Financial Loss (Million NGN)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"{portfolio_type}_cost_curve.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[OK] Saved Financial Cost Curve: {plot_path}")

def run_all_training():
    save_datasets("data")
    train_portfolio("personal", "data/personal_loans.csv")
    train_portfolio("retail", "data/retail_sme_loans.csv")

if __name__ == "__main__":
    run_all_training()
