"""
Full revised study: ONE run, ONE seed, every Chapter 4 table and figure.

    python -m src.experiment

Outputs
  results/thesis/tables/*.csv        every table (one CSV each)
  results/thesis/chapter4_tables.md  the same tables, formatted for copying
  results/thesis/figures/*.png       every figure
  results/thesis/run_info.json       seed, split, library versions
  results/{portfolio}_metrics.json   summary consumed by the dashboard/API
  models/{portfolio}_deployment.joblib  the model the API serves

Design (answers supervisor review, Section B)
  * 60/20/20 stratified train/validation/test split.
  * Four models, each trained under three imbalance strategies
    (none, class weighting, SMOTENC). SMOTENC is the main, like-for-like strategy.
  * 5-fold stratified CV on the training set (mean and SD).
  * Platt calibration and the cost-minimising cutoff are fitted on validation;
    the test set is used once for the final numbers.
  * Test-set ROC-AUC / PR-AUC with 1,000-sample bootstrap 95% CIs, DeLong tests.
  * Confusion matrix reported for every model; every metric derived from it.
"""

import json
import os
import platform
from typing import Dict

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from src import config
from src.data_generator import save_datasets
from src.evaluation import (PlattCalibrator, brier, delong_test, discrimination,
                            metrics_from_confusion)
from src.explainability import ModelExplainer, global_plots, waterfall_plot
from src.financial_cost import (THRESHOLD_GRID, bayes_cutoff, choose_cutoff, confusion_at,
                                cost_vectors, loss_curve, summarise_at_cutoffs)
from src.preprocessing import build_pipeline, feature_frame, load_portfolio, split_portfolio

OUT = "results/thesis"
TAB = os.path.join(OUT, "tables")
FIG = os.path.join(OUT, "figures")
MODEL_DIR = "models"
T = config.TARGET
NICE = {m: m.replace("_", " ") for m in config.MODEL_ORDER}
STRAT_NICE = {"none": "No resampling", "class_weight": "Class weighting", "smotenc": "SMOTENC"}

plt.rcParams.update({"font.size": 10})
_tables: Dict[str, pd.DataFrame] = {}


def save_table(name: str, df: pd.DataFrame, title: str):
    os.makedirs(TAB, exist_ok=True)
    df.to_csv(os.path.join(TAB, f"{name}.csv"), index=False)
    _tables[name] = (title, df)


def ngn_m(x):
    return round(x / 1e6, 1)


# ------------------------------------------------------------------ EDA --- #

def vif_table(df: pd.DataFrame, cols) -> Dict[str, float]:
    X = df[cols].fillna(df[cols].median()).to_numpy(float)
    out = {}
    for i, c in enumerate(cols):
        y = X[:, i]
        others = np.column_stack([np.ones(len(X)), np.delete(X, i, axis=1)])
        beta, *_ = np.linalg.lstsq(others, y, rcond=None)
        resid = y - others @ beta
        r2 = 1 - resid.var() / y.var()
        out[c] = 1 / (1 - r2) if r2 < 1 else np.inf
    return out


def eda(portfolio: str, df: pd.DataFrame):
    spec = config.PORTFOLIOS[portfolio]
    num, cat = spec["num_cols"], spec["cat_cols"]
    vifs = vif_table(df, num)
    rows = []
    for c in num:
        s = df[c]
        rows.append({"Feature": c, "Mean": round(s.mean(), 2), "Min": s.min(), "Median": s.median(),
                     "Max": s.max(), "Missing (n)": int(s.isna().sum()),
                     "Missing (%)": round(100 * s.isna().mean(), 2), "VIF": round(vifs[c], 2)})
    save_table(f"{portfolio}_eda_numeric", pd.DataFrame(rows),
               f"{spec['label']} portfolio: numeric features (n = {len(df):,}, "
               f"default rate {100 * df[T].mean():.2f}%). VIF threshold 5 (O'Brien, 2007).")

    rows = []
    for c in cat:
        g = df.groupby(c)[T].agg(["count", "sum", "mean"]).reset_index()
        for _, r in g.iterrows():
            rows.append({"Variable": c, "Category": r[c], "Loans": int(r["count"]),
                         "Defaults": int(r["sum"]), "Default rate (%)": round(100 * r["mean"], 2)})
    if portfolio == "personal":
        bands = pd.cut(df["applicant_age"], [0, 29, 44, 200], labels=["<30", "30-44", "45+"])
        g = df.groupby(bands, observed=True)[T].agg(["count", "sum", "mean"]).reset_index()
        for _, r in g.iterrows():
            rows.append({"Variable": "age_band", "Category": r["applicant_age"], "Loans": int(r["count"]),
                         "Defaults": int(r["sum"]), "Default rate (%)": round(100 * r["mean"], 2)})
    save_table(f"{portfolio}_eda_default_by_category", pd.DataFrame(rows),
               f"{spec['label']} portfolio: default rate by category")

    corr = df[num + [T]].corr()
    plt.figure(figsize=(7.5, 6))
    im = plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046)
    plt.xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title(f"Correlation matrix: {spec['label']} portfolio (Pearson)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"{portfolio}_correlation.png"), dpi=200)
    plt.close("all")


# ------------------------------------------------------- core experiment --- #

def cross_validate(portfolio, train, pos_weight):
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    X, y = feature_frame(train, portfolio), train[T].to_numpy()
    rows = []
    for strategy in config.STRATEGIES:
        for m in config.MODEL_ORDER:
            rocs, prs, brs = [], [], []
            for tr, va in skf.split(X, y):
                pipe = build_pipeline(portfolio, m, strategy, pos_weight)
                pipe.fit(X.iloc[tr], y[tr])
                p = pipe.predict_proba(X.iloc[va])[:, 1]
                rocs.append(roc_auc_score(y[va], p))
                prs.append(average_precision_score(y[va], p))
                brs.append(brier(y[va], p))
            rows.append({"Strategy": STRAT_NICE[strategy], "Model": NICE[m],
                         "ROC-AUC mean": round(np.mean(rocs), 4), "ROC-AUC SD": round(np.std(rocs, ddof=1), 4),
                         "PR-AUC mean": round(np.mean(prs), 4), "PR-AUC SD": round(np.std(prs, ddof=1), 4),
                         "Brier mean (uncalibrated)": round(np.mean(brs), 4)})
    return pd.DataFrame(rows)


def fit_and_score(portfolio, train, valid, test, pos_weight):
    """Fit every model x strategy on train; calibrate and pick cutoff on validation; score test."""
    Xtr, ytr = feature_frame(train, portfolio), train[T].to_numpy()
    Xva, yva = feature_frame(valid, portfolio), valid[T].to_numpy()
    Xte, yte = feature_frame(test, portfolio), test[T].to_numpy()
    cfn_va, cfp_va = cost_vectors(valid["loan_amount"], portfolio)
    cfn_te, cfp_te = cost_vectors(test["loan_amount"], portfolio)

    runs = {}
    for strategy in config.STRATEGIES:
        for m in config.MODEL_ORDER:
            pipe = build_pipeline(portfolio, m, strategy, pos_weight).fit(Xtr, ytr)
            raw_va = pipe.predict_proba(Xva)[:, 1]
            raw_te = pipe.predict_proba(Xte)[:, 1]
            cal = PlattCalibrator().fit(raw_va, yva)
            cal_va, cal_te = cal.transform(raw_va), cal.transform(raw_te)
            cutoff = choose_cutoff(yva, cal_va, cfn_va, cfp_va)
            runs[(strategy, m)] = dict(
                pipe=pipe, cal=cal, cutoff=cutoff, raw_va=raw_va, cal_va=cal_va,
                raw_te=raw_te, cal_te=cal_te,
                val_loss=float(loss_curve(yva, cal_va, cfn_va, cfp_va, [cutoff])[0]),
                disc=discrimination(yte, raw_te),
                brier_raw=brier(yte, raw_te), brier_cal=brier(yte, cal_te),
                cm050=confusion_at(yte, raw_te, 0.50),
                cmcut=confusion_at(yte, cal_te, cutoff),
                money=summarise_at_cutoffs(yte, cal_te, cfn_te, cfp_te, cutoff),
                loss050_raw=float(loss_curve(yte, raw_te, cfn_te, cfp_te, [0.50])[0]),
            )
    return runs


def model_rows(runs, strategy, which: str):
    rows = []
    for m in config.MODEL_ORDER:
        r = runs[(strategy, m)]
        cm = r["cm050"] if which == "050" else r["cmcut"]
        met = metrics_from_confusion(cm["TP"], cm["FP"], cm["FN"], cm["TN"])
        d = r["disc"]
        rows.append({
            "Model": NICE[m],
            "Cutoff": 0.50 if which == "050" else r["cutoff"],
            **cm,
            "Accuracy (%)": round(100 * met["accuracy"], 2),
            "Precision (%)": round(100 * met["precision"], 2),
            "Recall (%)": round(100 * met["recall"], 2),
            "Specificity (%)": round(100 * met["specificity"], 2),
            "F1": round(met["f1_score"], 4),
            "ROC-AUC [95% CI]": f"{d['roc_auc']:.3f} [{d['roc_auc_lo']:.3f}, {d['roc_auc_hi']:.3f}]",
            "PR-AUC [95% CI]": f"{d['pr_auc']:.3f} [{d['pr_auc_lo']:.3f}, {d['pr_auc_hi']:.3f}]",
            "Brier (raw)": round(r["brier_raw"], 4),
            "Brier (calibrated)": round(r["brier_cal"], 4),
        })
    return pd.DataFrame(rows)


def ablation_rows(runs):
    rows = []
    for m in ["XGBoost", "Logistic_Regression"]:
        for s in config.STRATEGIES:
            r = runs[(s, m)]
            a, b = r["cm050"], r["cmcut"]
            m50 = metrics_from_confusion(a["TP"], a["FP"], a["FN"], a["TN"])
            mc = metrics_from_confusion(b["TP"], b["FP"], b["FN"], b["TN"])
            rows.append({
                "Model": NICE[m], "Imbalance treatment": STRAT_NICE[s],
                "ROC-AUC": round(r["disc"]["roc_auc"], 4), "PR-AUC": round(r["disc"]["pr_auc"], 4),
                "Brier (raw)": round(r["brier_raw"], 4), "Brier (calibrated)": round(r["brier_cal"], 4),
                "Mean predicted PD (raw, %)": round(100 * r["raw_te"].mean(), 2),
                "Recall @0.50 (%)": round(100 * m50["recall"], 2), "Precision @0.50 (%)": round(100 * m50["precision"], 2),
                "Defaulters caught @0.50": f"{a['TP']} of {a['TP'] + a['FN']}",
                "Tuned cutoff": r["cutoff"],
                "Recall @cutoff (%)": round(100 * mc["recall"], 2), "Precision @cutoff (%)": round(100 * mc["precision"], 2),
                "Expected loss @cutoff (NGN M)": ngn_m(r["money"]["loss_at_cutoff"]),
            })
    return pd.DataFrame(rows)


def delong_rows(runs, yte):
    rows, ms = [], config.MODEL_ORDER
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            d = delong_test(yte, runs[(config.MAIN_STRATEGY, ms[i])]["raw_te"], runs[(config.MAIN_STRATEGY, ms[j])]["raw_te"])
            rows.append({"Model A": NICE[ms[i]], "Model B": NICE[ms[j]], "AUC A": round(d["auc_1"], 4),
                         "AUC B": round(d["auc_2"], 4), "Difference": round(d["diff"], 4),
                         "z": round(d["z"], 3), "p-value": round(d["p_value"], 4),
                         "Significant at 5%": "Yes" if d["p_value"] < 0.05 else "No"})
    return pd.DataFrame(rows)


def savings_rows(runs, label):
    rows = []
    for m in config.MODEL_ORDER:
        r = runs[(config.MAIN_STRATEGY, m)]
        mo = r["money"]
        rows.append({"Portfolio": label, "Model": NICE[m], "Cutoff (chosen on validation)": r["cutoff"],
                     "TP": mo["TP"], "FP": mo["FP"], "FN": mo["FN"], "TN": mo["TN"],
                     "Expected loss at 0.50, uncalibrated (NGN M)": ngn_m(r["loss050_raw"]),
                     "Expected loss at chosen cutoff (NGN M)": ngn_m(mo["loss_at_cutoff"]),
                     "Saving (NGN M)": ngn_m(r["loss050_raw"] - mo["loss_at_cutoff"])})
    return pd.DataFrame(rows)


def sensitivity_rows(runs, valid, test, portfolio, label):
    rows = []
    yva, yte = valid[T].to_numpy(), test[T].to_numpy()
    for lgd in config.LGD_SENSITIVITY:
        cfn_va, cfp_va = cost_vectors(valid["loan_amount"], portfolio, lgd)
        cfn_te, cfp_te = cost_vectors(test["loan_amount"], portfolio, lgd)
        for m in config.MODEL_ORDER:
            r = runs[(config.MAIN_STRATEGY, m)]
            cut = choose_cutoff(yva, r["cal_va"], cfn_va, cfp_va)
            l50 = float(loss_curve(yte, r["raw_te"], cfn_te, cfp_te, [0.50])[0])
            lc = float(loss_curve(yte, r["cal_te"], cfn_te, cfp_te, [cut])[0])
            rows.append({"Portfolio": label, "LGD": f"{int(lgd * 100)}%", "Model": NICE[m],
                         "Theoretical cutoff C_FP/(C_FP+C_FN)": round(bayes_cutoff(portfolio, lgd), 3),
                         "Cutoff chosen on validation": cut,
                         "Test loss at 0.50 (NGN M)": ngn_m(l50), "Test loss at cutoff (NGN M)": ngn_m(lc),
                         "Saving (NGN M)": ngn_m(l50 - lc)})
    return pd.DataFrame(rows)


def fairness_rows(run, test, portfolio, model_label):
    spec = config.PORTFOLIOS[portfolio]
    df = test.copy()
    if spec["fairness_group"] == "age_band":
        df["group"] = pd.cut(df["applicant_age"], [0, 29, 44, 200], labels=["<30", "30-44", "45+"]).astype(str)
        gname = "Age band"
    else:
        df["group"] = df["business_region"]
        gname = "Region"
    df["reject"] = run["cal_te"] >= run["cutoff"]
    rows = []
    for g, d in df.groupby("group"):
        good, bad = d[d[T] == 0], d[d[T] == 1]
        rows.append({"Model": model_label, gname: g, "Applicants": len(d), "Actual default rate (%)": round(100 * d[T].mean(), 1),
                     "Approval rate (%)": round(100 * (1 - d["reject"].mean()), 1),
                     "Good borrowers refused (FPR, %)": round(100 * good["reject"].mean(), 1) if len(good) else np.nan,
                     "Defaulters approved (FNR, %)": round(100 * (1 - bad["reject"].mean()), 1) if len(bad) else np.nan,
                     "Defaulters in group": len(bad)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- figures --- #

def fig_roc_pr(runs, yte, portfolio, label, app_copy=True):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.8))
    for m in config.MODEL_ORDER:
        r = runs[(config.MAIN_STRATEGY, m)]
        d = r["disc"]
        fpr, tpr, _ = roc_curve(yte, r["raw_te"])
        a1.plot(fpr, tpr, lw=2, label=f"{NICE[m]} (AUC {d['roc_auc']:.3f}, 95% CI {d['roc_auc_lo']:.3f} to {d['roc_auc_hi']:.3f})")
        pr, rc, _ = precision_recall_curve(yte, r["raw_te"])
        a2.plot(rc, pr, lw=2, label=f"{NICE[m]} (PR-AUC {d['pr_auc']:.3f})")
    a1.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    a1.set(xlabel="False positive rate (1 - specificity)", ylabel="True positive rate (recall)",
           title=f"ROC curves, test set: {label} portfolio")
    a1.legend(loc="lower right", fontsize=8)
    base = yte.mean()
    a2.axhline(base, color="k", ls="--", lw=1, label=f"Default rate in test set ({base:.3f})")
    a2.set(xlabel="Recall (defaulters caught)", ylabel="Precision", title=f"Precision-recall curves, test set: {label} portfolio")
    a2.legend(loc="upper right", fontsize=8)
    for a in (a1, a2):
        a.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(FIG, f"{portfolio}_roc_pr_curves.png")
    plt.savefig(p, dpi=200)
    if app_copy:
        plt.savefig(f"results/{portfolio}_roc_pr_curves.png", dpi=150)
    plt.close("all")


def fig_cost(runs, test, portfolio, label):
    yte = test[T].to_numpy()
    cfn, cfp = cost_vectors(test["loan_amount"], portfolio)
    plt.figure(figsize=(10, 6))
    for m in config.MODEL_ORDER:
        r = runs[(config.MAIN_STRATEGY, m)]
        losses = loss_curve(yte, r["cal_te"], cfn, cfp) / 1e6
        line, = plt.plot(THRESHOLD_GRID, losses, lw=2,
                         label=f"{NICE[m]} (cutoff chosen on validation {r['cutoff']:.2f}: NGN {r['money']['loss_at_cutoff'] / 1e6:,.0f}M)")
        plt.scatter([r["cutoff"]], [r["money"]["loss_at_cutoff"] / 1e6], color=line.get_color(), zorder=5)
    plt.axvline(bayes_cutoff(portfolio), color="gray", ls=":", label=f"Theoretical cutoff {bayes_cutoff(portfolio):.3f}")
    plt.axvline(0.5, color="black", ls="--", lw=1, label="Conventional 0.50 cutoff")
    plt.xlabel("Cutoff on calibrated probability of default")
    plt.ylabel("Expected loss on test set (NGN million)")
    plt.title(f"Expected financial loss vs cutoff (calibrated, SMOTENC models): {label} portfolio")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"{portfolio}_cost_curve.png"), dpi=200)
    plt.savefig(f"results/{portfolio}_cost_curve.png", dpi=150)
    plt.close("all")


def fig_calibration(runs, yte, portfolio, label):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, key, title in [(axes[0], "raw_te", "Before calibration (SMOTENC-trained)"),
                           (axes[1], "cal_te", "After Platt calibration on validation set")]:
        for m in config.MODEL_ORDER:
            r = runs[(config.MAIN_STRATEGY, m)]
            fp, mp = calibration_curve(yte, r[key], n_bins=10, strategy="quantile")
            b = r["brier_raw"] if key == "raw_te" else r["brier_cal"]
            ax.plot(mp, fp, marker="o", lw=1.5, label=f"{NICE[m]} (Brier {b:.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
        ax.set(xlabel="Mean predicted probability of default", title=title, xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Observed default rate")
    fig.suptitle(f"Calibration (reliability) curves, test set: {label} portfolio")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"{portfolio}_calibration.png"), dpi=200)
    plt.close("all")


# ----------------------------------------------------------------- main --- #

def run_portfolio(portfolio: str) -> Dict:
    spec = config.PORTFOLIOS[portfolio]
    label = spec["label"]
    print(f"\n=== {label} portfolio ===")
    df = load_portfolio(portfolio)
    eda(portfolio, df)
    train, valid, test = split_portfolio(df)
    pos_weight = float((train[T] == 0).sum() / (train[T] == 1).sum())

    # Partition table, including the SMOTENC-balanced training set.
    smote_pipe = build_pipeline(portfolio, "Logistic_Regression", "smotenc")
    Xi = smote_pipe.named_steps["impute"].fit_transform(feature_frame(train, portfolio))
    _, y_res = smote_pipe.named_steps["smote"].fit_resample(Xi, train[T].to_numpy())
    part = []
    for name, d in [("Training (original)", train[T]), ("Training after SMOTENC", pd.Series(y_res)),
                    ("Validation", valid[T]), ("Test", test[T])]:
        part.append({"Portfolio": label, "Set": name, "Performing": int((d == 0).sum()),
                     "Defaulted": int((d == 1).sum()), "Total": int(len(d)),
                     "Default rate (%)": round(100 * d.mean(), 2)})
    part = pd.DataFrame(part)

    print(" cross-validation ...")
    cv = cross_validate(portfolio, train, pos_weight)
    cv.insert(0, "Portfolio", label)
    print(" fitting, calibrating, scoring ...")
    runs = fit_and_score(portfolio, train, valid, test, pos_weight)
    yte = test[T].to_numpy()

    main_cut = model_rows(runs, config.MAIN_STRATEGY, "cut")
    main_050 = model_rows(runs, config.MAIN_STRATEGY, "050")
    save_table(f"{portfolio}_model_scores_at_cutoff", main_cut,
               f"{label} portfolio: test-set results, SMOTENC-trained models, Platt-calibrated, cutoff chosen on validation set")
    save_table(f"{portfolio}_model_scores_at_050", main_050,
               f"{label} portfolio: test-set results at the conventional 0.50 cutoff (uncalibrated SMOTENC models)")
    full = []
    for s in config.STRATEGIES:
        t = model_rows(runs, s, "cut")
        t.insert(0, "Strategy", STRAT_NICE[s])
        full.append(t)
    save_table(f"{portfolio}_all_strategies_at_cutoff", pd.concat(full),
               f"{label} portfolio: all models x imbalance strategies (test set, calibrated, validation cutoff)")
    save_table(f"{portfolio}_imbalance_ablation", ablation_rows(runs),
               f"{label} portfolio: effect of imbalance treatment (test set)")
    save_table(f"{portfolio}_delong", delong_rows(runs, yte),
               f"{label} portfolio: DeLong tests for paired test-set ROC-AUC differences (SMOTENC models)")

    fig_roc_pr(runs, yte, portfolio, label)
    fig_cost(runs, test, portfolio, label)
    fig_calibration(runs, yte, portfolio, label)

    # Model served by the API: lowest expected loss on the VALIDATION set.
    chosen = min(config.MODEL_ORDER, key=lambda m: runs[(config.MAIN_STRATEGY, m)]["val_loss"])
    cr = runs[(config.MAIN_STRATEGY, chosen)]
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"portfolio": portfolio, "model_name": chosen, "pipeline": cr["pipe"], "calibrator": cr["cal"],
                 "cutoff": cr["cutoff"], "review_band": config.REVIEW_BAND,
                 "background": train.sample(n=config.SHAP_BACKGROUND_ROWS, random_state=config.RANDOM_SEED)},
                os.path.join(MODEL_DIR, f"{portfolio}_deployment.joblib"), compress=3)

    # Fairness: deployed model and XGBoost.
    fair = pd.concat([fairness_rows(cr, test, portfolio, NICE[chosen] + " (deployed)"),
                      fairness_rows(runs[(config.MAIN_STRATEGY, "XGBoost")], test, portfolio, "XGBoost")]
                     if chosen != "XGBoost" else [fairness_rows(cr, test, portfolio, "XGBoost (deployed)")])
    save_table(f"{portfolio}_fairness", fair, f"{label} portfolio: approval and error rates by group (test set)")

    # SHAP for XGBoost (RQ3) and, if different, for the deployed model.
    print(" SHAP ...")
    shap_out, shap_grouped = {}, {}
    for m in dict.fromkeys(["XGBoost", chosen]):
        r = runs[(config.MAIN_STRATEGY, m)]
        expl = ModelExplainer(r["pipe"], m, portfolio, train)
        g = global_plots(expl, test, FIG, f"{portfolio}_{m.lower()}", label)
        g["ranking"].insert(0, "Model", NICE[m])
        shap_out[m] = g["ranking"]
        g["grouped"].insert(0, "Model", NICE[m])
        shap_grouped[m] = g["grouped"]
        if m == chosen:  # images used by the dashboard
            for k, dst in [("beeswarm", f"results/{portfolio}_shap_beeswarm.png"), ("bar", f"results/{portfolio}_shap_summary.png")]:
                import shutil
                shutil.copyfile(g[k], dst)
    xr = runs[(config.MAIN_STRATEGY, "XGBoost")]
    rejected = np.where((xr["cal_te"] >= xr["cutoff"]) & (yte == 1))[0]
    pick = rejected[np.argsort(xr["cal_te"][rejected])[len(rejected) // 2]]
    applicant = test.iloc[[pick]]
    waterfall_plot(ModelExplainer(xr["pipe"], "XGBoost", portfolio, train), applicant,
                   os.path.join(FIG, f"{portfolio}_xgboost_shap_waterfall.png"),
                   f"SHAP waterfall (log-odds), XGBoost: one rejected {label} applicant (test row {pick})\n"
                   f"calibrated PD {xr['cal_te'][pick]:.3f} vs cutoff {xr['cutoff']:.2f}")
    app_tbl = applicant[spec["num_cols"] + spec["cat_cols"] + [T]].T.reset_index()
    app_tbl.columns = ["Field", "Value"]
    save_table(f"{portfolio}_waterfall_applicant", app_tbl, f"{label} portfolio: applicant shown in the SHAP waterfall plot")
    save_table(f"{portfolio}_shap_ranking_by_variable", pd.concat(shap_grouped.values()),
               f"{label} portfolio: variables ranked by mean |SHAP| (one-hot columns summed per variable), test set")
    save_table(f"{portfolio}_shap_ranking", pd.concat(shap_out.values()),
               f"{label} portfolio: features ranked by mean |SHAP| on the test set")

    # SME only: with vs without interest rate (endogeneity check).
    extra = {}
    if portfolio == "retail":
        key = "retail_no_rate"
        config.PORTFOLIOS[key] = {**spec, "num_cols": [c for c in spec["num_cols"] if c != "interest_rate"]}
        rows = []
        for m in ["Logistic_Regression", "XGBoost"]:
            for variant, pkey in [("With interest rate", portfolio), ("Without interest rate", key)]:
                pipe = build_pipeline(pkey, m, config.MAIN_STRATEGY).fit(feature_frame(train, pkey), train[T])
                p = pipe.predict_proba(feature_frame(test, pkey))[:, 1]
                d = discrimination(yte, p)
                rows.append({"Model": NICE[m], "Variant": variant,
                             "ROC-AUC [95% CI]": f"{d['roc_auc']:.3f} [{d['roc_auc_lo']:.3f}, {d['roc_auc_hi']:.3f}]",
                             "PR-AUC": round(d["pr_auc"], 4)})
        save_table("retail_interest_rate_ablation", pd.DataFrame(rows),
                   "SME portfolio: models fitted with and without interest rate (test set)")
        del config.PORTFOLIOS[key]

    # Dashboard/API summary (same structure the existing front end expects).
    summary = {}
    for m in config.MODEL_ORDER:
        r = runs[(config.MAIN_STRATEGY, m)]
        cm = r["cmcut"]
        met = metrics_from_confusion(cm["TP"], cm["FP"], cm["FN"], cm["TN"])
        summary[m] = {**met, "roc_auc": r["disc"]["roc_auc"], "pr_auc": r["disc"]["pr_auc"],
                      "brier_calibrated": r["brier_cal"], "tp": cm["TP"], "fp": cm["FP"], "fn": cm["FN"], "tn": cm["TN"],
                      "financial_optimization": {
                          "optimal_threshold": r["cutoff"],
                          "min_financial_loss": r["money"]["loss_at_cutoff"],
                          "default_50_loss": r["loss050_raw"],
                          "monetary_savings_ngn": r["loss050_raw"] - r["money"]["loss_at_cutoff"]}}
    with open(f"results/{portfolio}_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    return {"partitions": part, "cv": cv, "savings": savings_rows(runs, label),
            "sensitivity": sensitivity_rows(runs, valid, test, portfolio, label),
            "chosen": chosen, "chosen_cutoff": cr["cutoff"]}


def cost_table():
    rows = []
    for p, a in config.COST_ASSUMPTIONS.items():
        rows.append({"Portfolio": config.PORTFOLIOS[p]["label"], "EAD": "Loan amount requested (full drawdown assumed)",
                     "LGD (base)": f"{int(a['lgd'] * 100)}%", "LGD sensitivity": ", ".join(f"{int(x * 100)}%" for x in config.LGD_SENSITIVITY),
                     "Net interest margin forgone": f"{int(a['net_margin'] * 100)}% p.a.", "Tenor": f"{a['tenor_years']:g} year",
                     "C_FN": "Loan amount x LGD", "C_FP": "Loan amount x margin x tenor",
                     "Theoretical cutoff C_FP/(C_FP+C_FN)": round(bayes_cutoff(p), 3)})
    save_table("cost_assumptions", pd.DataFrame(rows),
               "Cost assumptions (ASSUMED values; replace with sourced figures before relying on Naira totals)")


def model_settings_table():
    rows = []
    for m, prm in config.MODEL_PARAMS.items():
        rows.append({"Model": NICE[m], "Hyperparameters (fixed, not tuned)": ", ".join(f"{k}={v}" for k, v in prm.items()),
                     "Feature scaling": "StandardScaler" if m == "Logistic_Regression" else "None (raw units)",
                     "Imbalance treatment (main)": f"SMOTENC (k={config.SMOTE_K_NEIGHBORS}) on training data only",
                     "Also compared": "No resampling; class weighting"})
    save_table("model_settings", pd.DataFrame(rows), "Model settings (random seed 42 throughout)")


def write_markdown():
    lines = ["# Chapter 4 tables (generated by `python -m src.experiment`)", "",
             "All values come from a single run with random seed 42. Data are SIMULATED (src/data_generator.py).", ""]
    for name, (title, df) in _tables.items():
        lines += [f"## {title}", f"`tables/{name}.csv`", "", df.to_markdown(index=False), ""]
    with open(os.path.join(OUT, "chapter4_tables.md"), "w") as f:
        f.write("\n".join(lines))


def run_info():
    import imblearn, lightgbm, matplotlib as mpl, scipy, shap, sklearn, xgboost
    info = {"random_seed": config.RANDOM_SEED, "split": "60/20/20 stratified train/validation/test",
            "cv_folds": config.CV_FOLDS, "bootstrap_resamples": config.N_BOOTSTRAP,
            "smote": f"SMOTENC k={config.SMOTE_K_NEIGHBORS}", "data": "SIMULATED (src/data_generator.py, seed 42 / 52, 6,000 rows each)",
            "python": platform.python_version(), "platform": platform.platform(),
            "versions": {m.__name__: m.__version__ for m in [np, pd, sklearn, imblearn, xgboost, lightgbm, shap, scipy, mpl]}}
    with open(os.path.join(OUT, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    return info


def main(regenerate_data: bool = False):
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    if regenerate_data or not all(os.path.exists(s["csv"]) for s in config.PORTFOLIOS.values()):
        save_datasets("data")
    model_settings_table()
    cost_table()
    res = {p: run_portfolio(p) for p in list(config.PORTFOLIOS)}
    save_table("partitions", pd.concat([r["partitions"] for r in res.values()]),
               "Dataset partitions, before and after SMOTENC (SMOTENC applied to the training set only)")
    save_table("cross_validation", pd.concat([r["cv"] for r in res.values()]),
               "5-fold stratified cross-validation on the training set (mean and SD)")
    save_table("financial_savings", pd.concat([r["savings"] for r in res.values()]),
               "Financial effect of the cost-based cutoff (test set; per-loan costs; cutoff chosen on validation set)")
    save_table("lgd_sensitivity", pd.concat([r["sensitivity"] for r in res.values()]),
               "Sensitivity of cutoff and expected loss to LGD (test set)")
    deployed = pd.DataFrame([{"Portfolio": config.PORTFOLIOS[p]["label"], "Model served by API": NICE[r["chosen"]],
                              "Selection rule": "Lowest expected loss on validation set (SMOTENC, calibrated)",
                              "Cutoff": r["chosen_cutoff"], "Manual-review band starts at": round(r["chosen_cutoff"] * config.REVIEW_BAND, 3)}
                             for p, r in res.items()])
    save_table("deployed_models", deployed, "Models deployed in the API and dashboard")
    info = run_info()
    write_markdown()
    print("\nDone. Tables:", TAB, " Figures:", FIG)
    print(json.dumps(info["versions"], indent=1))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate-data", action="store_true")
    main(ap.parse_args().regenerate_data)
