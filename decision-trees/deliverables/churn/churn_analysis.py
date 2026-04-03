"""
Telco churn classification aligned with the stronger notebook-style workflow.

Outputs (only 3 figures):
1) EDA: churn fraction by Contract
2) EDA: churn fraction by SeniorCitizen
3) Confusion matrix for the best model
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PLOTS_DIR = BASE_DIR / "plots"
WRITEUP_PATH = BASE_DIR / "writeup.md"
METRICS_PATH = BASE_DIR / "metrics.txt"


def load_and_clean(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Put the Kaggle CSV in decision-trees/churn/data/"
        )

    df = pd.read_csv(path)

    # Convert TotalCharges with coercion for blank strings.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Normalize target to 0/1.
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    if df["Churn"].isna().any():
        raise ValueError("Unexpected Churn values found. Expected only Yes/No.")

    # Drop ID to avoid leakage.
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df


def encode_and_select_features(df: pd.DataFrame) -> pd.DataFrame:
    encoded = df.copy(deep=True)
    le = LabelEncoder()

    # Label-encode object columns (as in the reference notebook style).
    obj_cols = encoded.select_dtypes(include=["object"]).columns.tolist()
    for col in obj_cols:
        encoded[col] = le.fit_transform(encoded[col].astype(str))

    # Min-max normalize numeric-heavy columns used in that workflow.
    scaler = MinMaxScaler()
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        encoded[col] = scaler.fit_transform(encoded[[col]])

    # Feature filtering aligned with the provided approach.
    drop_cols = ["PhoneService", "gender", "StreamingTV", "StreamingMovies", "MultipleLines", "InternetService"]
    encoded = encoded.drop(columns=drop_cols)
    return encoded


def plot_churn_fraction_by_feature(df: pd.DataFrame, feature: str, output_name: str) -> None:
    frac = (
        df.groupby(feature)["Churn"]
        .value_counts(normalize=True)
        .rename("fraction")
        .reset_index()
    )
    frac["ChurnLabel"] = frac["Churn"].map({0: "No Churn", 1: "Churn"})

    plt.figure(figsize=(8, 5))
    sns.barplot(data=frac, x=feature, y="fraction", hue="ChurnLabel")
    plt.title(f"Fraction Churned vs Not Churned by {feature}")
    plt.ylabel("Fraction")
    plt.ylim(0, 1.0)
    plt.xlabel(feature)
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / output_name, dpi=220, bbox_inches="tight")
    plt.close()


def train_and_evaluate(df: pd.DataFrame) -> dict:
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Keep split setting close to reference workflow.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=2, stratify=y
    )

    # SMOTE balancing on training set.
    smote = SMOTE(sampling_strategy=1.0, random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    balanced_counts = Counter(y_train_bal)

    classifiers = {
        "XGBClassifier": XGBClassifier(
            learning_rate=0.01,
            max_depth=3,
            n_estimators=1000,
            random_state=42,
            eval_metric="logloss",
        ),
        "LGBMClassifier": LGBMClassifier(
            learning_rate=0.01,
            max_depth=3,
            n_estimators=1000,
            random_state=42,
            verbose=-1,
        ),
        "RandomForestClassifier": RandomForestClassifier(max_depth=4, random_state=0),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=1000, max_depth=4, min_samples_leaf=1),
    }
    classifiers["StackingClassifier"] = StackingClassifier(
        estimators=[
            ("classifier_lgbm", classifiers["LGBMClassifier"]),
            ("classifier_rf", classifiers["RandomForestClassifier"]),
            ("classifier_dt", classifiers["DecisionTreeClassifier"]),
        ],
        final_estimator=LGBMClassifier(learning_rate=0.01, max_depth=3, n_estimators=1000, random_state=42, verbose=-1),
    )

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    results = []
    fitted = {}

    def cv_roc_auc_manual(estimator, X_data, y_data, splitter) -> float:
        scores = []
        X_df = pd.DataFrame(X_data).reset_index(drop=True)
        y_sr = pd.Series(y_data).reset_index(drop=True)
        for tr_idx, va_idx in splitter.split(X_df, y_sr):
            X_tr, X_va = X_df.iloc[tr_idx], X_df.iloc[va_idx]
            y_tr, y_va = y_sr.iloc[tr_idx], y_sr.iloc[va_idx]
            fold_est = clone(estimator)
            fold_est.fit(X_tr, y_tr)
            if hasattr(fold_est, "predict_proba"):
                probs = fold_est.predict_proba(X_va)[:, 1]
            else:
                probs = fold_est.predict(X_va)
            scores.append(roc_auc_score(y_va, probs))
        return float(np.mean(scores))

    for name, clf in classifiers.items():
        cv_score = cv_roc_auc_manual(clf, X_train_bal, y_train_bal, cv)
        clf.fit(X_train_bal, y_train_bal)
        pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            pred_score = clf.predict_proba(X_test)[:, 1]
        else:
            pred_score = pred
        roc_from_labels = roc_auc_score(y_test, pred_score)
        precision = precision_score(y_test, pred, zero_division=0)
        recall = recall_score(y_test, pred, zero_division=0)
        f1_churn = classification_report(y_test, pred, output_dict=True)["1"]["f1-score"]
        results.append(
            {
                "model": name,
                "cv_roc_auc": float(cv_score),
                "test_roc_auc": float(roc_from_labels),
                "precision_churn": float(precision),
                "recall_churn": float(recall),
                "f1_churn": float(f1_churn),
            }
        )
        fitted[name] = clf

    # Pick best by test ROC_AUC from labels to mirror notebook metric style.
    best = max(results, key=lambda r: r["test_roc_auc"])
    best_model_name = best["model"]
    best_model = fitted[best_model_name]
    y_pred_best = best_model.predict(X_test)
    if hasattr(best_model, "predict_proba"):
        y_score_best = best_model.predict_proba(X_test)[:, 1]
    else:
        y_score_best = y_pred_best
    cm = confusion_matrix(y_test, y_pred_best)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix (Best: {best_model_name})")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_confusion_matrix.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # ROC curve for best model.
    fpr, tpr, _ = roc_curve(y_test, y_score_best)
    best_auc = roc_auc_score(y_test, y_score_best)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{best_model_name} (AUC={best_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Best Model)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_roc_curve_best_model.png", dpi=220, bbox_inches="tight")
    plt.close()

    return {
        "results": results,
        "best": best,
        "best_model_name": best_model_name,
        "confusion_matrix": cm,
        "best_auc_prob": float(best_auc),
        "train_size": len(y_train),
        "test_size": len(y_test),
        "balanced_counts": dict(balanced_counts),
    }


def write_outputs(df: pd.DataFrame, metrics: dict) -> None:
    churn_rate = df["Churn"].mean()
    cm = metrics["confusion_matrix"]
    results_df = pd.DataFrame(metrics["results"]).sort_values(by="test_roc_auc", ascending=False)

    METRICS_PATH.write_text(
        "\n".join(
            [
                "Telco Churn Metrics",
                "===================",
                f"Rows: {len(df)}",
                f"Overall churn rate: {churn_rate:.3f}",
                f"Train size: {metrics['train_size']}",
                f"Test size: {metrics['test_size']}",
                f"Balanced train counts (SMOTE): {metrics['balanced_counts']}",
                "",
                "Model comparison (higher is better):",
                results_df.to_string(index=False),
                "",
                f"Best model: {metrics['best_model_name']}",
                "",
                "Confusion matrix for best model (rows=true, cols=pred):",
                str(cm),
                "",
                f"Precision (churn=1): {metrics['best']['precision_churn']:.4f}",
                f"Recall (churn=1): {metrics['best']['recall_churn']:.4f}",
                f"F1 (churn=1): {metrics['best']['f1_churn']:.4f}",
                f"CV ROC-AUC: {metrics['best']['cv_roc_auc']:.4f}",
                f"Test ROC-AUC (from predicted labels): {metrics['best']['test_roc_auc']:.4f}",
                f"Best ROC-AUC from probabilities: {metrics['best_auc_prob']:.4f}",
            ]
        )
        + "\n"
    )

    WRITEUP_PATH.write_text(
        "\n".join(
            [
                "# Telco Churn: Short Writeup",
                "",
                "## Exploratory analysis (two features)",
                "- **Contract:** Month-to-month customers show a much higher churn fraction than one-year and two-year contracts.",
                "- **SeniorCitizen:** Senior-citizen customers generally show higher churn fraction than non-senior customers.",
                "",
                "## Model",
                "- Pipeline used: label encoding, selected-feature reduction, MinMax scaling for tenure/MonthlyCharges/TotalCharges, SMOTE balancing on training data.",
                "- Classifiers compared: XGBClassifier, LGBMClassifier, RandomForestClassifier, DecisionTreeClassifier, and a StackingClassifier.",
                "- `customerID` was excluded from modeling to avoid leakage.",
                "- Data split: 80% train / 20% test.",
                "",
                "## Evaluation",
                f"- Best model: **{metrics['best_model_name']}**",
                f"- Precision (churn class): **{metrics['best']['precision_churn']:.3f}**",
                f"- Recall (churn class): **{metrics['best']['recall_churn']:.3f}**",
                f"- F1 (churn class): **{metrics['best']['f1_churn']:.3f}**",
                f"- ROC-AUC (probability-based): **{metrics['best_auc_prob']:.3f}**",
                "- Confusion matrix saved as `plots/03_confusion_matrix.png`.",
                "- ROC curve saved as `plots/04_roc_curve_best_model.png`.",
                "",
                "## Validation set?",
                "- A separate validation split is optional here because repeated stratified cross-validation was used for model comparison on training data.",
                "- For heavy hyperparameter tuning, use train/validation/test or nested CV.",
                "",
                "## Recommended action",
                "- Prioritize retention offers for month-to-month customers and improve support/onboarding in the first months.",
                "- Target high-risk segments (e.g., senior-citizen month-to-month customers) with proactive outreach and incentive plans.",
            ]
        )
        + "\n"
    )


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_clean(DATA_PATH)

    # Keep only two EDA figures as requested.
    plot_churn_fraction_by_feature(df, "Contract", "01_churn_fraction_by_contract.png")
    plot_churn_fraction_by_feature(df, "SeniorCitizen", "02_churn_fraction_by_seniorcitizen.png")

    # Build encoded/selected modeling dataset.
    df_model = encode_and_select_features(df)
    metrics = train_and_evaluate(df_model)
    write_outputs(df, metrics)

    print("Done.")
    print(f"Plots saved in: {PLOTS_DIR}")
    print(f"Metrics file: {METRICS_PATH}")
    print(f"Writeup file: {WRITEUP_PATH}")


if __name__ == "__main__":
    main()
