"""
Task 3: Predictive Customer Churn Classifier (Random Forest & XGBoost)
======================================================================
Objectives:
1. Load the Telco Customer Churn dataset using Pandas.
2. Convert TotalCharges to numeric and handle missing/null records.
3. Preprocess categorical features with OneHotEncoder and balance target
   classes using SMOTE (or scale_pos_weight / class_weight).
4. Train RandomForestClassifier and XGBClassifier with 5-Fold Stratified
   Cross-Validation.
5. Evaluate and plot ROC-AUC curves (>80% target), Confusion Matrices,
   and Top 10 Most Influential Feature Importances.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Use imbalanced-learn SMOTE and xgboost XGBClassifier if installed, with automatic fallback
try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def fallback_smote(X_data, y_data, k=5, random_state=42):
    """Exact k-NN SMOTE oversampling fallback when imbalanced-learn is not installed."""
    rng = np.random.RandomState(random_state)
    y_arr = np.asarray(y_data)
    X_maj = X_data[y_arr == 0]
    X_min = X_data[y_arr == 1]
    n_syn = len(X_maj) - len(X_min)

    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
    _, indices = nn.kneighbors(X_min)

    syn_samples = []
    for _ in range(n_syn):
        idx = rng.randint(0, len(X_min))
        neighbor_idx = indices[idx, rng.randint(1, k + 1)]
        diff = X_min[neighbor_idx] - X_min[idx]
        gap = rng.uniform(0, 1)
        syn_samples.append(X_min[idx] + gap * diff)

    X_syn = np.vstack(syn_samples)
    return np.vstack([X_data, X_syn]), np.concatenate([y_arr, np.ones(n_syn, dtype=int)])


def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    sns.set_theme(style="whitegrid")

    # =========================================================================
    # STEP 1: Load the Telco Customer Churn Dataset
    # =========================================================================
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    local_csv = os.path.join("data", "Telco-Customer-Churn.csv")

    try:
        df = pd.read_csv(url)
        df.to_csv(local_csv, index=False)
        print(f"[INFO] Loaded Telco dataset from URL and cached to '{local_csv}'.")
    except Exception:
        df = pd.read_csv(local_csv)
        print(f"[INFO] Loaded Telco dataset from local cache '{local_csv}'.")

    print(f"Dataset Shape: {df.shape}")
    print("\nTarget Churn Distribution:")
    print(df["Churn"].value_counts())

    # =========================================================================
    # STEP 2: Convert TotalCharges to Numeric & Handle Nulls
    # =========================================================================
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    null_count = df["TotalCharges"].isnull().sum()
    print(f"\n[INFO] Found {null_count} null values in 'TotalCharges' after numeric conversion.")

    # Impute missing TotalCharges with median and drop non-predictive customerID
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Separate features and binary target (Yes -> 1, No -> 0)
    X = df.drop(columns=["Churn"])
    y = (df["Churn"] == "Yes").astype(int)

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = [c for c in X.columns if c not in num_cols]

    # 80/20 Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Preprocess Categorical Features with OneHotEncoder and Scale Numeric Features
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols),
        ]
    )

    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)
    feature_names = num_cols + list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)
    )

    # =========================================================================
    # STEP 3: Balance Target Classes using SMOTE & Compute scale_pos_weight
    # =========================================================================
    scale_pos_weight = round((y_train == 0).sum() / (y_train == 1).sum(), 2)
    if HAS_IMBLEARN:
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_enc, y_train)
    else:
        X_train_smote, y_train_smote = fallback_smote(X_train_enc, y_train, k=5, random_state=42)

    print(f"Before SMOTE: Class 0 = {(y_train == 0).sum()}, Class 1 = {(y_train == 1).sum()} (scale_pos_weight = {scale_pos_weight})")
    print(f"After SMOTE:  Class 0 = {(y_train_smote == 0).sum()}, Class 1 = {(y_train_smote == 1).sum()}")

    # =========================================================================
    # STEP 4: Train RandomForestClassifier and XGBClassifier with Cross-Validation
    # =========================================================================
    rf_model = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    if HAS_XGBOOST:
        xgb_model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=1.0,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    else:
        xgb_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.85,
            random_state=42,
        )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "Random Forest": rf_model,
        "XGBoost (XGBClassifier)": xgb_model,
    }

    benchmark_rows = []
    roc_data = {}
    cm_data = {}
    importances_data = {}

    for name, model in models.items():
        cv_scores = cross_val_score(
            model, X_train_smote, y_train_smote, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        model.fit(X_train_smote, y_train_smote)

        y_prob = model.predict_proba(X_test_enc)[:, 1]
        y_pred = model.predict(X_test_enc)

        test_auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, test_auc)
        cm_data[name] = confusion_matrix(y_test, y_pred)
        importances_data[name] = model.feature_importances_

        benchmark_rows.append(
            {
                "Model": name,
                "5-Fold CV ROC-AUC": round(cv_scores.mean(), 4),
                "Test ROC-AUC": round(test_auc, 4),
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "Precision": round(precision_score(y_test, y_pred), 4),
                "Recall": round(recall_score(y_test, y_pred), 4),
                "F1-Score": round(f1_score(y_test, y_pred), 4),
            }
        )

        print(f"\n--- {name} Classification Report ---")
        print(classification_report(y_test, y_pred, target_names=["Retained (0)", "Churned (1)"]))

    benchmark_df = pd.DataFrame(benchmark_rows)
    benchmark_df.to_csv("outputs/churn_benchmark_metrics.csv", index=False)

    print("\n==========================================================================")
    print("         TELCO CUSTOMER CHURN BENCHMARK RESULTS (80/20 TEST SPLIT)        ")
    print("==========================================================================")
    print(benchmark_df.to_string(index=False))

    # =========================================================================
    # STEP 5A: Plot ROC-AUC Curves (>80% Target)
    # =========================================================================
    plt.figure(figsize=(9, 6.5))
    colors = {"Random Forest": "#1f77b4", "XGBoost (XGBClassifier)": "#d95f02"}

    for name, (fpr, tpr, auc_val) in roc_data.items():
        plt.plot(
            fpr,
            tpr,
            linewidth=2.6,
            color=colors[name],
            label=f"{name} (Test ROC-AUC = {auc_val:.4f} / {auc_val*100:.2f}%)",
        )

    plt.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random Baseline (AUC = 0.5000)")
    plt.axhline(0.80, color="#2ca02c", linestyle=":", linewidth=1.5, label="Target Benchmark Threshold (80% TPR)")
    plt.fill_between(
        roc_data["XGBoost (XGBClassifier)"][0],
        roc_data["XGBoost (XGBClassifier)"][1],
        alpha=0.10,
        color="#d95f02",
    )
    plt.title("Telco Customer Churn: ROC-AUC Curve Comparison (>80% Target)", fontsize=14, fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    plt.legend(loc="lower right", fontsize=10.5, frameon=True)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.02])
    plt.tight_layout()
    plt.savefig("outputs/1_roc_auc_curves.png", dpi=300, bbox_inches="tight")
    plt.show()

    # =========================================================================
    # STEP 5B: Plot Confusion Matrices
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    class_labels = ["Retained (No)", "Churned (Yes)"]

    for ax, (name, cm) in zip(axes, cm_data.items()):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=ax,
            annot_kws={"size": 13, "weight": "bold"},
        )
        ax.set_title(f"{name}\nConfusion Matrix (Test Split)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted Churn Status", fontsize=11)
        ax.set_ylabel("Actual Churn Status", fontsize=11)

    plt.tight_layout()
    plt.savefig("outputs/2_confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.show()

    # =========================================================================
    # STEP 5C: Plot Top 10 Most Influential Feature Importances
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    palettes = ["Blues_r", "Oranges_r"]

    for ax, (name, imps), pal in zip(axes, importances_data.items(), palettes):
        imp_df = pd.DataFrame({"Feature": feature_names, "Importance": imps})
        imp_top10 = imp_df.sort_values(by="Importance", ascending=False).head(10)

        sns.barplot(data=imp_top10, x="Importance", y="Feature", palette=pal, ax=ax)
        ax.set_title(f"{name}\nTop 10 Most Influential Features", fontsize=13, fontweight="bold")
        ax.set_xlabel("Relative Feature Importance Score", fontsize=11)
        ax.set_ylabel("Feature Name", fontsize=11)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)
        ax.set_xlim(0, imp_top10["Importance"].max() * 1.15)

    plt.tight_layout()
    plt.savefig("outputs/3_top10_feature_importances.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("\n[SUCCESS] All evaluation metrics and plots saved to 'outputs/'.")


if __name__ == "__main__":
    main()
