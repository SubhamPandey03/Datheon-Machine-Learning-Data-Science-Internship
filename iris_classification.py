"""
Supervised Classification Benchmark on the Iris Dataset
=======================================================
Task Objectives:
1. Load the benchmark Iris dataset using Pandas.
2. Perform exploratory pairplots with Seaborn.
3. Handle train/test splitting (80/20) and feature scaling (StandardScaler).
4. Train Logistic Regression and Decision Tree models and benchmark
   Accuracy, Precision, Recall, and F1-Score.
5. Plot Confusion Matrix heatmaps and 2D Decision Boundary visualizations.
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def main():
    # Ensure output and data directories exist
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    sns.set_theme(style="whitegrid")

    # =========================================================================
    # STEP 1: Load the Benchmark Dataset using Pandas
    # =========================================================================
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    local_csv = os.path.join("data", "iris.csv")

    try:
        df = pd.read_csv(url)
        df.to_csv(local_csv, index=False)
        print(f"[INFO] Loaded dataset from remote URL and cached to '{local_csv}'.")
    except Exception:
        df = pd.read_csv(local_csv)
        print(f"[INFO] Loaded dataset from local cache '{local_csv}'.")

    print(f"Dataset Shape: {df.shape}")
    print("\nFirst 5 Rows:")
    print(df.head())

    # =========================================================================
    # STEP 2: Perform Exploratory Pairplots with Seaborn
    # =========================================================================
    print("\n[INFO] Generating Exploratory Pairplot...")
    pair_grid = sns.pairplot(
        df, hue="species", markers=["o", "s", "D"], palette="Set2", height=2.3
    )
    pair_grid.fig.suptitle(
        "Exploratory Pairplot of Iris Dataset", y=1.02, fontsize=14, fontweight="bold"
    )
    plt.savefig("outputs/1_iris_pairplot.png", dpi=300, bbox_inches="tight")
    plt.show()

    # =========================================================================
    # STEP 3: Train/Test Split (80/20) & Feature Normalization (StandardScaler)
    # =========================================================================
    X = df.drop("species", axis=1)
    y = df["species"]

    # Encode categorical target species ('setosa', 'versicolor', 'virginica') -> (0, 1, 2)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = list(le.classes_)

    # Perform stratified 80/20 train/test split to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )

    # Fit StandardScaler ONLY on training features to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nTraining Set Shape: {X_train_scaled.shape} (80%)")
    print(f"Testing Set Shape:  {X_test_scaled.shape} (20%)")

    # =========================================================================
    # STEP 4: Train Models & Benchmark Accuracy, Precision, Recall, and F1-Score
    # =========================================================================
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
    }

    benchmark_results = []
    predictions = {}

    for name, model in models.items():
        # Train classifier on scaled training set
        model.fit(X_train_scaled, y_train)

        # Predict on scaled test set
        y_pred = model.predict(X_test_scaled)
        predictions[name] = y_pred

        # Record evaluation metrics
        benchmark_results.append(
            {
                "Model": name,
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "Precision (Weighted)": round(
                    precision_score(y_test, y_pred, average="weighted"), 4
                ),
                "Recall (Weighted)": round(
                    recall_score(y_test, y_pred, average="weighted"), 4
                ),
                "F1-Score (Weighted)": round(
                    f1_score(y_test, y_pred, average="weighted"), 4
                ),
            }
        )

        print(f"\n--- {name} Classification Report ---")
        print(classification_report(y_test, y_pred, target_names=class_names))

    benchmark_df = pd.DataFrame(benchmark_results)
    benchmark_df.to_csv("outputs/benchmark_metrics.csv", index=False)

    print("\n=====================================================================")
    print("             FINAL MODEL BENCHMARK COMPARISON (80/20 SPLIT)          ")
    print("=====================================================================")
    print(benchmark_df.to_string(index=False))

    # =========================================================================
    # STEP 5A: Plot Confusion Matrix Heatmaps
    # =========================================================================
    print("\n[INFO] Generating Confusion Matrix Heatmaps...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            annot_kws={"size": 13, "weight": "bold"},
        )
        ax.set_title(f"{name} - Confusion Matrix", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted Species", fontsize=11)
        ax.set_ylabel("Actual Species", fontsize=11)

    plt.tight_layout()
    plt.savefig("outputs/2_confusion_matrix_heatmaps.png", dpi=300, bbox_inches="tight")
    plt.show()

    # =========================================================================
    # STEP 5B: Plot Decision Boundary Visualization (2D Feature Space)
    # =========================================================================
    print("[INFO] Generating 2D Decision Boundary Visualizations...")
    # Using the two most informative standardized features: petal_length (col 2) & petal_width (col 3)
    X_train_2d = X_train_scaled[:, [2, 3]]
    X_test_2d = X_test_scaled[:, [2, 3]]

    x_min, x_max = X_train_2d[:, 0].min() - 0.8, X_train_2d[:, 0].max() + 0.8
    y_min, y_max = X_train_2d[:, 1].min() - 0.8, X_train_2d[:, 1].max() + 0.8
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02),
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (name, base_model) in zip(axes, models.items()):
        # Fit a 2D instance of the classifier to visualize decision boundaries
        model_2d = base_model.__class__(**base_model.get_params())
        model_2d.fit(X_train_2d, y_train)

        # Predict class across every point in the 2D meshgrid
        Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        # Plot filled decision contours and overlay test samples
        ax.contourf(xx, yy, Z, alpha=0.35, cmap="Set2")
        scatter = ax.scatter(
            X_test_2d[:, 0],
            X_test_2d[:, 1],
            c=y_test,
            edgecolor="k",
            cmap="Set2",
            s=80,
        )

        ax.set_title(f"{name} - Decision Boundary", fontsize=13, fontweight="bold")
        ax.set_xlabel("Petal Length (Standardized)", fontsize=11)
        ax.set_ylabel("Petal Width (Standardized)", fontsize=11)
        ax.legend(
            handles=scatter.legend_elements()[0],
            labels=class_names,
            title="Species",
        )

    plt.tight_layout()
    plt.savefig("outputs/3_decision_boundaries.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("\n[SUCCESS] All outputs and plots saved to the 'outputs/' directory.")


if __name__ == "__main__":
    main()
