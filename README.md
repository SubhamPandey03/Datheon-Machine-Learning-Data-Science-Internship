# Task 3: Predictive Customer Churn Classifier (Random Forest & XGBoost)

An end-to-end ensemble machine learning pipeline designed to solve class imbalance and optimize **ROC-AUC (`>80%`)** for customer attrition prediction on the **IBM Telco Customer Churn** dataset.

---

## Technical Approach & Pipeline
1. **Dataset Loading:** Loads `7,043` customer subscription records across `21` attributes from the IBM Telco Customer Churn dataset.
2. **Data Cleaning (`TotalCharges`):** Converts `TotalCharges` from whitespace strings (`" "`) to numeric floats (`pd.to_numeric(..., errors="coerce")`), imputes the `11` missing records (corresponding to new `tenure = 0` customers) using median imputation, and removes the non-predictive `customerID` identifier.
3. **Categorical Encoding & Feature Scaling:** Applies `OneHotEncoder(drop="first")` to categorical subscription attributes and `StandardScaler()` to continuous variables (`tenure`, `MonthlyCharges`, `TotalCharges`).
4. **Class Imbalance Resolution (`SMOTE`):** Balances the training distribution (`73.6%` Retained vs. `26.4%` Churned) using **SMOTE** (*Synthetic Minority Over-sampling Technique*) to synthesize minority class examples (`4,142` Retained vs. `4,142` Churned).
5. **Ensemble Training & 5-Fold Cross-Validation:** Trains **Random Forest (`RandomForestClassifier`)** and **XGBoost (`XGBClassifier`)** using 5-Fold Stratified Cross-Validation (`StratifiedKFold`).
6. **Model Evaluation:** Benchmarks **ROC-AUC**, **Accuracy**, **Precision**, **Recall**, and **F1-Score**, and generates high-resolution plots for **ROC-AUC Curves**, **Confusion Matrices**, and **Top 10 Feature Importances**.

---

## Folder Structure
```text
Task_3_Customer_Churn_Classifier/
├── data/
│   └── Telco-Customer-Churn.csv               # IBM Telco Customer Churn dataset (7,043 rows)
├── outputs/
│   ├── 1_roc_auc_curves.png                   # ROC-AUC comparison curve (>80% target)
│   ├── 2_confusion_matrices.png               # Side-by-side confusion matrix heatmaps
│   ├── 3_top10_feature_importances.png        # Top 10 feature importance bar charts
│   └── churn_benchmark_metrics.csv            # Exported benchmark metrics table
├── Telco_Customer_Churn_Classifier.ipynb      # Executed Jupyter Notebook with embedded charts
├── churn_classification.py                    # Standalone well-commented Python script
├── requirements.txt                           # Required Python dependencies
└── README.md                                  # Task 3 documentation & benchmark report
```

---

## Model Benchmark Results (80/20 Stratified Test Split)

| Model | 5-Fold CV ROC-AUC | Test ROC-AUC | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (`RandomForestClassifier`)** | `0.9178` (91.78%) | **`0.8653` (86.53%)** | `0.7899` | `0.5729` | **`0.7690`** | `0.6566` |
| **XGBoost (`XGBClassifier`)** | **`0.9355` (93.55%)** | **`0.8671` (86.71%)** | **`0.8119`** | **`0.6388`** | `0.6440` | **`0.6414`** |

* Both ensemble classifiers comfortably exceed the **`>80%` ROC-AUC** benchmark requirement (`86.53%` for Random Forest and `86.71%` for XGBoost).
* **Top Churn Drivers:** Customer `tenure`, `TotalCharges`, `MonthlyCharges`, `Contract_Two year`, `Contract_One year`, `InternetService_Fiber optic`, and `PaymentMethod_Electronic check` rank as the most influential predictors of customer attrition.

---

## Generated Visualizations

### 1. ROC-AUC Curve Comparison (`>80%` Target)
![ROC-AUC Curves](outputs/1_roc_auc_curves.png)

### 2. Confusion Matrix Heatmaps
![Confusion Matrices](outputs/2_confusion_matrices.png)

### 3. Top 10 Most Influential Feature Importances
![Top 10 Feature Importances](outputs/3_top10_feature_importances.png)

---

## How to Run Locally
```bash
pip install -r requirements.txt
python churn_classification.py
```
