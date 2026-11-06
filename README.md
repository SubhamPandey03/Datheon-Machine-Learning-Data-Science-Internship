# Supervised Classification Benchmark on the Iris Dataset

A complete supervised machine learning pipeline benchmarking **Logistic Regression** and **Decision Tree** classifiers on the Iris dataset. Built with `pandas`, `scikit-learn`, `seaborn`, and `matplotlib`.

---

## Project Overview & Technical Approach
1. **Dataset Loading:** Loads the Iris dataset (`150` samples, `4` morphological features, `3` species classes) using `pandas`.
2. **Exploratory Data Analysis (EDA):** Visualizes pairwise feature relationships and class separability using `seaborn.pairplot()`.
3. **Preprocessing & Scaling:**
   - Splits data into **80% training (`120` samples)** and **20% testing (`30` samples)** using stratified sampling (`random_state=42`).
   - Normalizes features to zero mean and unit variance using `StandardScaler()` (fitted strictly on training data to prevent data leakage).
4. **Model Training & Benchmarking:** Trains `LogisticRegression` and `DecisionTreeClassifier` (`max_depth=3`) and evaluates **Accuracy, Precision, Recall, and F1-Score**.
5. **Visual Evaluation:** Generates side-by-side **Confusion Matrix Heatmaps** and **2D Decision Boundary Visualizations**.

---

## Repository Structure
```text
iris-classification-benchmark/
├── data/
│   └── iris.csv                        # Cached Iris benchmark dataset
├── outputs/
│   ├── 1_iris_pairplot.png             # Exploratory Seaborn pairplot
│   ├── 2_confusion_matrix_heatmaps.png # Side-by-side confusion matrices
│   ├── 3_decision_boundaries.png       # 2D decision boundary contour plots
│   └── benchmark_metrics.csv           # Exported evaluation metrics table
├── Iris_Classification_Benchmark.ipynb # Interactive Jupyter Notebook
├── iris_classification.py              # Standalone well-commented Python script
├── requirements.txt                    # Python package dependencies
└── README.md                           # Project documentation & results
```

---

## Benchmark Results (80/20 Stratified Test Split)

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | `0.9333` (93.33%) | `0.9333` | `0.9333` | `0.9333` |
| **Decision Tree (`max_depth=3`)** | **`0.9667` (96.67%)** | **`0.9697`** | **`0.9667`** | **`0.9666`** |

### Key Observations
- **Exploratory Pairplot:** *Iris-setosa* is linearly separable from the other two species across all petal features, while *Iris-versicolor* and *Iris-virginica* exhibit slight boundary overlap.
- **Model Comparison:** Both models classify 100% of *Iris-setosa* test instances accurately. On the 30-sample test split, **Decision Tree** achieves `96.67%` accuracy (only 1 misclassification) via orthogonal threshold splits, slightly outperforming **Logistic Regression** (`93.33%` accuracy, 2 misclassifications near the linear boundary between *versicolor* and *virginica*).

---

## Visualizations

### 1. Exploratory Pairplot
![Exploratory Pairplot](outputs/1_iris_pairplot.png)

### 2. Confusion Matrix Heatmaps
![Confusion Matrix Heatmaps](outputs/2_confusion_matrix_heatmaps.png)

### 3. Decision Boundary Visualization
![Decision Boundary Visualization](outputs/3_decision_boundaries.png)

---

## How to Run Locally in VS Code

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd iris-classification-benchmark
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Python script:**
   ```bash
   python iris_classification.py
   ```
   *(Or open `Iris_Classification_Benchmark.ipynb` in VS Code / Jupyter Notebook and click **Run All**).*
