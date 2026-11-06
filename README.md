# Task 4: NLP Sentiment Classification on Customer Reviews (TF-IDF)

An end-to-end Natural Language Processing (NLP) text classification pipeline that preprocesses raw customer product and movie reviews, extracts numerical **TF-IDF (`TfidfVectorizer(max_features=5000)`)** feature embeddings, benchmarks **Multinomial Naive Bayes (`MultinomialNB`)** and **Logistic Regression (`LogisticRegression`)** classifiers, and provides an interactive `predict_sentiment()` helper for live custom phrase inference.

---

## Technical Approach & Pipeline
1. **Corpus Ingestion:** Loads the customer sentiment review dataset (`data/sentiment_reviews.csv`, `1,201` total records comprising `600` positive, `600` negative, and `1` neutral starter record, alongside `data/sentiment_reviews_starter.csv`).
2. **Text Preprocessing (`preprocess_text`):**
   - Strips HTML markup tags (`<br />`, `<p>`, `<i>`, `<b>`) and URLs using regular expressions (`re.sub`).
   - Converts text to lowercase and removes punctuation and numeric characters.
   - Removes English stopwords while preserving negation words (`not`, `no`, `never`, `nothing`).
   - Lemmatizes inflected tokens to their base root form (`exceeded -> exceed`, `damaged -> damage`, `working -> work`, `disappointed -> disappoint`).
3. **TF-IDF Feature Extraction:** Converts cleaned text into numerical feature vectors using `TfidfVectorizer(max_features=5000, ngram_range=(1, 2))` on an 80/20 stratified train/test split (`960` train, `240` test samples).
4. **Model Training & Benchmarking:** Trains `MultinomialNB` and `LogisticRegression` with 5-Fold Stratified Cross-Validation and evaluates **Test Accuracy, Precision, Recall, F1-Score, and ROC-AUC**.
5. **Interactive Test Helper (`predict_sentiment`):** Provides a reusable function that cleans, vectorizes, and predicts sentiment polarity and confidence scores on live custom user input phrases.

---

## Folder Structure
```text
Task_4_NLP_Sentiment_Classification/
├── data/
│   ├── sentiment_reviews.csv                    # Full benchmark review corpus (1,201 rows)
│   └── sentiment_reviews_starter.csv            # Original 10-row starter file
├── outputs/
│   ├── 1_confusion_matrices.png                 # Side-by-side Confusion Matrix heatmaps
│   ├── 2_roc_auc_and_metrics_benchmark.png      # ROC-AUC curves & metric bar chart comparison
│   ├── 3_top_tfidf_sentiment_features.png       # Top 12 Positive vs. Negative TF-IDF terms
│   └── sentiment_benchmark_metrics.csv          # Exported benchmark metrics table
├── NLP_Sentiment_Classification.ipynb           # Executed Jupyter Notebook with embedded charts
├── nlp_sentiment_classification.py              # Standalone well-commented Python script
├── requirements.txt                             # Required Python dependencies
└── README.md                                    # Task 4 documentation & benchmark report
```

---

## Model Benchmark Results (80/20 Stratified Test Split)

| Model | 5-Fold CV Accuracy | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes (`MultinomialNB`)** | `0.9510` (95.10%) | **`0.9542` (95.42%)** | **`0.9580`** | `0.9500` | **`0.9540`** | `0.9830` (98.30%) |
| **Logistic Regression (`LogisticRegression`)** | `0.9510` (95.10%) | `0.9458` (94.58%) | `0.9421` | `0.9500` | `0.9461` | **`0.9883` (98.83%)** |

---

## Generated Visualizations

### 1. Confusion Matrix Heatmaps (`MultinomialNB` vs. `LogisticRegression`)
![Confusion Matrices](outputs/1_confusion_matrices.png)

### 2. ROC-AUC Curves & Classification Metric Benchmark
![ROC-AUC and Metrics Benchmark](outputs/2_roc_auc_and_metrics_benchmark.png)

### 3. Top 12 Positive vs. Negative TF-IDF Sentiment Features
![Top TF-IDF Sentiment Features](outputs/3_top_tfidf_sentiment_features.png)

---

## How to Run Locally
```bash
pip install -r requirements.txt
python nlp_sentiment_classification.py
```
