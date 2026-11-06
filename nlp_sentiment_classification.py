"""
Task 4: NLP Sentiment Classification on Customer Reviews (TF-IDF)
=================================================================
Objectives:
1. Load customer product/movie sentiment review records (sentiment_reviews.csv).
2. Preprocess raw text using NLTK / regex (remove HTML tags, punctuation,
   lowercasing, stopword removal, and lemmatization).
3. Transform cleaned text into numerical feature matrices using
   TfidfVectorizer(max_features=5000).
4. Train Multinomial Naive Bayes (MultinomialNB) and Logistic Regression
   (LogisticRegression) classifiers to identify positive and negative sentiment,
   and benchmark Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
5. Provide an interactive predict_sentiment() helper function to test live
   custom user input phrases.
"""

import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

# Attempt to load NLTK stopwords and WordNetLemmatizer if available locally
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    try:
        STOPWORDS_SET = set(stopwords.words("english")) - {"not", "no", "never", "nothing", "none", "nor", "very", "too"}
    except LookupError:
        STOPWORDS_SET = set(ENGLISH_STOP_WORDS) - {"not", "no", "never", "nothing", "none", "nor", "very", "too"}
    W_LEMMATIZER = WordNetLemmatizer()
    HAS_NLTK = True
except ImportError:
    STOPWORDS_SET = set(ENGLISH_STOP_WORDS) - {"not", "no", "never", "nothing", "none", "nor", "very", "too"}
    W_LEMMATIZER = None
    HAS_NLTK = False

# Deterministic morphological lemmatization lookup for offline reliability
LEMMA_DICT = {
    "nothing": "nothing", "something": "something", "everything": "everything", "anything": "anything",
    "amazing": "amazing", "outstanding": "outstanding", "exciting": "exciting", "interesting": "interesting",
    "boring": "boring", "packaging": "packaging", "shipping": "shipping", "morning": "morning",
    "evening": "evening", "during": "during", "ceiling": "ceiling", "feeling": "feeling",
    "exceeded": "exceed", "exceeding": "exceed", "exceeds": "exceed", "expectations": "expectation",
    "arrived": "arrive", "arrives": "arrive", "arriving": "arrive", "damaged": "damage", "damaging": "damage",
    "works": "work", "working": "work", "worked": "work", "stopped": "stop", "stopping": "stop",
    "days": "day", "disappointed": "disappoint", "disappointing": "disappoint", "helped": "help",
    "minutes": "minute", "hours": "hour", "weeks": "week", "months": "month", "missing": "missing",
    "controls": "control", "recommended": "recommend", "recommending": "recommend", "students": "student",
    "alternatives": "alternative", "purchased": "purchase", "purchasing": "purchase", "bought": "buy",
    "loved": "love", "loving": "love", "hated": "hate", "features": "feature", "movies": "movie",
    "films": "film", "actors": "actor", "scenes": "scene", "products": "product", "items": "item",
    "orders": "order", "ordered": "order", "packages": "package", "deliveries": "delivery",
    "delivered": "deliver", "instructions": "instruction", "batteries": "battery", "screens": "screen",
    "buttons": "button", "headphones": "headphone", "speakers": "speaker", "laptops": "laptop",
    "cameras": "camera", "chargers": "charger", "adapters": "adapter", "prices": "price",
    "reviews": "review", "customers": "customer", "services": "service", "issues": "issue",
    "problems": "problem", "defects": "defect", "defective": "defect", "broken": "break", "broke": "break",
    "cracked": "crack", "scratched": "scratch", "overheated": "overheat", "overheating": "overheat",
    "crashed": "crash", "crashing": "crash", "crashes": "crash", "freezes": "freeze", "freezing": "freeze",
    "refunded": "refund", "refunds": "refund", "returned": "return", "returning": "return",
    "wasted": "waste", "wasting": "waste", "fails": "fail", "failed": "fail", "failing": "fail",
    "satisfied": "satisfy", "worried": "worry", "delayed": "delay", "promised": "promise",
    "described": "describe", "advertised": "advertise", "resolved": "resolve", "refused": "refuse",
    "ignored": "ignore", "manufactured": "manufacture", "confused": "confuse", "confusing": "confusing",
    "performed": "perform", "performing": "perform", "performs": "perform"
}


def lemmatize_token(token):
    """Lemmatize a single token using NLTK WordNet (if downloaded) and morphological rules."""
    if token in LEMMA_DICT:
        return LEMMA_DICT[token]
    if HAS_NLTK and W_LEMMATIZER is not None:
        try:
            v_lem = W_LEMMATIZER.lemmatize(token, pos="v")
            if v_lem != token:
                return v_lem
            return W_LEMMATIZER.lemmatize(token, pos="n")
        except LookupError:
            pass
    if len(token) > 5:
        if token.endswith("ied") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("ing") and len(token) > 5:
            base = token[:-3]
            if len(base) >= 3 and base[-1] == base[-2]:
                base = base[:-1]
            return base
        if token.endswith("ed") and len(token) > 5:
            base = token[:-2]
            if len(base) >= 3 and base[-1] == base[-2]:
                base = base[:-1]
            return base
        if token.endswith("s") and not token.endswith(("ss", "us", "is", "ous")):
            return token[:-1]
    return token


def preprocess_text(text):
    """
    Step 2: Clean and tokenize raw review text:
    - Strip HTML tags (<br />, <p>, <i>, etc.) and URLs via regex
    - Lowercase and remove punctuation / numeric characters
    - Remove stopwords (retaining key negation modifiers)
    - Lemmatize tokens to their root form
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    cleaned_tokens = [
        lemmatize_token(tok)
        for tok in tokens
        if tok not in STOPWORDS_SET and len(tok) > 1
    ]
    return " ".join(cleaned_tokens)


def predict_sentiment(text, model, vectorizer, return_dict=False):
    """
    Step 5: Interactive helper function that preprocesses a live custom user input
    phrase, vectorizes it using the trained TfidfVectorizer, and predicts sentiment.
    """
    cleaned = preprocess_text(text)
    tfidf_vec = vectorizer.transform([cleaned])
    pred_label = int(model.predict(tfidf_vec)[0])
    probs = model.predict_proba(tfidf_vec)[0]

    polarity = "positive" if pred_label == 1 else "negative"
    confidence = float(probs[pred_label])

    if return_dict:
        return {
            "raw_text": text,
            "cleaned_text": cleaned,
            "predicted_sentiment": polarity,
            "confidence": round(confidence, 4),
            "prob_positive": round(float(probs[1]), 4),
            "prob_negative": round(float(probs[0]), 4),
        }
    return f"{polarity.upper()} (Confidence: {confidence * 100:.2f}%)"


def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    sns.set_theme(style="whitegrid")

    # =========================================================================
    # STEP 1: Load Sentiment Reviews Dataset
    # =========================================================================
    csv_path = os.path.join("data", "sentiment_reviews.csv")
    if not os.path.exists(csv_path) and os.path.exists("sentiment_reviews.csv"):
        csv_path = "sentiment_reviews.csv"

    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded dataset from '{csv_path}' | Shape: {df.shape}")
    print("\nSentiment Class Distribution:")
    print(df["sentiment"].value_counts())

    # Filter binary positive and negative sentiment records for polarity classification
    df_binary = df[df["sentiment"].isin(["positive", "negative"])].copy().reset_index(drop=True)

    # =========================================================================
    # STEP 2: Preprocess Text (HTML Stripping, Punctuation, Stopwords, Lemmatization)
    # =========================================================================
    print("\n[INFO] Preprocessing and lemmatizing textual corpus...")
    df_binary["cleaned_text"] = df_binary["review_text"].apply(preprocess_text)
    df_binary["label"] = (df_binary["sentiment"] == "positive").astype(int)

    print("\nSample Raw vs. Cleaned & Lemmatized Reviews:")
    for i in range(min(5, len(df_binary))):
        print(f"  [{df_binary.loc[i, 'sentiment'].upper()}] Raw:   {df_binary.loc[i, 'review_text']}")
        print(f"             Clean: {df_binary.loc[i, 'cleaned_text']}")

    # =========================================================================
    # STEP 3: Convert Text to Numerical Vectors using TfidfVectorizer(max_features=5000)
    # =========================================================================
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df_binary["cleaned_text"],
        df_binary["label"],
        test_size=0.20,
        random_state=42,
        stratify=df_binary["label"],
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)

    print(f"\n[INFO] TF-IDF Training Matrix Shape: {X_train_tfidf.shape}")
    print(f"[INFO] TF-IDF Testing Matrix Shape:  {X_test_tfidf.shape}")

    # =========================================================================
    # STEP 4: Train MultinomialNB & LogisticRegression and Benchmark Accuracy
    # =========================================================================
    models = {
        "Multinomial Naive Bayes (MultinomialNB)": MultinomialNB(alpha=1.0),
        "Logistic Regression (LogisticRegression)": LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    benchmark_rows = []
    roc_data = {}
    cm_data = {}

    for name, model in models.items():
        cv_acc = cross_val_score(model, X_train_tfidf, y_train, cv=cv, scoring="accuracy")
        model.fit(X_train_tfidf, y_train)

        y_pred = model.predict(X_test_tfidf)
        y_prob = model.predict_proba(X_test_tfidf)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, auc)
        cm_data[name] = confusion_matrix(y_test, y_pred)

        benchmark_rows.append(
            {
                "Model": name,
                "5-Fold CV Accuracy": round(cv_acc.mean(), 4),
                "Test Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1-Score": round(f1, 4),
                "ROC-AUC": round(auc, 4),
            }
        )

        print(f"\n--- {name} Classification Report ---")
        print(classification_report(y_test, y_pred, target_names=["Negative (0)", "Positive (1)"]))

    benchmark_df = pd.DataFrame(benchmark_rows)
    benchmark_df.to_csv("outputs/sentiment_benchmark_metrics.csv", index=False)

    print("\n====================================================================================")
    print("             NLP SENTIMENT CLASSIFICATION BENCHMARK (80/20 TEST SPLIT)              ")
    print("====================================================================================")
    print(benchmark_df.to_string(index=False))

    # Plot 1: Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    class_labels = ["Negative (0)", "Positive (1)"]
    cmaps = ["Blues", "Greens"]

    for ax, (name, cm), cmap in zip(axes, cm_data.items(), cmaps):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            cbar=False,
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=ax,
            annot_kws={"size": 14, "weight": "bold"},
        )
        ax.set_title(f"{name}\nConfusion Matrix (Test Split)", fontsize=12.5, fontweight="bold")
        ax.set_xlabel("Predicted Sentiment Polarity", fontsize=11)
        ax.set_ylabel("Actual Sentiment Polarity", fontsize=11)

    plt.tight_layout()
    plt.savefig("outputs/1_confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Plot 2: ROC-AUC & Metrics Benchmark
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = {
        "Multinomial Naive Bayes (MultinomialNB)": "#1f77b4",
        "Logistic Regression (LogisticRegression)": "#2ca02c",
    }
    for name, (fpr, tpr, auc_val) in roc_data.items():
        short_name = "MultinomialNB" if "Naive" in name else "LogisticRegression"
        axes[0].plot(
            fpr,
            tpr,
            linewidth=2.6,
            color=colors[name],
            label=f"{short_name} (ROC-AUC = {auc_val:.4f} / {auc_val * 100:.2f}%)",
        )
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1.4, label="Random Baseline (AUC = 0.5000)")
    axes[0].set_title("ROC-AUC Curve Comparison (TF-IDF Features)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    axes[0].legend(loc="lower right", fontsize=10.5, frameon=True)

    metric_cols = ["5-Fold CV Accuracy", "Test Accuracy", "Precision", "Recall", "F1-Score"]
    melted_df = benchmark_df.melt(
        id_vars="Model", value_vars=metric_cols, var_name="Metric", value_name="Score"
    )
    melted_df["ShortModel"] = melted_df["Model"].apply(
        lambda x: "MultinomialNB" if "Naive" in x else "LogisticRegression"
    )
    sns.barplot(
        data=melted_df,
        x="Metric",
        y="Score",
        hue="ShortModel",
        palette=["#1f77b4", "#2ca02c"],
        ax=axes[1],
    )
    axes[1].set_title("Model Performance Benchmark (TF-IDF + 80/20 Split)", fontsize=13, fontweight="bold")
    axes[1].set_ylim(0.85, 1.01)
    axes[1].tick_params(axis="x", rotation=15)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.3f", padding=3, fontsize=8.5)

    plt.tight_layout()
    plt.savefig("outputs/2_roc_auc_and_metrics_benchmark.png", dpi=300, bbox_inches="tight")
    plt.show()

    # =========================================================================
    # STEP 5: Test Live Custom User Input Phrases with predict_sentiment()
    # =========================================================================
    nb_model = models["Multinomial Naive Bayes (MultinomialNB)"]
    lr_model = models["Logistic Regression (LogisticRegression)"]

    print("\n====================================================================================")
    print("             INTERACTIVE SENTIMENT PREDICTION HELPER (predict_sentiment)            ")
    print("====================================================================================")
    custom_test_phrases = [
        "<b>Amazing product!</b> Easy to setup, robust, and the battery lasts all day.",
        "Terrible customer service! The item arrived completely damaged and stopped working.",
        "The movie had stunning visuals, great acting, and exceeded all my expectations!",
        "Not worth the money at all. Missing adapter cable and software freezes constantly.",
        "Clean interface and intuitive controls. Highly recommended for college students.",
    ]

    for phrase in custom_test_phrases:
        nb_res = predict_sentiment(phrase, model=nb_model, vectorizer=vectorizer)
        lr_res = predict_sentiment(phrase, model=lr_model, vectorizer=vectorizer)
        print(f"Phrase: {phrase}")
        print(f"  -> Cleaned:            {preprocess_text(phrase)}")
        print(f"  -> MultinomialNB:      {nb_res}")
        print(f"  -> LogisticRegression: {lr_res}\n")


if __name__ == "__main__":
    main()
