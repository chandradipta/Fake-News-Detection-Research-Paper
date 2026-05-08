"""
news_credibility_classifier.py
================================
Automated credibility classification of news articles using NLP
and supervised machine learning.

Pipeline overview:
  1. Load and validate the dataset
  2. Construct a combined text representation from available metadata fields
  3. Normalise text through regex cleaning, lowercasing, stopword removal, and stemming
  4. Vectorise using TF-IDF with sublinear term-frequency scaling
  5. Train and evaluate three classifiers: Logistic Regression, Naive Bayes, Random Forest
  6. Report per-model metrics and produce a confusion matrix

Dataset expected columns: id, title, author, text, label
  label -- 0 = credible/real, 1 = fabricated/fake

Usage:
  python news_credibility_classifier.py --data fake_news.csv
"""

import re
import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ── NLTK resource check ──────────────────────────────────────────────────────

def ensure_nltk_resources():
    """Download required NLTK data if not already present."""
    for resource in ("stopwords", "punkt"):
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

# ── Text normalisation ───────────────────────────────────────────────────────

_stemmer = PorterStemmer()
_stop_words = None  # initialised lazily after NLTK download


def _get_stop_words():
    global _stop_words
    if _stop_words is None:
        _stop_words = set(stopwords.words("english"))
    return _stop_words


def normalise_text(raw: str) -> str:
    """
    Apply a four-stage normalisation pipeline to a raw text string:
      1. Strip non-alphabetic characters (numerals, punctuation, symbols)
      2. Lowercase all tokens
      3. Remove English stopwords
      4. Reduce each remaining token to its Porter stem

    Parameters
    ----------
    raw : str
        Input text (may contain mixed content).

    Returns
    -------
    str
        Space-joined string of normalised stems.
    """
    # Stage 1 – remove non-alpha characters
    alpha_only = re.sub(r"[^a-zA-Z]", " ", raw)

    # Stage 2 – lowercase and tokenise
    tokens = alpha_only.lower().split()

    # Stages 3 & 4 – stopword removal and stemming
    stop_words = _get_stop_words()
    processed = [
        _stemmer.stem(token)
        for token in tokens
        if token not in stop_words
    ]

    return " ".join(processed)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Read the CSV dataset, fill missing values, and construct the composite
    input field used for classification.

    The composite field concatenates the author name and article title.
    Missing values in either column are imputed with empty strings so that
    all records survive the preprocessing stage.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional 'composite' column ready for vectorisation.
    """
    df = pd.read_csv(filepath)

    required_columns = {"author", "title", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Impute missing author / title with empty strings
    df["author"] = df["author"].fillna("")
    df["title"] = df["title"].fillna("")

    # Build composite input representation
    df["composite"] = df["author"] + " " + df["title"]

    print(f"[INFO] Loaded {len(df):,} records. "
          f"Class balance — real: {(df['label']==0).sum():,}, "
          f"fake: {(df['label']==1).sum():,}")

    return df


# ── Feature engineering ──────────────────────────────────────────────────────

def build_feature_matrix(train_texts, test_texts):
    """
    Fit a TF-IDF vectoriser on the training split and transform both splits.

    Sublinear TF scaling (log(1 + tf)) is applied to reduce the dominance
    of very frequent terms. The vectoriser is fitted exclusively on training
    data to prevent information leakage into the evaluation set.

    Parameters
    ----------
    train_texts : array-like of str
    test_texts  : array-like of str

    Returns
    -------
    X_train_tfidf, X_test_tfidf : sparse matrices
    vectoriser : fitted TfidfVectorizer instance
    """
    vectoriser = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.95)
    X_train_tfidf = vectoriser.fit_transform(train_texts)
    X_test_tfidf  = vectoriser.transform(test_texts)

    print(f"[INFO] Vocabulary size: {len(vectoriser.vocabulary_):,} terms")
    return X_train_tfidf, X_test_tfidf, vectoriser


# ── Model definitions ────────────────────────────────────────────────────────

def build_classifiers():
    """
    Return a dictionary of classifier instances keyed by a descriptive name.
    All hyperparameters are set to production-sensible defaults; no
    dataset-specific tuning has been applied.
    """
    return {
        "Logistic Regression": LogisticRegression(
            solver="liblinear",
            C=1.0,
            max_iter=1000,
            random_state=42
        ),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            criterion="gini",
            max_features="sqrt",
            n_jobs=-1,
            random_state=42
        ),
    }


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_classifier(name, model, X_train, y_train, X_test, y_test):
    """
    Train a classifier, evaluate it on the held-out test set, and return
    a summary dictionary of performance metrics.
    """
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "model":     name,
        "accuracy":  accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall":    recall_score(y_test, predictions, zero_division=0),
        "f1":        f1_score(y_test, predictions, zero_division=0),
    }

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(classification_report(y_test, predictions,
                                 target_names=["Real", "Fake"]))
    return metrics, predictions


def plot_confusion_matrix(y_true, y_pred, model_name, output_path=None):
    """Render and optionally save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar(im, ax=ax)

    labels = ["Real", "Fake"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, pad=10)

    for r in range(2):
        for c in range(2):
            ax.text(c, r, str(cm[r, c]),
                    ha="center", va="center",
                    color="white" if cm[r, c] > cm.max() / 2 else "black",
                    fontsize=13, fontweight="bold")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Confusion matrix saved to {output_path}")
    else:
        plt.show()
    plt.close()


def summarise_results(all_metrics):
    """Print a formatted comparison table of all classifiers."""
    print("\n" + "="*60)
    print("  COMPARATIVE RESULTS SUMMARY")
    print("="*60)
    header = f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}"
    print(header)
    print("-"*60)
    for m in all_metrics:
        print(f"{m['model']:<22} {m['accuracy']:>9.4f} "
              f"{m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}")
    print("="*60)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="News credibility classifier — NLP + ML pipeline"
    )
    parser.add_argument(
        "--data", default="fake_news.csv",
        help="Path to the labelled CSV dataset (default: fake_news.csv)"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data held out for testing (default: 0.2)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--save-cm", action="store_true",
        help="Save confusion matrix images to disk"
    )
    args = parser.parse_args()

    # Step 1 – ensure NLTK resources
    ensure_nltk_resources()

    # Step 2 – load data
    df = load_dataset(args.data)

    # Step 3 – normalise text
    print("\n[INFO] Normalising text (this may take a moment)…")
    df["normalised"] = df["composite"].apply(normalise_text)

    # Step 4 – train / test split (stratified)
    X_raw = df["normalised"]
    y     = df["label"]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.seed
    )
    print(f"[INFO] Train size: {len(X_train_raw):,} | Test size: {len(X_test_raw):,}")

    # Step 5 – build TF-IDF features
    X_train, X_test, _ = build_feature_matrix(X_train_raw, X_test_raw)

    # Step 6 – train and evaluate classifiers
    classifiers  = build_classifiers()
    all_metrics  = []

    for name, clf in classifiers.items():
        metrics, preds = evaluate_classifier(
            name, clf, X_train, y_train, X_test, y_test
        )
        all_metrics.append(metrics)

        if args.save_cm:
            safe_name = name.lower().replace(" ", "_")
            plot_confusion_matrix(
                y_test, preds, name,
                output_path=f"confusion_matrix_{safe_name}.png"
            )

    # Step 7 – summary table
    summarise_results(all_metrics)


if __name__ == "__main__":
    main()
