#!/usr/bin/env python3
"""
Train the Fake News Detection model (TF-IDF + Logistic Regression).
Creates models/fake_news_model.pkl and models/vectorizer.pkl.

Usage:
  python train_model.py                    # Use built-in demo data
  python train_model.py --data path.csv    # Use your own CSV (text, label columns)
"""

import os
import sys
import pickle
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Built-in demo data (minimal but sufficient for a working model)
DEMO_FAKE = [
    "SHOCKING!!! Government HIDING cure for cancer! Big Pharma secret!",
    "You WON'T BELIEVE what happens next!!! Doctors HATE this trick!",
    "BREAKING: Alien invasion COVER UP!!! They don't want you to KNOW!",
    "One weird trick to lose weight - experts are FURIOUS!!!",
    "Fake miracle cure EXPOSED - but Big Pharma silenced them!",
    "CONSPIRACY: NASA never went to moon!!! Proof inside!",
    "Celebrity DEATH HOAX - click to see the TRUTH they hide!",
    "Viral post claims unbelievable fact - share before deleted!!!",
    "The SECRET they don't want you to know - 100% guaranteed!!!",
    "FAKE NEWS alert: outrageous claim spreads like wildfire!!!",
] * 50  # Repeat for more training data

DEMO_REAL = [
    "Scientists at MIT announce breakthrough in battery technology according to research published in Nature Energy.",
    "Local community center announces new after-school programming for youth including sports and arts activities.",
    "Federal Reserve raises interest rates by quarter point citing inflation concerns.",
    "Study finds moderate exercise improves cognitive function in older adults.",
    "City council approves budget for infrastructure improvements and park maintenance.",
    "Researchers at Harvard publish findings on climate change impact in coastal regions.",
    "School district reports improved test scores following new curriculum implementation.",
    "Health department recommends flu vaccination ahead of winter season.",
    "Economic indicators suggest gradual recovery in manufacturing sector.",
    "University study examines effects of sleep on academic performance in students.",
] * 50


def get_demo_data():
    """Create demo DataFrame."""
    fake_df = pd.DataFrame({"text": DEMO_FAKE, "label": 0})
    real_df = pd.DataFrame({"text": DEMO_REAL, "label": 1})
    return pd.concat([fake_df, real_df], ignore_index=True)


def load_csv(path):
    """Load CSV with 'text' and 'label' columns (label: 0=fake, 1=real)."""
    df = pd.read_csv(path)
    if "text" not in df.columns:
        # Try common alternatives
        for col in ["title", "content", "article", "news"]:
            if col in df.columns:
                df = df.rename(columns={col: "text"})
                break
    if "label" not in df.columns:
        for col in ["labels", "target", "is_fake", "category"]:
            if col in df.columns:
                df = df.rename(columns={col: "label"})
                break
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            "CSV must have 'text' and 'label' columns (or similar). "
            f"Found: {list(df.columns)}"
        )
    # Normalize labels to 0/1
    df["label"] = (df["label"].astype(str).str.lower().str.contains("real|1|true")).astype(int)
    return df[["text", "label"]].dropna()


def train_and_save(data_path=None):
    if data_path and os.path.exists(data_path):
        print(f"Loading data from {data_path}...")
        df = load_csv(data_path)
    else:
        print("Using built-in demo data...")
        df = get_demo_data()

    print(f"Dataset: {len(df)} samples ({df['label'].sum()} real, {len(df) - df['label'].sum()} fake)")

    X = df["text"].fillna("").astype(str)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.2%}")
    print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

    os.makedirs("models", exist_ok=True)
    model_path = "models/fake_news_model.pkl"
    vec_path = "models/vectorizer.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"\nSaved: {model_path}")
    print(f"Saved: {vec_path}")
    print("You can now run: streamlit run App.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", help="Path to CSV (columns: text, label)")
    args = parser.parse_args()
    train_and_save(args.data)
