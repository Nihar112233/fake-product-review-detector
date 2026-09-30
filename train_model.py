import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from preprocessing import clean_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "amazon_reviews.txt")
PROCESSED_PATH = os.path.join(BASE_DIR, "dataset", "processed_reviews.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

LABEL_MAP = {
    "__label1__": 1,  # Fake
    "__label2__": 0,  # Genuine
}

TARGET_NAMES = ["Genuine", "Fake"]


def load_and_clean_data(file_path=DATASET_PATH):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}\n"
        )

    df = pd.read_csv(file_path, sep="\t", dtype=str)
    
    required = ["REVIEW_TEXT", "LABEL", "RATING", "VERIFIED_PURCHASE", "PRODUCT_CATEGORY"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing expected columns {missing_cols}. "
        )

    df["REVIEW_TEXT"] = df["REVIEW_TEXT"].fillna("").astype(str)
    
    df["raw_label"] = df["LABEL"].astype(str).str.strip()
    
    known_labels = set(LABEL_MAP.keys())
    df = df[df["raw_label"].isin(known_labels)].copy()
    
    df["label"] = df["raw_label"].map(LABEL_MAP)
    
    df["RATING"] = pd.to_numeric(df["RATING"], errors="coerce").fillna(3)
    df["VERIFIED_PURCHASE"] = df["VERIFIED_PURCHASE"].fillna("N").astype(str)
    df["PRODUCT_CATEGORY"] = df["PRODUCT_CATEGORY"].fillna("Unknown").astype(str)
    
    df = df.drop_duplicates(subset=["REVIEW_TEXT"]).reset_index(drop=True)
    
    genuine = (df["label"] == 0).sum()
    fake = (df["label"] == 1).sum()
    print("\nUsable reviews:", len(df))
    print(f"  Genuine: {genuine}")
    print(f"  Fake:    {fake}")
    
    return df


def _metrics_from_predictions(y_true, y_pred, model_name):
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=TARGET_NAMES, zero_division=0
    )

    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def train_and_evaluate_pipeline(df, test_size=0.20, random_state=42, save_artifacts=True):
    data = df.copy()
    
    print("Cleaning text...")
    data["cleaned_text"] = data["REVIEW_TEXT"].apply(clean_text)
    
    data = data[data["cleaned_text"].str.strip() != ""].reset_index(drop=True)
    
    genuine = int((data["label"] == 0).sum())
    fake = int((data["label"] == 1).sum())

    X = data[["cleaned_text", "RATING", "VERIFIED_PURCHASE", "PRODUCT_CATEGORY"]]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, sublinear_tf=True), "cleaned_text"),
            ("num", MinMaxScaler(), ["RATING"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["VERIFIED_PURCHASE", "PRODUCT_CATEGORY"])
        ]
    )

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(X_train_transformed, y_train)
    nb_pred = nb_model.predict(X_test_transformed)

    lr_model = LogisticRegression(max_iter=1000, random_state=random_state)
    lr_model.fit(X_train_transformed, y_train)
    lr_pred = lr_model.predict(X_test_transformed)

    nb_metrics = _metrics_from_predictions(y_test, nb_pred, "Naive Bayes")
    lr_metrics = _metrics_from_predictions(y_test, lr_pred, "Logistic Regression")

    results = {
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "total_samples": int(len(data)),
        "genuine_count": genuine,
        "fake_count": fake,
        "vocabulary_size": len(preprocessor.named_transformers_['text'].vocabulary_),
        "label_mapping": {
            "__label1__": "Fake (1)",
            "__label2__": "Genuine (0)",
        },
        "nb_metrics": nb_metrics,
        "lr_metrics": lr_metrics,
        "target_names": TARGET_NAMES,
    }

    if save_artifacts:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.joblib"))
        joblib.dump(lr_model, os.path.join(MODEL_DIR, "lr_model.joblib"))
        joblib.dump(nb_model, os.path.join(MODEL_DIR, "nb_model.joblib"))
        joblib.dump(results, os.path.join(MODEL_DIR, "metrics.joblib"))
        with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return {
        "preprocessor": preprocessor,
        "lr_model": lr_model,
        "nb_model": nb_model,
        "results": results,
    }


def _probability_map(model, features):
    if not hasattr(model, "predict_proba"):
        return None
    probs = model.predict_proba(features)[0]
    return {int(cls): float(p) for cls, p in zip(model.classes_, probs)}


def predict_review(review_text, rating=3, verified_purchase="N", product_category="Unknown", preprocessor=None, model=None, algorithm="Logistic Regression"):
    if not str(review_text).strip():
        return {"ok": False, "error": "Please enter a product review before analyzing."}

    if preprocessor is None or model is None:
        prep_path = os.path.join(MODEL_DIR, "preprocessor.joblib")
        lr_path = os.path.join(MODEL_DIR, "lr_model.joblib")
        nb_path = os.path.join(MODEL_DIR, "nb_model.joblib")
        if not (os.path.exists(prep_path) and os.path.exists(lr_path) and os.path.exists(nb_path)):
            return {"ok": False, "error": "Trained models not found."}
        preprocessor = joblib.load(prep_path)
        model = joblib.load(lr_path if algorithm == "Logistic Regression" else nb_path)

    cleaned = clean_text(str(review_text))
    if not cleaned:
        return {"ok": False, "error": "The review did not contain enough usable words."}

    input_df = pd.DataFrame([{
        "cleaned_text": cleaned,
        "RATING": float(rating),
        "VERIFIED_PURCHASE": str(verified_purchase),
        "PRODUCT_CATEGORY": str(product_category)
    }])

    features = preprocessor.transform(input_df)
    pred_idx = int(model.predict(features)[0])

    prob_map = _probability_map(model, features)
    confidence = prob_map.get(pred_idx) if prob_map else None

    if pred_idx == 0:
        prediction = "Genuine Review"
    else:
        prediction = "Fake Review"

    return {
        "ok": True,
        "error": None,
        "prediction": prediction,
        "class_idx": pred_idx,
        "confidence": confidence,
    }


if __name__ == "__main__":
    df = load_and_clean_data()
    artifacts = train_and_evaluate_pipeline(df)
    print("Training finished.")
    
    # print evaluation
    results = artifacts["results"]
    print("\n--- Evaluation (same 80/20 stratified split) ---")
    for key in ("nb_metrics", "lr_metrics"):
        m = results[key]
        print(f"\n{m['model_name']}")
        print(f"  Accuracy:  {m['accuracy'] * 100:.2f}%")
        print(f"  Precision: {m['precision'] * 100:.2f}%")
        print(f"  Recall:    {m['recall'] * 100:.2f}%")
        print(f"  F1-score:  {m['f1_score'] * 100:.2f}%")

