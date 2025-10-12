#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_data_preprocessing.py
- Starts a new MLflow run named "DataPreprocessing"
- Loads processed_data/validated_data.csv
- Cleans text: lowercase, keep digits/!?', remove other non-word chars, keep negations, remove other English stopwords
- Creates 'clean_review', defines X=clean_review, y=Star
- Splits data (train/test 80/20, random_state=42, stratify=y)
- Fits a stronger TfidfVectorizer on X_train, transforms both
- Logs fitted vectorizer.pkl to MLflow as an artifact
- Saves train/test CSVs to processed_data/
- Writes current MLflow Run ID to mlops_pipeline/preprocessing_run_id.txt
"""

import re
import sys
import joblib
from pathlib import Path
from typing import Iterable

import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.pipeline import FeatureUnion


# --- Cleaning config: keep negations; allow digits and !?',. ---
NEGATIONS = {
    "no", "not", "nor", "never", "n't",
    "cannot", "can't", "won't", "isn't", "aren't", "wasn't", "weren't",
    "don't", "doesn't", "didn't", "haven't", "hasn't", "hadn't",
    "wouldn't", "shouldn't", "mightn't", "mustn't"
}
STOPWORDS = set(ENGLISH_STOP_WORDS) - NEGATIONS

# เก็บตัวอักษร/ตัวเลข/เว้นวรรค + !?'.,  (อย่าตัดอีโมจิออก — ปล่อยให้คงอยู่)
CLEAN_RE = re.compile(r"[^a-z0-9\s!?\.',]")

def clean_text(text: str, stopwords: Iterable[str]) -> str:
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    text = text.lower()
    text = CLEAN_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords]
    return " ".join(tokens)


def main() -> int:
    # === Paths ===
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent  # repo root
    processed_dir = project_root / "processed_data"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Inputs / outputs
    validated_csv = processed_dir / "validated_data.csv"
    train_csv = processed_dir / "train.csv"
    test_csv = processed_dir / "test.csv"
    vec_path = processed_dir / "vectorizer.pkl"
    run_id_file = project_root / "mlops_pipeline" / "preprocessing_run_id.txt"

    # === MLflow tracking (local dir so it works in GitHub Actions) ===
    tracking_dir = project_root / "mlruns"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment("MLOps_Pipeline_Experiment")

    mlflow.start_run(run_name="DataPreprocessing")
    try:
        # === Load validated data ===
        if not validated_csv.exists():
            raise FileNotFoundError(f"Missing input file: {validated_csv}")

        df = pd.read_csv(validated_csv, encoding="utf-8")
        required_cols = {"review", "Star"}
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"validated_data.csv missing required columns: {sorted(missing)}")

        mlflow.log_param("input_file", str(validated_csv))
        mlflow.log_metric("rows_loaded", len(df))

        # === Text cleaning ===
        df = df.copy()
        df["clean_review"] = df["review"].apply(lambda x: clean_text(x, STOPWORDS))

        # Diagnostics
        mlflow.log_metric("empty_clean_reviews", int((df["clean_review"].str.len() == 0).sum()))
        avg_len = float(df["clean_review"].str.split().map(len).mean())
        med_len = float(df["clean_review"].str.split().map(len).median())
        mlflow.log_metric("avg_clean_len", avg_len)
        mlflow.log_metric("median_clean_len", med_len)

        # === Define X, y ===
        X = df["clean_review"].astype(str)
        y = df["Star"].astype(int)

        # === Train/test split ===
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # === Stronger HYBRID TF-IDF (word + char) ===
        word_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),     # word 1–3
            min_df=3,
            max_df=0.95,
            sublinear_tf=True,
            norm="l2",
            lowercase=False,
            token_pattern=r"(?u)\b\w+\b",
        )
        char_vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 6),     # char 3–6
            min_df=3,
            sublinear_tf=True,
            norm="l2",
            lowercase=False,
        )
        hybrid_vec = FeatureUnion([
            ("word", word_vec),
            ("char", char_vec),
        ])

        X_train_vec = hybrid_vec.fit_transform(X_train)
        X_test_vec  = hybrid_vec.transform(X_test)

        mlflow.log_param("vectorizer_type", "FeatureUnion(word[1,3]+char[3,6])")
        mlflow.log_param("word_ngram", "(1,3)")
        mlflow.log_param("char_ngram", "(3,6)")
        mlflow.log_param("min_df", 3)
        mlflow.log_param("max_df", 0.95)
        mlflow.log_param("sublinear_tf", True)

        # === Log fitted vectorizer ===
        joblib.dump(hybrid_vec, vec_path)
        mlflow.log_artifact(str(vec_path))

        # === Save splits (CSV) ===
        pd.DataFrame({"clean_review": X_train.values, "Star": y_train.values}).to_csv(train_csv, index=False, encoding="utf-8")
        pd.DataFrame({"clean_review": X_test.values,  "Star": y_test.values}).to_csv(test_csv,  index=False, encoding="utf-8")

        # Log artifacts (รวมโฟลเดอร์)
        mlflow.log_artifact(str(train_csv))
        mlflow.log_artifact(str(test_csv))
        mlflow.log_artifacts(str(processed_dir), artifact_path="processed_data")

        # === Save run id ===
        run_id_file.parent.mkdir(parents=True, exist_ok=True)
        run_id_file.write_text(mlflow.active_run().info.run_id, encoding="utf-8")

        print("[OK] Preprocessing complete (hybrid TF-IDF logged).")
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
        return 0
    except Exception as e:
        # Helpful tags/params for CI logs
        mlflow.set_tag("stage", "preprocessing")
        mlflow.set_tag("status", "failed")
        mlflow.log_param("error_type", type(e).__name__)
        mlflow.log_param("error_message", str(e))
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    finally:
        mlflow.end_run()


if __name__ == "__main__":
    sys.exit(main())
