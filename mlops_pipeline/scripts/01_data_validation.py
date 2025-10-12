#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_data_validation.py
- Loads McDonald_s_Reviews.csv from the project root (encoding="latin-1")
- Drops rows with any nulls
- Extracts integer 'Star' rating from the 'rating' column
- Logs dataset info & validation metrics to MLflow (tracking at ./mlruns)
- Saves cleaned dataset to <project_root>/processed_data/validated_data.csv
"""

import os
import sys
import io
from datetime import datetime
from pathlib import Path
import pandas as pd
import mlflow


def main() -> int:
    # === Paths ===
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent  # go up from scripts/
    csv_path = project_root / "McDonald_s_Reviews.csv"
    processed_dir = project_root / "processed_data"   # << fixed location
    processed_dir.mkdir(parents=True, exist_ok=True)

    tracking_dir = project_root / "mlruns"
    tracking_dir.mkdir(parents=True, exist_ok=True)

    # === MLflow setup ===
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment("MLOps_Pipeline_Experiment")
    run = mlflow.start_run(run_name="DataValidation")

    try:
        dataset_name = "McDonald_s_Reviews.csv"
        mlflow.log_param("dataset_name", dataset_name)

        # === Load CSV ===
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found at {csv_path}")

        df = pd.read_csv(csv_path, encoding="latin-1")
        rows_before = len(df)

        # === Validation ===
        if "review" not in df.columns or "rating" not in df.columns:
            raise ValueError("CSV must contain 'review' and 'rating' columns.")

        # === Drop nulls ===
        df_clean = df.dropna().copy()
        rows_after_dropna = len(df_clean)
        mlflow.log_metric("rows_before_dropna", rows_before)
        mlflow.log_metric("rows_after_dropna", rows_after_dropna)

        # === Extract 'Star' ===
        stars = (
            df_clean["rating"]
            .astype(str)
            .str.extract(r"(\d)", expand=False)
            .pipe(pd.to_numeric, errors="coerce")
        )

        df_clean = df_clean.assign(Star=stars).dropna(subset=["Star"])
        df_clean["Star"] = df_clean["Star"].astype("int64")
        df_clean = df_clean.loc[df_clean["Star"].between(1, 5)].copy()
        rows_final = len(df_clean)

        # === Summary report ===
        summary_text = io.StringIO()
        print("=== Data Validation Summary ===", file=summary_text)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", file=summary_text)
        print(f"Rows before dropna: {rows_before}", file=summary_text)
        print(f"Rows after dropna:  {rows_after_dropna}", file=summary_text)
        print(f"Rows final (1–5 Star): {rows_final}", file=summary_text)
        print("", file=summary_text)
        print("Star Distribution:", file=summary_text)
        print(df_clean["Star"].value_counts().sort_index(), file=summary_text)

        summary_path = processed_dir / "validation_summary.txt"
        summary_path.write_text(summary_text.getvalue(), encoding="utf-8")
        mlflow.log_artifact(str(summary_path))

        # === Save cleaned data ===
        output_csv = processed_dir / "validated_data.csv"
        df_clean.to_csv(output_csv, index=False, encoding="utf-8")
        mlflow.log_artifact(str(output_csv))

        print(f"[OK] Validation complete.")
        print(f"  → Cleaned CSV: {output_csv}")
        print(f"  → Report: {summary_path}")
        return 0

    except Exception as e:
        mlflow.set_tag("data_validation_status", "failed")
        mlflow.log_param("error_type", type(e).__name__)
        mlflow.log_param("error_message", str(e))
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    finally:
        mlflow.end_run()


if __name__ == "__main__":
    sys.exit(main())
