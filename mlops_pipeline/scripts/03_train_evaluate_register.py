#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, warnings, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
import joblib
import mlflow
import mlflow.sklearn
from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException

EVAL_THRESHOLD = 0.85
EXPERIMENT_NAME = "MLOps_Pipeline_Experiment"
MODEL_NAME = "mcd-review-classifier"

TEXT_COL_CANDIDATES = ["clean_review", "review", "text", "message", "content"]
TARGET_COL_CANDIDATES = ["Star", "rating", "target", "label", "y"]

def parse_args(argv):
    if len(argv) != 3:
        raise SystemExit(
            "Usage: python mlops_pipeline/scripts/03_train_evaluate_register.py <preprocessing_run_id> <grid|auto|single>\n"
            "Examples:\n"
            "  ... 03_train_evaluate_register.py abc123 auto\n"
            "  ... 03_train_evaluate_register.py abc123 0.1,0.5,1,2,5\n"
            "  ... 03_train_evaluate_register.py abc123 1.5"
        )
    preprocessing_run_id = argv[1].strip()
    raw = argv[2].strip().lower()
    if raw == "auto":
        grid = [0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    elif "," in raw:
        try:
            grid = [float(x) for x in raw.split(",")]
        except Exception:
            raise SystemExit("Error: Could not parse comma-separated grid.")
        if any(g <= 0 for g in grid): raise SystemExit("Error: All grid values must be positive.")
    else:
        try:
            v = float(raw); 
            if v <= 0: raise ValueError
            grid = [v]
        except Exception:
            raise SystemExit("Error: Provide positive float, comma-separated list, or 'auto'.")
    return preprocessing_run_id, grid

def _paths_exist(base: Path, names):
    return [base / n for n in names], all((base / n).exists() for n in names)

def _detect_cols(df: pd.DataFrame):
    text_col = next((c for c in TEXT_COL_CANDIDATES if c in df.columns), None)
    target_col = next((c for c in TARGET_COL_CANDIDATES if c in df.columns), None)
    if text_col is None or target_col is None:
        raise ValueError(f"Could not detect text/target columns. Found: {list(df.columns)}")
    return text_col, target_col

def load_artifacts_any(preprocessing_run_id: str, project_root: Path):
    try:
        dl = Path(download_artifacts(run_id=preprocessing_run_id, artifact_path="processed_data"))
        print(f"[Artifacts] Using MLflow: {dl}")
        vec_files = ["X_train.npz", "X_test.npz", "y_train.npy", "y_test.npy"]
        vec_paths, ok_vec = _paths_exist(dl, vec_files)
        if ok_vec:
            return {"format":"vectorized","X_train":vec_paths[0],"X_test":vec_paths[1],
                    "y_train":vec_paths[2],"y_test":vec_paths[3],"vectorizer":dl/"vectorizer.pkl"}
        csv_files = ["train.csv", "test.csv"]
        csv_paths, ok_csv = _paths_exist(dl, csv_files)
        if ok_csv:
            return {"format":"csv","train":csv_paths[0],"test":csv_paths[1],"vectorizer":dl/"vectorizer.pkl"}
    except (MlflowException, FileNotFoundError) as e:
        print(f"[Warn] MLflow artifacts unavailable: {e}")

    local = project_root / "processed_data"
    print(f"[Artifacts] Falling back to local: {local}")
    vec_files = ["X_train.npz", "X_test.npz", "y_train.npy", "y_test.npy"]
    vec_paths, ok_vec = _paths_exist(local, vec_files)
    if ok_vec:
        return {"format":"vectorized","X_train":vec_paths[0],"X_test":vec_paths[1],
                "y_train":vec_paths[2],"y_test":vec_paths[3],"vectorizer":local/"vectorizer.pkl"}
    csv_files = ["train.csv", "test.csv"]
    csv_paths, ok_csv = _paths_exist(local, csv_files)
    if ok_csv:
        return {"format":"csv","train":csv_paths[0],"test":csv_paths[1],"vectorizer":local/"vectorizer.pkl"}
    raise FileNotFoundError("Missing artifacts (.npz/.npy or train.csv/test.csv)")

def eval_predictions(y_true, y_pred):
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")

def ensure_feature_pipeline(cache):
    """ถ้าไม่มี vectorizer.pkl ให้สร้าง hybrid word+char ตรงนี้ (ใช้กับ CSV schema เท่านั้น)"""
    if cache.get("vectorizer"):
        return cache["vectorizer"]
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1,3), min_df=3, max_df=0.95,
                                 sublinear_tf=True, norm="l2", lowercase=False, token_pattern=r"(?u)\b\w+\b")),
        ("char", TfidfVectorizer(analyzer="char", ngram_range=(3,6), min_df=3,
                                 sublinear_tf=True, norm="l2", lowercase=False)),
    ])

def dump_eval_artifacts(prefix, y_true, y_pred):
    rep = classification_report(y_true, y_pred, digits=4, output_dict=True)
    cm  = confusion_matrix(y_true, y_pred)
    # save report json
    report_path = f"{prefix}_cls_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    mlflow.log_artifact(report_path)
    # save cm csv
    cm_path = f"{prefix}_confusion_matrix.csv"
    pd.DataFrame(cm).to_csv(cm_path, index=False)
    mlflow.log_artifact(cm_path)

def main():
    preprocessing_run_id, grid = parse_args(sys.argv)

    project_root = Path(__file__).resolve().parents[2]
    tracking_dir = project_root / "mlruns"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="ModelTraining") as parent:
        parent_id = parent.info.run_id
        mlflow.set_tag("ml.step", "training")
        mlflow.set_tag("preprocessing_run_id", preprocessing_run_id)

        schema = load_artifacts_any(preprocessing_run_id, project_root)

        cache = {}
        if schema["format"] == "vectorized":
            cache["X_train"] = load_npz(schema["X_train"])
            cache["X_test"]  = load_npz(schema["X_test"])
            cache["y_train"] = np.load(schema["y_train"])
            cache["y_test"]  = np.load(schema["y_test"])
            if schema["vectorizer"].exists():
                try: cache["vectorizer"] = joblib.load(schema["vectorizer"])
                except Exception as e: warnings.warn(f"vectorizer.pkl load failed: {e}")
        else:
            tr = pd.read_csv(schema["train"]); te = pd.read_csv(schema["test"])
            text_col, target_col = _detect_cols(tr)
            tr = tr[[text_col, target_col]].dropna()
            te = te[[text_col, target_col]].dropna()
            cache.update({"train_df":tr, "test_df":te, "text_col":text_col, "target_col":target_col})
            if schema["vectorizer"].exists():
                try: cache["vectorizer"] = joblib.load(schema["vectorizer"])
                except Exception as e: warnings.warn(f"vectorizer.pkl load failed: {e}")

        # ==== candidates ====
        best = {"name": None, "params": None, "acc": -1, "f1": -1, "model": None, "tag": None, "y_pred": None}
        def update_best(name, params, acc, f1, model, tag, y_pred):
            nonlocal best
            order = {"linear_svc":0, "ridge":1, "logreg":2, "sgd_log":3, "cnb":4}
            better = (f1 > best["f1"]) or \
                     (np.isclose(f1, best["f1"]) and acc > best["acc"]) or \
                     (np.isclose(f1, best["f1"]) and np.isclose(acc, best["acc"]) and (best["name"] is None or order[name] < order[best["name"]]))
            if better:
                best = {"name":name,"params":params,"acc":acc,"f1":f1,"model":model,"tag":tag,"y_pred":y_pred}

        # helper: train/eval one candidate
        def run_candidate(name, params):
            with mlflow.start_run(run_name=f"{name}-{params}", nested=True):
                if schema["format"] == "vectorized":
                    X_tr, X_te = cache["X_train"], cache["X_test"]
                    y_tr, y_te = cache["y_train"], cache["y_test"]
                    if name == "linear_svc":
                        clf = LinearSVC(C=params["C"], class_weight="balanced", max_iter=10000, random_state=42)
                    elif name == "ridge":
                        clf = RidgeClassifier(alpha=params["alpha"], class_weight="balanced", random_state=42)
                    elif name == "logreg":
                        clf = LogisticRegression(C=params["C"], solver="liblinear", multi_class="ovr",
                                                 class_weight="balanced", max_iter=500, random_state=42)
                    elif name == "sgd_log":
                        clf = SGDClassifier(loss="log_loss", alpha=params["alpha"], class_weight="balanced", random_state=42)
                    elif name == "cnb":
                        clf = ComplementNB(alpha=params["alpha"])
                    else:
                        raise ValueError(name)
                    clf.fit(X_tr, y_tr)
                    y_pred = clf.predict(X_te)
                    acc, f1m = eval_predictions(y_te, y_pred)
                    if cache.get("vectorizer"):
                        model = Pipeline([("vectorizer", cache["vectorizer"]), ("clf", clf)])
                        tag = f"pipeline(vectorizer+{name})"
                    else:
                        model = clf; tag = "classifier_only"
                else:
                    tr, te = cache["train_df"], cache["test_df"]
                    xtr, ytr = tr[cache["text_col"]], tr[cache["target_col"]]
                    xte, yte = te[cache["text_col"]], te[cache["target_col"]]
                    feats = ensure_feature_pipeline(cache)
                    if name == "linear_svc":
                        clf = LinearSVC(C=params["C"], class_weight="balanced", max_iter=10000, random_state=42)
                    elif name == "ridge":
                        clf = RidgeClassifier(alpha=params["alpha"], class_weight="balanced", random_state=42)
                    elif name == "logreg":
                        clf = LogisticRegression(C=params["C"], solver="liblinear", multi_class="ovr",
                                                 class_weight="balanced", max_iter=500, random_state=42)
                    elif name == "sgd_log":
                        clf = SGDClassifier(loss="log_loss", alpha=params["alpha"], class_weight="balanced", random_state=42)
                    elif name == "cnb":
                        clf = ComplementNB(alpha=params["alpha"])
                    pipe = Pipeline([("feats", feats), ("clf", clf)])
                    pipe.fit(xtr, ytr)
                    y_pred = pipe.predict(xte)
                    acc, f1m = eval_predictions(yte, y_pred)
                    model = pipe; tag = f"pipeline(hybrid+{name})"

                mlflow.log_param("candidate_model", name)
                for k,v in params.items(): mlflow.log_param(k, v)
                mlflow.log_metric("test_accuracy", acc); mlflow.log_metric("test_f1_macro", f1m)
                print(f"[{name} {params}] acc={acc:.4f}, f1={f1m:.4f}")
                update_best(name, params, acc, f1m, model, tag, y_pred)

        # grids
        C_grid     = grid
        alpha_grid = [1.0/max(C_grid), 1e-3, 1e-4, 1e-5] if len(C_grid)>1 else [1.0/max(C_grid), 1e-3, 1e-4]
        # candidates
        for C in C_grid:     run_candidate("linear_svc", {"C": float(C)})
        for a in alpha_grid: run_candidate("ridge",      {"alpha": float(a)})
        for C in C_grid:     run_candidate("logreg",     {"C": float(C)})
        for C in C_grid:     run_candidate("sgd_log",    {"alpha": float(1.0/(C*1000.0))})
        for C in C_grid:     run_candidate("cnb",        {"alpha": float(1.0/C)})

        # log best to parent
        assert best["model"] is not None
        mlflow.set_tag("best_model_type", best["tag"])
        mlflow.log_param("best_candidate_model", best["name"])
        for k,v in (best["params"] or {}).items(): mlflow.log_param(f"best_{k}", v)
        mlflow.log_metric("best_test_accuracy", best["acc"])
        mlflow.log_metric("best_test_f1_macro", best["f1"])

        # dump eval artifacts
        if schema["format"] == "vectorized":
            y_true = cache["y_test"]
        else:
            y_true = cache["test_df"][cache["target_col"]].values
        dump_eval_artifacts("eval_best", y_true, best["y_pred"])

        artifact_path = "model"
        mlflow.sklearn.log_model(best["model"], artifact_path=artifact_path)
        model_uri = f"runs:/{parent.info.run_id}/{artifact_path}"

        if best["acc"] > EVAL_THRESHOLD:
            print(f"[Register] Best acc {best['acc']:.4f} > {EVAL_THRESHOLD:.2f}. Registering '{MODEL_NAME}' ...")
            registered = mlflow.register_model(model_uri, MODEL_NAME)
            mlflow.set_tag("registration_status", "submitted")
            print(f"[OK] Registered version {registered.version}")
        else:
            print(f"[Skip] Best acc {best['acc']:.4f} <= {EVAL_THRESHOLD:.2f}. Not registering.")
            mlflow.set_tag("registration_status", "skipped_below_threshold")

        print(f"[Done] Best: {best['name']} {best['params']} | acc={best['acc']:.4f}, f1={best['f1']:.4f}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
