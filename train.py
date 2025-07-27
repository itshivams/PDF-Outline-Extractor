from __future__ import annotations
import argparse, json, logging, sys
from datetime import datetime
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier

# ───────────────────────── CLI ──────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGB for PDF heading detection (v2.0)")
    p.add_argument("csv", type=Path, help="train.csv from extract_features.py")
    p.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=Path("models"),
        help="directory to store artefacts",
    )
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.1)
    return p.parse_args()


def setup_logging() -> None:
    logging.basicConfig(format="%(levelname)s  %(message)s", level=logging.INFO)

# ───────────────────────── Helpers ───────────────────────────────
DROP_LABEL = "NONE"


def detect_categoricals(df: pd.DataFrame, max_unique: int = 40) -> List[str]:
    out: List[str] = []
    for col in df.columns:
        if col == "label":
            continue
        dt = df[col].dtype
        if str(dt) in ("object", "category"):
            out.append(col)
        elif np.issubdtype(dt, np.integer) and df[col].nunique() <= max_unique:
            out.append(col)
    return out


def encode_features(
    df: pd.DataFrame, categorical_cols: List[str]
) -> tuple[np.ndarray, OneHotEncoder, List[str]]:
    try:
        ohe = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
            dtype=np.float32,
        )
    except TypeError:  
        ohe = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        )

    X_cat = ohe.fit_transform(df[categorical_cols]) if categorical_cols else np.empty(
        (len(df), 0), dtype=np.float32
    )

    numeric_cols = [c for c in df.columns if c not in categorical_cols + ["label"]]
    X_num = df[numeric_cols].astype(np.float32).values

    X = np.hstack([X_num, X_cat])

    onehot_names = list(ohe.get_feature_names_out(categorical_cols)) if categorical_cols else []
    feature_names = numeric_cols + onehot_names
    return X, ohe, feature_names



def make_class_weights(y: np.ndarray) -> np.ndarray:
    uniques, counts = np.unique(y, return_counts=True)
    class_freq = dict(zip(uniques, counts))
    total = float(len(y))
    weights = {c: total / (len(uniques) * cnt) for c, cnt in class_freq.items()}
    return np.asarray([weights[c] for c in y], dtype=np.float32)

# ──────────────────────────── Main ─────────────────────────────────
def main() -> None:
    setup_logging()
    args = parse_args()

    if not args.csv.exists():
        logging.error("CSV file not found: %s", args.csv)
        sys.exit(1)

    df = pd.read_csv(args.csv)

    df = df[df["label"] != DROP_LABEL]
    if df.empty:
        logging.error("No labelled rows after dropping '%s'.", DROP_LABEL)
        sys.exit(1)
    logging.info("Rows with labels: %s", len(df))

    cat_cols = detect_categoricals(df)
    logging.info("Categorical columns: %s", cat_cols)

    X, ohe, feature_names = encode_features(df, cat_cols)

    le = LabelEncoder()
    y = le.fit_transform(df["label"].astype(str))
    classes = list(le.classes_)
    logging.info("Classes: %s", classes)

    sample_weight = make_class_weights(y)

    if len(df) < 120:
        cv = StratifiedShuffleSplit(n_splits=8, test_size=0.25, random_state=42)
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    y_pred_cv = np.empty_like(y)
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        clf = XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=8,
            random_state=42 + fold,
        )
        clf.fit(
            X[train_idx],
            y[train_idx],
            sample_weight=sample_weight[train_idx],
        )
        y_pred_cv[test_idx] = clf.predict(X[test_idx])

    acc = accuracy_score(y, y_pred_cv)
    macro_f1 = f1_score(y, y_pred_cv, average="macro")
    logging.info("CV Accuracy = %.3f | Macro‑F1 = %.3f", acc, macro_f1)
    logging.info("\n" + classification_report(y, y_pred_cv, target_names=classes))

    final_clf = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=8,
        random_state=42,
    )
    final_clf.fit(X, y, sample_weight=sample_weight)

    # ── Save artefacts ────────────────────────────────────────────────
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = out_dir / f"model_{ts}.pkl"
    joblib.dump(final_clf, model_path)

    label_path = out_dir / "label_encoder.pkl"
    joblib.dump(le, label_path)

    ohe_path = out_dir / "onehot_encoder.pkl"
    joblib.dump(ohe, ohe_path)

    feature_path = out_dir / "feature_list.json"
    feature_path.write_text(json.dumps(feature_names, indent=2))

    # metrics 
    cm = confusion_matrix(y, y_pred_cv).tolist()
    metrics_json = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "classes": classes,
        "confusion_matrix": cm,
    }
    (out_dir / f"metrics_{ts}.json").write_text(json.dumps(metrics_json, indent=2))

    logging.info("✓ Training complete – artefacts saved to %s/", out_dir)


if __name__ == "__main__":
    main()
