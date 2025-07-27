from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd


from extract_features import extract_features_from_pdf, _norm_quotes 

# ────────────── Load artefacts ───────────────────────────────
HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"
if not MODEL_DIR.exists():
    sys.stderr.write("╳  models/ directory not found next to infer.py\n")
    sys.exit(1)

model_paths = sorted(MODEL_DIR.glob("model_*.pkl"))
if not model_paths:
    sys.stderr.write("╳  No model_*.pkl found in models/\n")
    sys.exit(1)

MODEL = joblib.load(model_paths[-1])
LABEL_ENCODER = joblib.load(MODEL_DIR / "label_encoder.pkl")
ONEHOT = joblib.load(MODEL_DIR / "onehot_encoder.pkl")
FEATURE_LIST: List[str] = json.loads((MODEL_DIR / "feature_list.json").read_text())

CATEGORICAL_COLS: List[str] = ONEHOT.feature_names_in_.tolist()

# ────────────── Helpers ───────────────────────────────────────────────
def encode_features(feats: pd.DataFrame) -> np.ndarray:
    X_cat = (
        ONEHOT.transform(feats[CATEGORICAL_COLS])
        if CATEGORICAL_COLS
        else np.empty((len(feats), 0), dtype=np.float32)
    )

    numeric_cols = [
        c
        for c in feats.columns
        if c not in CATEGORICAL_COLS + ["label", "text", "clean"]
    ]
    X_num = feats[numeric_cols].astype(np.float32).values
    X_full = np.hstack([X_num, X_cat])

    if X_full.shape[1] != len(FEATURE_LIST):
        raise RuntimeError(
            f"Feature mismatch: {X_full.shape[1]} vs {len(FEATURE_LIST)}"
        )
    return X_full


def build_outline(df: pd.DataFrame) -> Dict:
    title_row = df[df["pred"] == "TITLE"].head(1)
    if title_row.empty:
        title_row = df.iloc[[df["font_size_avg"].idxmax()]]
    title_text = _norm_quotes(title_row["text"].iloc[0])

    outline: List[Dict] = []
    for _, row in df[df["pred"].isin({"H1", "H2", "H3", "H4"})].iterrows():
        outline.append(
            {
                "level": row["pred"],
                "text": _norm_quotes(row["text"]),
                "page": int(row["page"] + 1),
            }
        )
    return {"title": title_text, "outline": outline}


def process_single_pdf(pdf_path: Path) -> Dict:
    feats = extract_features_from_pdf(pdf_path, use_nlp=False)
    if feats.empty:
        return {"title": "", "outline": []}

    X = encode_features(feats)
    preds = LABEL_ENCODER.inverse_transform(MODEL.predict(X))
    feats = feats.assign(pred=preds)

    return build_outline(feats)


# ────────────── CLI ────────────────────────────────────────
def main() -> None:
    if len(sys.argv) != 3:
        sys.stderr.write("Usage: python infer.py /input_dir /output_dir\n")
        sys.exit(1)

    input_dir = Path(sys.argv[1]).expanduser().resolve()
    output_dir = Path(sys.argv[2]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        sys.stderr.write(f"No PDFs found in {input_dir}\n")
        sys.exit(1)

    for pdf in pdf_files:
        res = process_single_pdf(pdf)
        out_path = output_dir / f"{pdf.stem}.json"
        out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2))
        print(f"✓ {pdf.name} → {out_path.name}")

    print(f"\nDone. {len(pdf_files)} file(s) processed.")


if __name__ == "__main__":
    main()
