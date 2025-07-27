from __future__ import annotations
import argparse, json, sys, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pdfplumber
import regex as re
from rapidfuzz import fuzz

# ─────────────────────  spaCy  ──────────────────────────────
def _load_spacy():
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", exclude=[])
        for p in ("parser", "attribute_ruler", "lemmatizer"):
            if p in nlp.pipe_names:
                nlp.disable_pipe(p)
        return nlp
    except Exception:
        return None

NLP = _load_spacy()
POS_OK = NLP is not None and "tagger" in getattr(NLP, "pipe_names", [])
NER_OK = NLP is not None and "ner" in getattr(NLP, "pipe_names", [])

# ───────────────────────── Constants ───────────────────────────────────
BULLETS       = {"•", "◦", "‣", "∙", "●", "○", "-", "–", "—", "*", "→", "⇒", "・"}
TITLE_WORDS   = {"title", "report", "study", "analysis", "paper", "article", "overview"}
SECTION_PATTS = [r"^(section|chapter|part)\s+[ivxlcdm\d]+", r"^\d+(?:\.\d+)*\s+"]
MATCH_THRESH  = 80

SCRIPT_RGX = re.compile(
    r"[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}"
    r"\p{Script=Arabic}\p{Script=Cyrillic}\p{Script=Devanagari}]"
)

def _norm_quotes(txt: str) -> str:
    return txt.translate(str.maketrans({
        "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
        "‘": "'", "’": "'", "‚": "'", "‹": "'", "›": "'",
    }))

def _clean(text: str) -> str:
    t = unicodedata.normalize("NFKD", _norm_quotes(text)).lower()
    t = t.replace("\t", " ")
    t = re.sub(r"\p{P}+", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def _clean_variants(text: str) -> list[str]:
    base = _clean(text)
    out  = {base}
    out.add(re.sub(r"^\d+(?:\.\d+)*\s*", "", base).strip())
    out.add(re.sub(r"[:.\-–—]+$", "", base).strip())
    return [v for v in out if v]

def _is_phone(t: str) -> bool:
    return bool(re.search(r"\d{3}[-.\s]\d{3}[-.\s]\d{4}", t))

def _all_caps_words(t: str) -> bool:
    words = [w for w in re.split(r"\s+", t) if w]
    return bool(words) and sum(1 for w in words if w[0].isupper()) >= 0.8 * len(words)

def _is_non_latin(txt: str) -> bool:
    return bool(SCRIPT_RGX.search(txt))


def extract_features_from_pdf(pdf: Path, *, use_nlp: bool = True) -> pd.DataFrame:
    """
    Return a DataFrame of line-level features for a single PDF.
    """
    rows: List[Dict[str, Any]] = []
    nlp = NLP if (use_nlp and NLP) else None

    with pdfplumber.open(str(pdf)) as doc:
        for pg_idx, page in enumerate(doc.pages):
            words = page.extract_words(
                x_tolerance=1,
                y_tolerance=3,
                keep_blank_chars=True,
                use_text_flow=True,
                extra_attrs=["fontname", "size"],  
            )
            if not words:
                continue

            pw, ph = page.width, page.height
            med_size = np.median([w["size"] for w in words]) or 1.0

            line_buckets: Dict[int, List[dict]] = defaultdict(list)
            for w in words:
                line_buckets[w["top"]].append(w)

            prev_bottom = None
            for top in sorted(line_buckets):
                line_words = line_buckets[top]
                raw = "".join(
                    w["text"] for w in sorted(line_words, key=lambda w: w["x0"])
                ).rstrip()
                if not raw:
                    continue

                sizes = np.asarray([w["size"] for w in line_words], dtype=float)
                fonts = [w["fontname"].lower() for w in line_words]
                left = min(w["x0"] for w in line_words)
                right = pw - max(w["x1"] for w in line_words)
                align = "center" if abs(left - right) < 0.15 * pw else (
                    "right" if right < 0.15 * pw else "left"
                )

                indent_depth = round(left / 36)
                spacing_above = (top - prev_bottom) if prev_bottom is not None else 0.0
                prev_bottom = max(w["bottom"] for w in line_words)

                doc_nlp = nlp(raw) if nlp and pg_idx < 2 else None

                rows.append(
                    {
                        "text": raw,
                        "clean": _clean(raw),
                        "page": pg_idx,
                        "y": top,
                        "position_ratio": top / ph,
                        "font_size_avg": float(sizes.mean()),
                        "font_size_max": float(sizes.max()),
                        "font_size_rel": float(sizes.mean() / med_size),
                        "indent_depth": int(indent_depth),
                        "spacing_above": float(spacing_above),
                        "main_fontname": Counter(fonts).most_common(1)[0][0],
                        "bold_pct": float(
                            sum("bold" in f or "bd" in f for f in fonts) / len(fonts)
                        ),
                        "italic_pct": float(
                            sum("italic" in f for f in fonts) / len(fonts)
                        ),
                        "alignment": align,
                        "len_chars": len(raw),
                        "word_count": len(re.split(r"\s+", raw))
                        if not _is_non_latin(raw)
                        else len(raw),
                        "num_digits": sum(c.isdigit() for c in raw),
                        "num_upper": sum(c.isupper() for c in raw),
                        "all_caps_line": int(raw.isupper()),
                        "all_caps_words": int(_all_caps_words(raw)),
                        "starts_bullet": int(raw[:1] in BULLETS),
                        "is_section_hdr": int(
                            any(re.match(p, raw.lower()) for p in SECTION_PATTS)
                        ),
                        "title_kw_ratio": float(
                            sum(w in TITLE_WORDS for w in raw.lower().split())
                            / max(1, len(raw.split()))
                        ),
                        "noun_ratio": (
                            float(
                                sum(t.pos_ in {"NOUN", "PROPN"} for t in doc_nlp)
                                / max(1, len(doc_nlp))
                            )
                            if POS_OK and doc_nlp
                            else 0.0
                        ),
                        "verb_ratio": (
                            float(
                                sum(t.pos_ == "VERB" for t in doc_nlp)
                                / max(1, len(doc_nlp))
                            )
                            if POS_OK and doc_nlp
                            else 0.0
                        ),
                        "ent_org": int(
                            any(e.label_ == "ORG" for e in doc_nlp.ents)
                        )
                        if NER_OK and doc_nlp
                        else 0,
                        "has_url": int("http" in raw.lower()),
                        "has_email": int("@" in raw),
                        "has_phone": int(_is_phone(raw)),
                        "is_non_latin": int(_is_non_latin(raw)),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["alignment"] = df["alignment"].astype("category")
    df["main_fontname"] = df["main_fontname"].astype("category")

    raw_cols = ["text", "clean", "page"]
    return df[raw_cols + sorted(c for c in df.columns if c not in raw_cols)]


# ────────────────  Heuristic merge ─────────────────
def _best_match(clean_s: pd.Series, raw_s: pd.Series, variants: list[str]):
    best_idx, best_score = -1, -1
    for v in variants:
        ts = clean_s.apply(lambda t: fuzz.token_set_ratio(t, v))
        pr = clean_s.apply(lambda t: fuzz.partial_ratio(t, v))
        if int(ts.max()) > best_score:
            best_idx, best_score = int(ts.idxmax()), int(ts.max())
        if int(pr.max()) > best_score:
            best_idx, best_score = int(pr.idxmax()), int(pr.max())

        if len(v.split()) <= 6:
            for idx, c in clean_s.items():
                if v in c and best_score < 100:
                    best_idx, best_score = idx, 100
    pr_raw = raw_s.apply(lambda t: fuzz.partial_ratio(_clean(t), variants[0]))
    if int(pr_raw.max()) > best_score:
        best_idx, best_score = int(pr_raw.idxmax()), int(pr_raw.max())
    return best_idx, best_score

def _apply_dynamic_font_heuristic(feats: pd.DataFrame) -> int:
    unl = feats[feats["label"] == "NONE"]
    if unl.empty:
        return 0

    size_ranks = sorted((unl["font_size_avg"].round(1)).unique(), reverse=True)
    levels = ["H1", "H2", "H3", "H4"]

    added = 0
    for lvl, fs in zip(levels, size_ranks[:4]):
        mask = (feats["label"] == "NONE") & (feats["font_size_avg"].round(1) == fs)
        mask &= feats["len_chars"] < 120
        mask &= ~(feats["has_url"] | feats["has_email"])
        feats.loc[mask, "label"] = lvl
        added += int(mask.sum())
    return added

def _build_training_set(pdf_dir: Path, label_dir: Path, out_csv: Path):
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"[ERROR] No PDFs in {pdf_dir}", file=sys.stderr)
        return

    frames = []
    print(f"[INFO] Processing {len(pdfs)} PDFs …")
    for pdf in pdfs:
        lbl = label_dir / f"{pdf.stem}.json"
        if not lbl.exists():
            print(f"[WARN] {pdf.name}: missing {lbl.name}")
            continue

        feats = extract_features_from_pdf(pdf, use_nlp=False)
        if feats.empty:
            print(f"[WARN] {pdf.name}: no text")
            continue

        truth = json.load(lbl.open(encoding="utf-8"))
        gold = []
        if truth.get("title"):
            gold.append({"page": 0, "variants": _clean_variants(truth["title"]), "label": "TITLE"})
        for o in truth.get("outline", []):
            gold.append({"page": o["page"] - 1, "variants": _clean_variants(o["text"]), "label": o["level"]})

        feats["label"] = "NONE"
        for g in gold:
            pmask = feats["page"] == g["page"]
            if not pmask.any():
                continue
            idx, sc = _best_match(
                feats.loc[pmask, "clean"],
                feats.loc[pmask, "text"],
                g["variants"],
            )
            if sc >= MATCH_THRESH:
                feats.at[idx, "label"] = g["label"]

        added = _apply_dynamic_font_heuristic(feats)
        print(f"  ✓ {pdf.name}: {feats['label'].value_counts().to_dict()} (+{added} via heuristic)")
        frames.append(feats.drop(columns=["text", "clean"]))

    if not frames:
        print("[╳] No data – abort.")
        return

    full = pd.concat(frames, ignore_index=True)
    print("\n[INFO] Final label distribution:")
    print(full["label"].value_counts())
    full.to_csv(out_csv, index=False)
    print(f"[✓] Saved {len(full):,} rows → {out_csv}")

# ───────────────────────────── CLI ─────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="PDF outline feature extractor – v2.1")
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("single", help="Extract features for one PDF")
    s1.add_argument("pdf", type=Path)
    s1.add_argument("-o", "--out-csv", type=Path)
    s1.add_argument("--no-nlp", action="store_true")

    s2 = sub.add_parser("build-train-csv", help="Create train.csv from folders")
    s2.add_argument("pdf_dir", type=Path)
    s2.add_argument("label_dir", type=Path)
    s2.add_argument("out_csv", type=Path)
    return p.parse_args()

def main():
    args = _parse_args()
    if args.cmd == "single":
        df = extract_features_from_pdf(args.pdf, use_nlp=not args.no_nlp)
        if args.out_csv:
            df.to_csv(args.out_csv, index=False)
            print(f"[✓] saved → {args.out_csv}")
        else:
            pd.set_option("display.max_columns", None)
            print(df.head())
    else:
        _build_training_set(args.pdf_dir, args.label_dir, args.out_csv)

if __name__ == "__main__":
    main()
