# 📄 PDF Outline Extractor (Adobe Hackathon Challenge-1A)

This project intelligently predicts the **structure of a PDF** (Title, H1–H4) using machine‑learning and robust PDF parsing.

---

## 🚀 Features

* Predicts heading labels via an **XGBoost** classifier
* Outputs clean JSON outlines ready for sem‑search / downstream tasks
* Ships as a lightweight, offline **Docker** image (≤ 200 MB, AMD64)

---

## 📂 Project Structure

```text
Adobe-India-Hackathon25/
├── extract_features.py   
├── infer.py              # runtime prediction pipeline
├── train.py              # model training script 
├── Dockerfile            # Docker container definition
├── .dockerignore         # trims build context
├── models/               # ⇢ trained model + encoders
│   ├── model_<ts>.pkl
│   ├── label_encoder.pkl
│   ├── onehot_encoder.pkl
│   └── feature_list.json
├── input/                # place PDFs here when running
└── output/               # JSON results appear here
```

---

## 🐳 Run via Docker

### 1 · Build Image

```bash
docker build -t pdf-outline .
```

### 2 · Prepare Folders

```bash
mkdir -p input output
# copy your PDFs into ./input
```

### 3 · Run Inference

```bash
docker run --rm \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  --network none \
  pdf-outline
```

> Every `*.pdf` in **input/** produces a `*.json` in **output/**.

---

## 💻 Local (Non‑Docker) Run

```bash
pip install pdfplumber pandas numpy regex rapidfuzz scikit-learn xgboost joblib
python infer.py input/ output/
```

---

## ✅ JSON Schema

```jsonc
{
  "title": "Document Title",
  "outline": [
    { "level": "H1", "text": "Section 1", "page": 1 },
    { "level": "H2", "text": "Subsection", "page": 2 }
  ]
}
```

---

## 📌 Implementation Notes

* **Feature set** – font size, position, indent, spacing, bold/italic %, etc.
* **Model** – XGBoost multi‑class (TITLE, H1‑H4) with class‑weighted loss.
* **Post‑processing** – fixes curly quotes, ensures 1‑based pages, fills title fallback.
* **Constraints met** – CPU‑only, offline, AMD64, runtime < 10 s on 50‑page PDF.

---
## Our Team
We are a cross-functional team of machine learning engineers, NLP researchers, full-stack developers, and software architects passionate about document intelligence. Our mission is to make complex document structures easily interpretable by building accurate, scalable, and user-friendly PDF outline extraction systems powered by AI.


- [Shivam](https://github.com/myselfshivams)
- [Ritik Gupta](https://github.com/ritikgupta06)
- [Sanskar Soni](https://github.com/sunscar-sony)


## GitHub Repository
You can find the complete source code to the project on GitHub:
[GitHub Repository](https://github.com/itshivams/PDF-Outline-Extractor/)

## Acknowledgment
Special thanks to Adobe India for organizing this hackathon.
