FROM --platform=linux/amd64 python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        pdfplumber==0.10.3 \
        pandas==2.2.2 \
        numpy==1.26.4 \
        regex==2023.12.25 \
        rapidfuzz==3.6.2 \
        scikit-learn==1.3.2 \
        xgboost==2.0.3 \
        joblib==1.4.2

ENTRYPOINT ["python", "-u", "infer.py", "/app/input", "/app/output"]
