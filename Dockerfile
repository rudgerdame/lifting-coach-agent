FROM python:3.11-slim

# System deps for LightGBM, FAISS, and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Pre-build: generate synthetic data + train model + build FAISS index
# (skipped if volume-mounted artifacts already exist)
RUN python -m ingestion.synthetic --out data/synthetic && \
    python -m models.train --data-dir data/synthetic && \
    python -m index.build

EXPOSE 8000 8501

# Default: run FastAPI. Override CMD to run Streamlit.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
