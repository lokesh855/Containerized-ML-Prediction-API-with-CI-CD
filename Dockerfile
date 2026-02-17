# Stage 1: Build environment for installing dependencies
FROM python:3.9-slim-buster AS builder

WORKDIR /app

# Copy only requirements.txt first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Stage 2: Minimal runtime environment
FROM python:3.9-slim-buster

WORKDIR /app

# Copy installed Python packages
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Copy application source code ONLY
COPY src/ ./src/

# Expose API port
EXPOSE 8000

# Environment variables
ENV MODEL_PATH=/app/models/my_classifier_model.h5
ENV LOG_LEVEL=INFO

# Start FastAPI app
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
