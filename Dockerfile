FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt .
COPY setup.py .
COPY src/ src/
COPY README.md .
COPY LICENSE .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install anomalyflow package
RUN pip install --no-cache-dir -e .

EXPOSE 5000

CMD ["python", "-c", "from src.ensemble import EnsembleAnomalyDetector; print('AnomalyFlow Ready!')"]
