# Backend Dockerfile for HFRS API
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch and related packages (CPU version for Railway)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install torch-geometric dependencies
RUN pip install --no-cache-dir torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.8.0+cpu.html || \
    pip install --no-cache-dir torch-scatter torch-sparse --index-url https://download.pytorch.org/whl/cpu

# Install torch-geometric
RUN pip install --no-cache-dir torch-geometric

# Copy application code
COPY . .

# Copy data files
COPY data/ ./data/
COPY checkpoints/ ./checkpoints/ || true

# Expose port (Railway uses PORT env var)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; import os; port = os.getenv('PORT', '8000'); requests.get(f'http://localhost:{port}/api/health')" || exit 1

# Run application (Railway sets PORT env var)
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}

