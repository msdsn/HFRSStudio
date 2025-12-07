# Production Dockerfile for HFRS
# Multi-stage build: Frontend + Backend in one container
# Use this for Railway deployment

# ============================================
# Stage 1: Build Frontend
# ============================================
FROM node:20-alpine AS frontend-builder

WORKDIR /frontend

# Copy package files
COPY frontend/package.json frontend/package-lock.json* ./

# Install dependencies
RUN npm ci

# Copy frontend source
COPY frontend/ .

# Build arguments for Vite environment variables
ARG VITE_API_URL=/api
ARG VITE_SUPABASE_URL
ARG VITE_SUPABASE_KEY

# Set environment variables for build (Vite reads these at build time)
ENV VITE_API_URL=$VITE_API_URL
ENV VITE_SUPABASE_URL=$VITE_SUPABASE_URL
ENV VITE_SUPABASE_KEY=$VITE_SUPABASE_KEY

# Build frontend
RUN npm run build

# ============================================
# Stage 2: Python Backend with Frontend
# ============================================
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements first for better caching
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install gdown for downloading from Google Drive
RUN pip install --no-cache-dir gdown

# Install PyTorch and related packages (CPU version for Railway)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install torch-geometric dependencies
RUN pip install --no-cache-dir torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.8.0+cpu.html || \
    pip install --no-cache-dir torch-scatter torch-sparse --index-url https://download.pytorch.org/whl/cpu

# Install torch-geometric
RUN pip install --no-cache-dir torch-geometric

# Copy backend application code
COPY backend/ .

# Copy frontend build from frontend-builder stage to /app/static
COPY --from=frontend-builder /frontend/dist /app/static

# Download data and checkpoints from Google Drive during build
# Checkpoint folder: https://drive.google.com/drive/folders/14TxMYw52lacpJrKBgTCzSoeXPNPiizHp
RUN mkdir -p /tmp/downloads/checkpoints && \
    gdown --folder --id 14TxMYw52lacpJrKBgTCzSoeXPNPiizHp -O /tmp/downloads/checkpoints --quiet && \
    mkdir -p checkpoints && \
    (find /tmp/downloads/checkpoints -mindepth 1 -maxdepth 1 -exec cp -r {} checkpoints/ \; 2>/dev/null || \
     cp -r /tmp/downloads/checkpoints/* checkpoints/ 2>/dev/null || true) && \
    rm -rf /tmp/downloads/checkpoints || true

# Data folder: https://drive.google.com/drive/folders/1Gq4Lmbus51wSa4tSGWbNMoYm8MmTl9Wg
RUN mkdir -p /tmp/downloads/data && \
    gdown --folder --id 1Gq4Lmbus51wSa4tSGWbNMoYm8MmTl9Wg -O /tmp/downloads/data --quiet && \
    mkdir -p data && \
    (find /tmp/downloads/data -mindepth 1 -maxdepth 1 -exec cp -r {} data/ \; 2>/dev/null || \
     cp -r /tmp/downloads/data/* data/ 2>/dev/null || true) && \
    rm -rf /tmp/downloads/data || true

# Expose port (Railway uses PORT env var)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; import os; port = os.getenv('PORT', '8000'); requests.get(f'http://localhost:{port}/api/health')" || exit 1

# Run application (Railway sets PORT env var)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
