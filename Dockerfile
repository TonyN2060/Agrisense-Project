# ==============================================================================
# AgriSense ML Module - Production Dockerfile
# 
# Multi-stage build for optimized production deployment
# Target: <100ms prediction latency, <2GB image size
# ==============================================================================

# ==============================================================================
# Stage 1: Build stage
# ==============================================================================
FROM python:3.11-slim-bookworm AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
WORKDIR /install
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ==============================================================================
# Stage 2: Production stage
# ==============================================================================
FROM python:3.11-slim-bookworm AS production

# Labels
LABEL maintainer="AgriSense Team" \
      version="1.0.0" \
      description="AgriSense ML Prediction Service"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    # Application settings
    MODEL_DIR=/app/models \
    JWT_SECRET_KEY=change-me-in-production \
    # Uvicorn settings
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000 \
    UVICORN_WORKERS=2 \
    UVICORN_LOOP=uvloop \
    UVICORN_HTTP=httptools

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user
RUN groupadd -r agrisense && useradd -r -g agrisense agrisense

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=agrisense:agrisense physics_engine.py .
COPY --chown=agrisense:agrisense digital_twin.py .
COPY --chown=agrisense:agrisense synthetic_data.py .
COPY --chown=agrisense:agrisense feature_engineering.py .
COPY --chown=agrisense:agrisense hierarchical_model.py .
COPY --chown=agrisense:agrisense explainability.py .
COPY --chown=agrisense:agrisense api_service.py .
COPY --chown=agrisense:agrisense config.py .

# Copy existing modules (if needed)
COPY --chown=agrisense:agrisense environment.py .
COPY --chown=agrisense:agrisense model.py .
COPY --chown=agrisense:agrisense spoilage_engine.py .
COPY --chown=agrisense:agrisense ingestion_preprocessing.py .

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/logs && \
    chown -R agrisense:agrisense /app

# Copy pre-trained models if available
# COPY --chown=agrisense:agrisense models/ /app/models/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${UVICORN_PORT}/v1/health || exit 1

# Switch to non-root user
USER agrisense

# Expose port
EXPOSE ${UVICORN_PORT}

# Start command
CMD ["sh", "-c", "uvicorn api_service:app --host ${UVICORN_HOST} --port ${UVICORN_PORT} --workers ${UVICORN_WORKERS}"]

# ==============================================================================
# Stage 3: Development stage (optional)
# ==============================================================================
FROM production AS development

# Switch back to root for dev dependencies
USER root

# Install development dependencies
RUN pip install \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    ruff \
    mypy \
    httpx

# Copy test files
COPY --chown=agrisense:agrisense tests/ /app/tests/

# Switch back to agrisense user
USER agrisense

# Override command for development
CMD ["uvicorn", "api_service:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
