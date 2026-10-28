FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup -u 1000 appuser

WORKDIR /app

# Install Python dependencies first for efficient layer caching
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and project configurations
COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY README.md ./

# Install project in editable/production mode
RUN pip install --no-cache-dir -e .

# Create necessary runtime directories and adjust permissions
RUN mkdir -p data/raw data/processed data/reference reports mlruns && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
