# ── Stage 1: builder ──────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps for prophet, psycopg2, tensorflow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install only runtime system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app/ ./app/

# Non-root user for security
RUN useradd -m -u 1001 kast && chown -R kast:kast /app
USER kast

EXPOSE 8000

# Health check — large start-period: TensorFlow + Prophet cold import ~60s
HEALTHCHECK --interval=20s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Use shell form so $PORT env var is expanded at runtime
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
