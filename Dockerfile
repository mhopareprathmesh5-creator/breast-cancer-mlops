# Base image: Debian Linux with Python 3.11 already installed.
# "slim" = a minimal version (no compilers, docs, etc.) -> smaller and safer.
# 3.11 matches the venv, so the code runs on the same Python everywhere.
FROM python:3.11-slim

# Environment settings for every command in the image:
#   PYTHONDONTWRITEBYTECODE=1  don't write .pyc files (useless in a container)
#   PYTHONUNBUFFERED=1         print logs immediately (visible in `docker logs`)
#   PIP_NO_CACHE_DIR=1         don't keep pip's download cache -> smaller image
#   PIP_DISABLE_PIP_VERSION_CHECK=1   skip the "new pip available" check
#   MLFLOW_DISABLE_AGENT_HINT=1       hide MLflow's agent hint message
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

# All following commands run in /app (created automatically).
WORKDIR /app

# Install dependencies BEFORE copying the code. Docker caches each step: as
# long as requirements.txt doesn't change, this slow step is reused, and a
# code change only rebuilds the fast COPY steps below.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy only the code the app needs (the rest is excluded by .dockerignore).
COPY src/ src/
COPY api/ api/
COPY monitoring/ monitoring/
COPY scripts/ scripts/

# Security: don't run as root. Create a normal user and give it the folders
# it must write to: /app/logs (prediction log), /app/reports (drift reports)
# and /mlflow (MLflow data).
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/logs /app/reports /mlflow \
    && chown appuser:appuser /app/logs /app/reports /mlflow
USER appuser

# Documents that the API listens on port 8000 (the port is opened by `-p`).
EXPOSE 8000

# Default command when a container starts: the API, production style
# (no --reload; 0.0.0.0 so it's reachable from outside the container).
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]