# =========================================================
# Base image with Python + system dependencies
# =========================================================
FROM python:3.12-bookworm AS base

LABEL authors="dennisfashimpaur"

# Prevent Python from writing .pyc files and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Ensure Python can find your src/ folder
ENV PYTHONPATH="/app/src"

# -----------------------
# Install system dependencies
# -----------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        && rm -rf /var/lib/apt/lists/*

# -----------------------
# Copy and install Python dependencies first (layer caching)
# -----------------------
COPY requirements.txt .
RUN pip install --upgrade pip wheel setuptools
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------
# Copy project code
# -----------------------
COPY src/ src/
COPY models/ models/

# -----------------------
# Create non-root user and switch
# -----------------------
RUN useradd -m appuser
USER appuser

# -----------------------
# Expose FastAPI port
# -----------------------
EXPOSE 8000

# -----------------------
# Default command to run FastAPI
# -----------------------
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
