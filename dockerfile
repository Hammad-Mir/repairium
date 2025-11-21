FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    gcc \
    g++ \
    git \
    curl \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application code
COPY . .

# Create directories for data persistence
RUN mkdir -p /app/llmware_data /app/chromadb_data

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    LLMWARE_PATH=/app/llmware_data \
    CHROMADB_PERSIST_PATH=/app/chromadb_data

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]