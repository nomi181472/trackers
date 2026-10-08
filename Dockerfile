FROM python:3.12-slim

WORKDIR /app

# Ensure Python unbuffered mode
ENV PYTHONUNBUFFERED=1

# Install system dependencies including ffmpeg for video generation and OpenCV support
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install dependencies
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the backend source code
COPY backend/ /app/

# Expose default port
EXPOSE 8000

# Start Uvicorn dynamically binding to Render's $PORT (or fallback 8000)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
