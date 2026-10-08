FROM python:3.12-slim

WORKDIR /app

# Ensure Python output is unbuffered
ENV PYTHONUNBUFFERED=1

# Install system dependencies including ffmpeg for video encoding and cv2 libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight CPU-only PyTorch first to reduce build time & avoid huge CUDA downloads
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY . .

# Expose default port (Render will override via $PORT at runtime)
EXPOSE 8000

# Start Uvicorn dynamically binding to $PORT (defaulting to 8000)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]