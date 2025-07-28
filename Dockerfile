# Use Python 3.10 as specified in challenge requirements for AMD64 compatibility
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for PDF processing
RUN apt-get update && apt-get install -y \
  build-essential \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libgomp1 \
  libgraphicsmagick1-dev \
  libatlas-base-dev \
  poppler-utils \
  tesseract-ocr \
  pandoc \
  curl \
  && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install uv

# Copy project configuration files
COPY pyproject.toml uv.lock ./

# Install Python dependencies using uv
RUN uv sync --frozen --no-dev

# Copy application code and models
COPY . .

RUN ls

# Set environment variables for proper execution
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set NLTK_DATA to include your download directory first
ENV NLTK_DATA=/app/nltk_data:/app/.venv/nltk_data:/root/nltk_data

# Create necessary directories
RUN mkdir -p logs output input

# Pre-download NLTK data to avoid runtime download issues
RUN uv run python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng', download_dir='/app/nltk_data'); nltk.download('punkt_tab', download_dir='/app/nltk_data')" || echo "NLTK download failed, will retry at runtime"

# Run the PDF processing script directly (no API server)
CMD ["uv", "run", "main.py"]

