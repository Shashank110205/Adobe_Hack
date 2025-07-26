FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
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

# Install uv (Python package installer)
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock ./

# Install Python dependencies using uv
RUN uv sync --frozen

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs output

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uv", "run", "api.py"]

