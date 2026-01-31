# Python base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app \
    PORT=5000

# Set working directory
WORKDIR /app

# Install system dependencies required for scientific stack and Matplotlib rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libomp-dev \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first to leverage Docker layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . ./

# Expose Flask port
EXPOSE 5000

# Start the Flask server (honours Azure Container Apps PORT environment variable)
CMD ["sh", "-c", "python -m flask --app app run --host=0.0.0.0 --port=${PORT:-5000}"]