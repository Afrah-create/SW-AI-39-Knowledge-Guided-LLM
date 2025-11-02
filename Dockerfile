FROM python:3.11-slim

WORKDIR /app

# Install system dependencies and clean up in the same layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir $(grep -v '^torch' requirements.txt) && \
    pip cache purge

# Copy application code
COPY . .

# Expose port (can be overridden by cloud platform)
EXPOSE 8080

# Set default environment variables
ENV PORT=8080
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Run the application
CMD exec gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 4 --threads 8 --timeout 120 app.main:app
