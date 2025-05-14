FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and HTML
COPY ./src/app.py .
COPY ./src/index.html .

# Create directory for model caching
RUN mkdir -p /root/.keras/models

# Download the MobileNet weights during build for faster startup
RUN python -c "from tensorflow.keras.applications import MobileNet; MobileNet(weights='imagenet')"

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]