# Use official Python runtime as base image
FROM python:3.12-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY train.py .
COPY predict.py .
COPY model/ model/

# Expose port for Flask app
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=predict.py
ENV PYTHONUNBUFFERED=1

# Run the Flask application
CMD ["python", "predict.py"]