# Use Python 3.10 (which includes distutils)
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies and Python packages
COPY requirements.txt .
RUN apt-get update && \
    apt-get install -y gcc && \
    pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy project files into container
COPY . .

# Expose the app port
EXPOSE 5000

# Run your API app with Uvicorn
CMD ["uvicorn", "API:app", "--host", "0.0.0.0", "--port", "5000"]
