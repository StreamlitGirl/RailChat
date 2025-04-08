FROM python:3.11-slim

# Install required system packages for matplotlib and other scientific libs
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    python3-dev \
    libx11-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# Copy rest of your project files
COPY . .

# Run your app (adjust as needed)
CMD ["uvicorn", "API:app", "--host", "0.0.0.0", "--port", "5000"]
