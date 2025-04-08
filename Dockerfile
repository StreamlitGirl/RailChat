FROM python:3.12-slim

# System dependencies for matplotlib & building packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    python3-dev \
    libglib2.0-0 \
    libxext6 \
    libxrender-dev \
    libsm6 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy files
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# Copy the rest of the code
COPY . .

# Command to run your app
CMD ["uvicorn", "API:app", "--host", "0.0.0.0", "--port", "5000"]
