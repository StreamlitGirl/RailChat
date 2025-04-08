FROM python:3.11-slim

# Install required system packages for building C-based Python packages (like matplotlib)
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

# Copy the requirements file into the container
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# Copy the rest of your application
COPY . .

# Default command to run your application
CMD ["uvicorn", "API:app", "--host", "0.0.0.0", "--port", "5000"]
