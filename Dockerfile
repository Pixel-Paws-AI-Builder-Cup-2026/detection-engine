FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for PyTorch & OpenCV/Pillow
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run FastAPI using Uvicorn on port 8080 (Cloud Run default)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]