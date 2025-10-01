FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias para olmocr
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de la API
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar aplicación FastAPI
COPY app/ .

# Ejecutar directamente con uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
