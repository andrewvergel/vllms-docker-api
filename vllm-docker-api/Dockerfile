FROM alleninstituteforai/olmocr:latest

WORKDIR /app

# Instalar dependencias adicionales para la API
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar aplicación FastAPI
COPY app/ .

# Ejecutar directamente con uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
