# DotsOCR vLLM Model

This directory contains the configuration for running the DotsOCR model using vLLM on a GPU-enabled machine.

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose installed
- NVIDIA GPU with drivers installed
- Sufficient GPU memory (8GB+ recommended)

### Run the Model

```bash
# From the project root directory
cd vllm-models/dotsocr

# Configure for your GPU (optional)
nano .env

# Start the vLLM server
docker-compose up -d

# View logs in real-time
docker-compose logs -f ${CONTAINER_NAME}

# Stop the server
docker-compose down
```

### Verification

Once started, you can verify the server is working:

```bash
# Health check
curl http://localhost:${VLLM_PORT}/health

# List available models
curl http://localhost:${VLLM_PORT}/v1/models

# View model information
curl http://localhost:${VLLM_PORT}/v1/models/${VLLM_SERVED_MODEL_NAME}
```

### Validación del Modelo

Una vez que el servidor esté funcionando, puedes validar que el modelo DotsOCR responde correctamente con estos comandos curl:

#### **Uso Rápido - Script Automatizado**
```bash
cd vllm-models/dotsocr
./curl_examples.sh ${VLLM_PORT}
```

#### **Ejemplos Individuales**

Una vez que el servidor esté funcionando, puedes validar que el modelo DotsOCR responde correctamente con estos comandos curl:

#### **Ejemplo 1: Análisis de Imagen (Recomendado)**

```bash
curl -X POST "http://localhost:${VLLM_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dotsocr",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe this image in one sentence."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
            }
          }
        ]
      }
    ],
    "max_tokens": 100,
    "temperature": 0.1
  }'
```

#### **Ejemplo 2: Pregunta Simple sobre Imagen**

```bash
curl -X POST "http://localhost:${VLLM_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dotsocr",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "¿Qué ves en esta imagen? Responde en español."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://images.unsplash.com/photo-1544568100-847a948585b9?w=800"
            }
          }
        ]
      }
    ],
    "max_tokens": 150,
    "temperature": 0.2
  }'
```

#### **Ejemplo 3: Completions Básico**

```bash
curl -X POST "http://localhost:${VLLM_PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dotsocr",
    "prompt": "The future of artificial intelligence",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

#### **Ejemplo 4: Información del Modelo**

```bash
# Verificar que el modelo está cargado correctamente
curl http://localhost:${VLLM_PORT}/v1/models

# Información detallada del modelo
curl http://localhost:${VLLM_PORT}/v1/models/dotsocr
```

#### **Ejemplo 5: Procesamiento de Documento Complejo**

```bash
curl -X POST "http://localhost:${VLLM_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dotsocr",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Extrae toda la información importante de esta imagen: nombres, fechas, números y conceptos clave. Organiza la información de manera estructurada."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://images.unsplash.com/photo-1586953208448-b95a79798f07?w=800"
            }
          }
        ]
      }
    ],
    "max_tokens": 300,
    "temperature": 0.1
  }'
```

### ⚠️ Troubleshooting de Validación

Si los comandos curl no funcionan:

1. **Verificar que el servidor esté corriendo**:
   ```bash
   docker-compose ps
   ```

2. **Verificar logs del contenedor**:
   ```bash
   docker-compose logs vllm-dotsocr
   ```

3. **Probar el health check primero**:
   ```bash
   curl http://localhost:${VLLM_PORT}/health
   # Debería retornar: {"status":"healthy"}
   ```

4. **Si hay errores de conexión**:
   ```bash
   # Verificar que el puerto esté abierto
   netstat -tulpn | grep :${VLLM_PORT}

   # Verificar configuración de firewall
   ufw status
   ```

5. **Para debugging más detallado**:
   ```bash
   # Logs con más detalles
   docker-compose logs -f --tail=100 vllm-dotsocr

   # Información del sistema
   nvidia-smi
   docker system df
   ```

## ⚙️ Configuration

All configuration is done through environment variables in the `.env` file. This allows you to easily adjust parameters according to your GPU capabilities.

### 📋 Configuración Optimizada para RTX 4070 Ti (12GB)

Esta configuración está específicamente optimizada para GPUs RTX 4070 Ti con 12GB de VRAM:

| Parámetro | Valor | Razón |
|-----------|-------|-------|
| `GPU_MEMORY_UTILIZATION` | `0.5` | Usa 50% de VRAM (6GB) para evitar errores OOM |
| `MAX_MODEL_LEN` | `2048` | Longitud de contexto reducida para ahorrar memoria |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Tamaño de batch optimizado para 12GB VRAM |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Previene fragmentación de memoria |

### 🔧 Ajustes por Capacidad de GPU

| GPU | VRAM | GPU_MEMORY_UTIL | MAX_MODEL_LEN | MAX_BATCH_TOKENS |
|-----|------|----------------|---------------|------------------|
| RTX 4090 | 24GB | 0.85 | 8192 | 32768 |
| RTX 4070 Ti | 12GB | 0.5 | 2048 | 8192 |
| RTX 3070 | 8GB | 0.4 | 2048 | 4096 |
| RTX 3060 | 6GB | 0.35 | 1024 | 4096 |
| GTX 1650 | 4GB | 0.3 | 1024 | 2048 |

### .env File

```bash
# Model
VLLM_MODEL=rednote-hilab/dots.ocr
MODEL_NAME=dotsocr

# GPU
CUDA_VISIBLE_DEVICES=0
GPU_MEMORY_UTILIZATION=0.9

# Server
HOST=0.0.0.0
PORT=8002

# Performance
MAX_MODEL_LEN=8192
TENSOR_PARALLEL_SIZE=1
MAX_NUM_BATCHED_TOKENS=32768
```

### Model Parameters

Current values are:
- **Model:** `${VLLM_MODEL}`
- **Port:** `${VLLM_PORT}`
- **Served model name:** `${VLLM_SERVED_MODEL_NAME}`
- **Max length:** `${VLLM_MAX_MODEL_LEN}` tokens
- **GPU memory utilization:** `${VLLM_GPU_MEMORY_UTILIZATION}`
- **Max batch size:** `${VLLM_MAX_NUM_BATCHED_TOKENS}` tokens

### Required Resources

- **GPU:** 1 NVIDIA GPU (any model with sufficient memory)
- **GPU Memory:** Minimum 8GB recommended
- **System RAM:** 16GB+ recommended
- **Storage:** 50GB+ for models and cache

## 🔧 Customization

### Adjust Parameters by GPU

Edit the `.env` file to adjust for your GPU:

```bash
# For 8GB GPUs (RTX 3070, RTX 4060 Ti)
GPU_MEMORY_UTILIZATION=0.8
MAX_MODEL_LEN=4096
MAX_NUM_BATCHED_TOKENS=16384

# For 6GB GPUs (RTX 3060)
GPU_MEMORY_UTILIZATION=0.7
MAX_MODEL_LEN=2048
MAX_NUM_BATCHED_TOKENS=8192

# For 4GB GPUs (GTX 1650)
GPU_MEMORY_UTILIZATION=0.6
MAX_MODEL_LEN=1024
MAX_NUM_BATCHED_TOKENS=4096
```

### Additional Environment Variables

```bash
# Number of GPUs to use
CUDA_VISIBLE_DEVICES=0,1

# Logging level
VLLM_LOGGING_LEVEL=INFO

# Advanced configuration
TENSOR_PARALLEL_SIZE=1
```

## 📡 Usage with API

Once the vLLM server is running, you can use the DotsOCR API:

```bash
# From the project root directory
cd ../vllm-docker-api

# Configure the vLLM server URL in .env
echo "VLLM_SERVER_URL=http://localhost:${VLLM_PORT}" >> .env

# Start the DotsOCR API
docker-compose -f docker-compose.gpu.yml up -d

# API will be available at http://localhost:8000
# vLLM server at http://localhost:${VLLM_PORT}
```

## 🛠️ Troubleshooting

### CUDA Error

```bash
# Check CUDA installation
nvidia-smi

# Check drivers
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

### Memoria Insuficiente (RTX 4090)

Si experimentas errores `CUDA out of memory` con RTX 4090:

1. **Verifica el uso actual de GPU**:
   ```bash
   nvidia-smi
   ```

2. **Aplica la configuración optimizada**:
   ```bash
   cd vllm-models/dotsocr
   cp .env.example .env
   docker-compose down
   docker-compose up -d
   ```

3. **Si persiste, reduce gradualmente**:
   - `GPU_MEMORY_UTILIZATION`: prueba `0.8` (19.2GB)
   - `MAX_MODEL_LEN`: prueba `4096`
   - `MAX_NUM_BATCHED_TOKENS`: prueba `16384`

4. **Limpieza de memoria**:
   ```bash
   docker system prune -f
   ```

### Insufficient Memory

```bash
# Edit .env for your GPU
nano .env

# Reduce memory usage for GPUs with less VRAM
GPU_MEMORY_UTILIZATION=0.7
MAX_MODEL_LEN=4096
MAX_NUM_BATCHED_TOKENS=8192

# Restart the server
docker-compose down
docker-compose up -d
```

### Port Issues

```bash
# Check available ports
netstat -tulpn | grep :${VLLM_PORT}

# Change port in .env
VLLM_PORT=8003

# Restart the server
docker-compose down
docker-compose up -d
```

## 📊 Monitoring

### Resource Usage

```bash
# GPU usage
nvidia-smi

# Container stats
docker stats vllm-dotsocr

# Container logs
docker-compose logs -f vllm-dotsocr

# Detailed model information
curl http://localhost:${VLLM_PORT}/v1/models/${VLLM_SERVED_MODEL_NAME}
```

### Health Checks

```bash
# Health endpoint
curl http://localhost:${VLLM_PORT}/health

# Loaded models
curl http://localhost:${VLLM_PORT}/v1/models

# Specific model information
curl http://localhost:${VLLM_PORT}/v1/models/${VLLM_SERVED_MODEL_NAME}

# Metrics (if available)
curl http://localhost:${VLLM_PORT}/metrics

# View current configuration
echo "Model: ${VLLM_MODEL}"
echo "Port: ${VLLM_PORT}"
echo "Served name: ${VLLM_SERVED_MODEL_NAME}"
```

## 🔄 Updates

To update the model:

```bash
# Stop the container
docker-compose down

# Clear cache if necessary
docker volume rm vllm-models_model_cache

# Restart
docker-compose up -d
```

## 📚 References

- [Official vLLM Documentation](https://docs.vllm.ai/)
- [DotsOCR Model on Hugging Face](https://huggingface.co/rednote-hilab/dots.ocr)
- [GPU Installation Guide](https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html)

## 📋 Archivos Adicionales

### `curl_examples.sh`
Script ejecutable con ejemplos de curl para validar el funcionamiento del modelo:
```bash
cd vllm-models/dotsocr
./curl_examples.sh              # Usa puerto por defecto (8002)
./curl_examples.sh 8002         # Especifica puerto custom
```

**Incluye ejemplos de:**
- ✅ Health checks y verificación de modelos
- 🖼️ Análisis de imágenes con visión por computadora
- 💬 Chat completions con texto e imágenes
- 🔧 Requests de debugging y troubleshooting
- 🌍 Requests multi-idioma (español, inglés)