# OlmOCR vLLM Model

Este directorio contiene la configuración para ejecutar el modelo OlmOCR usando vLLM en una máquina con GPU.

## 🚀 Inicio Rápido

### Prerrequisitos

- Docker y Docker Compose instalados
- GPU NVIDIA con drivers instalados
- Suficiente memoria GPU (recomendado 8GB+)

### Ejecutar el Modelo

```bash
# Desde el directorio raíz del proyecto
cd vllm-models/olmocr

# Configurar según tu GPU (opcional)
nano .env

# Iniciar el servidor vLLM
docker-compose up -d

# Ver logs en tiempo real
docker-compose logs -f ${CONTAINER_NAME}

# Detener el servidor
docker-compose down
```

### Verificación

Una vez iniciado, puedes verificar que el servidor esté funcionando:

```bash
# Health check
curl http://localhost:${VLLM_PORT}/health

# Lista de modelos disponibles
curl http://localhost:${VLLM_PORT}/v1/models

# Ver información del modelo
curl http://localhost:${VLLM_PORT}/v1/models/${VLLM_SERVED_MODEL_NAME}
```

## ⚙️ Configuración

Toda la configuración se realiza mediante variables de entorno en el archivo `.env`. Esto permite ajustar fácilmente los parámetros según las capacidades de tu GPU.

### Archivo .env

```bash
# Modelo
VLLM_MODEL=allenai/olmOCR-7B-0825-FP8
MODEL_NAME=olmocr

# GPU
CUDA_VISIBLE_DEVICES=0
GPU_MEMORY_UTILIZATION=0.9

# Servidor
HOST=0.0.0.0
PORT=8001

# Rendimiento
MAX_MODEL_LEN=8192
TENSOR_PARALLEL_SIZE=1
MAX_NUM_BATCHED_TOKENS=32768
```

### Parámetros del Modelo

Los valores actuales son:
- **Modelo:** `${VLLM_MODEL}`
- **Puerto:** `${VLLM_PORT}`
- **Nombre del modelo servido:** `${VLLM_SERVED_MODEL_NAME}`
- **Longitud máxima:** `${VLLM_MAX_MODEL_LEN}` tokens
- **Utilización de memoria GPU:** `${VLLM_GPU_MEMORY_UTILIZATION}`
- **Batch size máximo:** `${VLLM_MAX_NUM_BATCHED_TOKENS}` tokens

### Recursos Requeridos

- **GPU:** 1 GPU NVIDIA (cualquier modelo con suficiente memoria)
- **Memoria GPU:** Mínimo 8GB recomendado
- **RAM del sistema:** 16GB+ recomendado
- **Almacenamiento:** 50GB+ para modelos y cache

## 🔧 Personalización

### Ajustar Parámetros por GPU

Edita el archivo `.env` para ajustar según tu GPU:

```bash
# Para GPUs con 8GB (RTX 3070, RTX 4060 Ti)
GPU_MEMORY_UTILIZATION=0.8
MAX_MODEL_LEN=4096
MAX_NUM_BATCHED_TOKENS=16384

# Para GPUs con 6GB (RTX 3060)
GPU_MEMORY_UTILIZATION=0.7
MAX_MODEL_LEN=2048
MAX_NUM_BATCHED_TOKENS=8192

# Para GPUs con 4GB (GTX 1650)
GPU_MEMORY_UTILIZATION=0.6
MAX_MODEL_LEN=1024
MAX_NUM_BATCHED_TOKENS=4096
```

### Variables de Entorno Adicionales

```bash
# Número de GPUs a usar
CUDA_VISIBLE_DEVICES=0,1

# Nivel de logging
VLLM_LOGGING_LEVEL=INFO

# Configuración avanzada
TENSOR_PARALLEL_SIZE=1
```

## 📡 Uso con la API

Una vez que el servidor vLLM esté corriendo, puedes usar la API de OlmOCR:

```bash
# Desde el directorio raíz del proyecto
cd ../vllm-docker-api

# Configurar la URL del servidor vLLM en .env
echo "VLLM_SERVER_URL=http://localhost:${VLLM_PORT}" >> .env

# Iniciar la API de OlmOCR
docker-compose -f docker-compose.cpu.yml up -d

# La API estará disponible en http://localhost:8000
# El servidor vLLM en http://localhost:${VLLM_PORT}
```

## 🛠️ Solución de Problemas

### Error de CUDA

```bash
# Verificar instalación de CUDA
nvidia-smi

# Verificar drivers
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

### Memoria Insuficiente

```bash
# Editar .env según tu GPU
nano .env

# Reducir uso de memoria para GPUs con menos VRAM
GPU_MEMORY_UTILIZATION=0.7
MAX_MODEL_LEN=4096
MAX_NUM_BATCHED_TOKENS=8192

# Reiniciar el servidor
docker-compose down
docker-compose up -d
```

### Problemas de Puerto

```bash
# Verificar puertos disponibles
netstat -tulpn | grep :${VLLM_PORT}

# Cambiar puerto en .env
VLLM_PORT=8002

# Reiniciar el servidor
docker-compose down
docker-compose up -d
```

## 📊 Monitoreo

### Uso de Recursos

```bash
# GPU usage
nvidia-smi

# Container stats
docker stats vllm-olmocr

# Logs del container
docker-compose logs -f vllm-olmocr

# Información detallada del modelo
curl http://localhost:${VLLM_PORT}/v1/models/${VLLM_SERVED_MODEL_NAME}
```

### Health Checks

```bash
# Health endpoint
curl http://localhost:${VLLM_PORT}/health

# Modelos cargados
curl http://localhost:${VLLM_PORT}/v1/models

# Información específica del modelo
curl http://localhost:${VLLM_PORT}/v1/models/${VLLM_SERVED_MODEL_NAME}

# Métricas (si disponibles)
curl http://localhost:${VLLM_PORT}/metrics

# Ver configuración actual
echo "Modelo: ${VLLM_MODEL}"
echo "Puerto: ${VLLM_PORT}"
echo "Nombre servido: ${VLLM_SERVED_MODEL_NAME}"
```

## 🔄 Actualizaciones

Para actualizar el modelo:

```bash
# Detener el container
docker-compose down

# Limpiar cache si es necesario
docker volume rm vllm-models_model_cache

# Reiniciar
docker-compose up -d
```

## 📚 Referencias

- [Documentación oficial de vLLM](https://docs.vllm.ai/)
- [Modelo OlmOCR en Hugging Face](https://huggingface.co/allenai/olmOCR-7B-0825)
- [Guía de instalación GPU](https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html)