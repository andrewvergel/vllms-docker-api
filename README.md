# VLLMs Docker API

Una plataforma completa y lista para producción que permite ejecutar múltiples modelos de IA utilizando vLLM (Very Large Language Models) con arquitectura de microservicios. Este proyecto permite conectar múltiples servidores vLLM para procesar diferentes tipos de modelos de IA de manera eficiente y escalable.

> **⚠️ IMPORTANTE: Esta solución requiere GPU NVIDIA obligatoriamente. No es compatible con sistemas que solo tienen CPU.**

## 🌟 Características Principales

- 🚀 **Multi-Modelo Flexible**: Soporte para modelos de lenguaje, visión, OCR, generación de texto, y más
- 📁 **Procesamiento Versátil**: Maneja texto, imágenes, documentos PDF, y otros formatos
- 🔄 **API RESTful**: Endpoints HTTP simples para integración fácil
- 🐳 **Contenedorizado**: Despliegue sencillo con Docker Compose
- ⚡ **Aceleración GPU**: Soporte NVIDIA CUDA para rendimiento óptimo (requerido)
- 📊 **Monitoreo de Salud**: Chequeos de salud y endpoints de estado
- 🔧 **Configurable**: Ajustes flexibles de memoria GPU y parámetros de modelo
- 🎯 **Alto Rendimiento**: Optimizado para GPUs NVIDIA con máxima eficiencia

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                   VLLMs Docker API                          │
│                   (Puerto 8000)                            │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP/REST
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  Servidores vLLM                           │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│  │     OCR         │ │   Lenguaje      │ │     Visión      │ │
│  │   (OlmOCR)      │ │   (Llama)       │ │     (LLaVA)     │ │
│  │   (Puerto 8001) │ │   (Puerto 8002) │ │   (Puerto N)    │ │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🤖 Tipos de Modelos Soportados

Esta plataforma puede ejecutar diversos tipos de modelos de IA:

### 📝 Modelos de Lenguaje (LLM)
- **Generación de texto**: Llama, GPT, Falcon, Mistral
- **Traducción automática**: NLLB, M2M100
- **Resumen de texto**: PEGASUS, BART
- **Análisis de sentimientos**: BERT, RoBERTa

### 👁️ Modelos de Visión-Lenguaje
- **OCR y documentos**: OlmOCR, TrOCR, Donut
- **Descripción de imágenes**: LLaVA, BLIP, OpenFlamingo
- **Análisis visual**: GPT-4V, Claude-3
- **Detección de objetos**: OWL-ViT, GLIP

### 🎯 Modelos Especializados
- **Código**: CodeLlama, StarCoder, DeepSeek Coder
- **Matemáticas**: MATH models, Minerva
- **Ciencia**: Galactica, scientific models
- **Audio**: Wav2CLIP, AudioLDM

## 📋 Prerrequisitos

### Requisitos Obligatorios
- **Docker** y **Docker Compose**
- **Git** (para clonar el repositorio)
- **GPU NVIDIA** con soporte CUDA (requerido)
- **Drivers NVIDIA** instalados
- **Docker con soporte GPU** habilitado
- **Python 3.8+** (para desarrollo local)

### Especificaciones GPU Recomendadas
- **Mínimo:** GPU con 8GB VRAM (RTX 3070, RTX 4060 Ti o superior)
- **Recomendado:** GPU con 12GB+ VRAM para modelos más grandes
- **Óptimo:** GPU con 24GB+ VRAM para máxima capacidad de procesamiento

## 🚀 Inicio Rápido

### 1. Clonar el Repositorio
```bash
git clone <repository-url>
cd vllms-docker-api
```

### 2. Configuración GPU

#### Configurar Modelos Disponibles
```bash
# Configurar modelo OlmOCR
cd vllm-models/olmocr
nano .env  # Ajustar parámetros según tu GPU

# Iniciar modelo OlmOCR
docker-compose up -d

# Volver al directorio raíz
cd ../..

# Configurar la API para conectar con el modelo
nano vllm-api/.env
```

### 3. Verificar Instalación
```bash
# Verificar estado de la API
curl http://localhost:8000/health

# Listar modelos disponibles
curl http://localhost:8000/models

# Información del servicio
curl http://localhost:8000/
```

### 4. Ejemplos de Uso

#### Procesar Documento con OCR
```bash
# Procesar documento PDF con modelo OCR
curl -X POST "http://localhost:8000/process?model=olmocr" \
  -F "file=@documento.pdf" \
  -F "output_format=markdown"
```

#### Generar Texto con LLM
```bash
# Generar texto con modelo de lenguaje
curl -X POST "http://localhost:8000/process?model=llama" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explica cuántica de manera simple", "max_tokens": 100}'
```

#### Describir Imagen con Modelo de Visión
```bash
# Describir imagen con modelo visión-lenguaje
curl -X POST "http://localhost:8000/process?model=llava" \
  -F "file=@imagen.jpg" \
  -F "prompt=Describe esta imagen en detalle"
```

## 📁 Estructura del Proyecto

```
vllms-docker-api/
├── README.md                    # Este archivo
├── vllm-api/                   # API principal multi-modelo
│   ├── app/                    # Aplicación FastAPI
│   │   ├── main.py            # Endpoints y lógica de la API
│   │   ├── requirements.txt   # Dependencias Python
│   │   └── models/            # Modelos y configuración
│   ├── Dockerfile             # Definición del contenedor API
│   ├── docker-compose.yml     # Configuración GPU
│   ├── .env                   # Variables de entorno API
│   └── README.md              # Documentación detallada de la API
├── vllm-models/               # Configuraciones de modelos vLLM
│   ├── README.md              # Guía de modelos
│   ├── model_cache/           # Cache de modelos (volumen Docker)
│   ├── olmocr/               # Modelo OCR (allenai/olmOCR-7B-0825-FP8)
│   │   ├── docker-compose.yml # Configuración vLLM específica
│   │   ├── .env.example       # Variables de entorno del modelo
│   │   └── README.md          # Documentación del modelo
│   └── llama-7b/             # Modelo de lenguaje (meta-llama/Llama-2-7b)
│       ├── docker-compose.yml # Configuración vLLM específica
│       ├── .env.example       # Variables de entorno del modelo
│       └── README.md          # Documentación del modelo
└── workspace/                 # Espacio de trabajo para procesamiento
```

## ⚙️ Configuración

### Variables de Entorno Principales

#### Configuración de la API (`vllm-api/.env`)
```bash
# Configuración general
DEFAULT_GPU_UTIL=0.90
DEFAULT_MAX_LEN=18608

# Servidores vLLM (formato: VLLM_SERVER_NOMBRE_MODELO=URL)
VLLM_SERVER_OLMOCR=http://vllm-olmocr:8001
VLLM_SERVER_LLAMA=http://vllm-llama:8002
VLLM_SERVER_LLAVA=http://vllm-llava:8003

```

#### Configuración del Modelo (`vllm-models/olmocr/.env`)
```bash
# Modelo específico
VLLM_MODEL=allenai/olmOCR-7B-0825-FP8
MODEL_NAME=olmocr

# Configuración GPU
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

## 🔧 Ajustes por GPU

### GPUs con Diferentes Capacidades

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

## 📡 Endpoints de la API

### GET /
Información del servicio y modelos disponibles
```json
{
  "service": "VLLMs Docker API",
  "vllm_servers": {
    "olmocr": "http://vllm-olmocr:8001",
    "llama": "http://vllm-llama:8002"
  },
  "available_models": ["olmocr", "llama"],
  "status": "running"
}
```

### GET /health
Chequeo de salud de todos los servidores vLLM configurados
```json
{
  "status": "healthy",
  "servers_health": {
    "olmocr (http://vllm-olmocr:8001)": true,
    "model2 (http://vllm-model2:8002)": true
  },
  "total_servers": 2,
  "healthy_servers": 2
}
```

### GET /models
Lista de modelos disponibles y su estado
```json
{
  "available_models": {
    "olmocr": {
      "server_url": "http://vllm-olmocr:8001",
      "served_name": "olmocr",
      "healthy": true
    }
  },
  "total_models": 1
}
```

### POST /process
Procesar un archivo PDF o imagen con un modelo específico

**Parámetros:**
- `file` (UploadFile): Archivo PDF o imagen a procesar
- `model` (str): **Requerido** - Nombre del modelo a usar
- `output_format` (str): Formato de salida (`markdown` o `text`)
- `gpu_util` (float): Utilización de memoria GPU (0.0-1.0)
- `max_len` (int): Longitud máxima del modelo

## 🛠️ Desarrollo Local

### 1. Instalar Dependencias
```bash
cd vllm-api/app
pip install -r requirements.txt
```

### 2. Iniciar Servidores vLLM

#### Servidor OCR (OlmOCR)
```bash
# Iniciar servidor vLLM con modelo OCR
docker run -d --name vllm-ocr \
  --gpus all \
  -p 8001:8001 \
  vllm/vllm-openai:nightly \
  --model allenai/olmOCR-7B-0825-FP8 \
  --host 0.0.0.0 --port 8001
```

#### Servidor LLM (Llama)
```bash
# Iniciar servidor vLLM con modelo de lenguaje
docker run -d --name vllm-llama \
  --gpus all \
  -p 8002:8002 \
  vllm/vllm-openai:nightly \
  --model meta-llama/Llama-2-7b-chat-hf \
  --host 0.0.0.0 --port 8002
```

#### Servidor Visión-Lenguaje (LLaVA)
```bash
# Iniciar servidor vLLM con modelo de visión
docker run -d --name vllm-llava \
  --gpus all \
  -p 8003:8003 \
  vllm/vllm-openai:nightly \
  --model llava-hf/llava-1.5-7b-hf \
  --host 0.0.0.0 --port 8003
```

### 3. Iniciar API
```bash
cd vllm-api/app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## ➕ Agregar Nuevo Modelo

### 1. Crear Directorio del Modelo
```bash
cd vllm-models
mkdir nuevo-modelo
```

### 2. Copiar Configuración Base
```bash
# Para modelos basados en plantillas existentes
cp -r olmocr/* nuevo-modelo/
# O crear desde cero
```

### 3. Personalizar Configuración

#### Para Modelo de Lenguaje (LLM)
```bash
# nuevo-modelo/.env
VLLM_MODEL=meta-llama/Llama-2-7b-chat-hf
MODEL_NAME=llama-7b
MAX_MODEL_LEN=4096
GPU_MEMORY_UTILIZATION=0.8
```

#### Para Modelo de Visión (VLM)
```bash
# nuevo-modelo/.env
VLLM_MODEL=llava-hf/llava-1.5-7b-hf
MODEL_NAME=llava
MAX_MODEL_LEN=2048
GPU_MEMORY_UTILIZATION=0.9
```

#### Para Modelo Especializado
```bash
# nuevo-modelo/.env
VLLM_MODEL=codellama/CodeLlama-7b-hf
MODEL_NAME=codellama
MAX_MODEL_LEN=16384
GPU_MEMORY_UTILIZATION=0.85
```

### 4. Probar Nuevo Modelo
```bash
cd vllm-models/nuevo-modelo
docker-compose up -d
docker-compose logs -f
```

## 🔍 Solución de Problemas

### Problemas Comunes con GPU
```bash
# Verificar instalación GPU
nvidia-smi

# Verificar soporte GPU en Docker
docker run --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# Verificar que Docker tenga acceso a GPU
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# Limpiar cache de modelos si hay problemas
docker volume rm vllm-models_model_cache

# Verificar logs específicos de GPU
docker logs nombre-del-contenedor --tail 50
```

### Problemas de Memoria
```bash
# Reducir parámetros en archivos .env
GPU_MEMORY_UTILIZATION=0.7
MAX_MODEL_LEN=4096
MAX_NUM_BATCHED_TOKENS=8192

# Reiniciar servicios
docker-compose down
docker-compose up -d
```

### Problemas de Puerto
```bash
# Verificar puertos disponibles (Linux/macOS)
lsof -i :8000
lsof -i :8001

# Cambiar puertos en archivos .env si es necesario
```

## 📊 Ejemplos de Modelos Disponibles

### Modelos OCR y Documentos
- **OlmOCR:** `allenai/olmOCR-7B-0825-FP8` - OCR avanzado y procesamiento de documentos
- **TrOCR:** `microsoft/trocr-base-printed` - OCR para texto impreso
- **Donut:** `naver-clova-ix/donut-base-finetuned-cord-v2` - Procesamiento de documentos

### Modelos de Lenguaje (LLM)
- **Llama 2:** `meta-llama/Llama-2-7b-chat-hf` - Generación de texto y chat
- **Mistral:** `mistralai/Mistral-7B-Instruct-v0.1` - Modelo instructivo de alta calidad
- **Falcon:** `tiiuae/falcon-7b-instruct` - Modelo eficiente y rápido

### Modelos de Visión-Lenguaje
- **LLaVA:** `llava-hf/llava-1.5-7b-hf` - Descripción y análisis de imágenes
- **BLIP:** `Salesforce/blip-image-captioning-base` - Generación de captions
- **OpenFlamingo:** `openflamingo/OpenFlamingo-3B-vitl-mpt1b` - Modelo visión-lenguaje

### Modelos de Código
- **CodeLlama:** `codellama/CodeLlama-7b-hf` - Generación y comprensión de código
- **StarCoder:** `bigcode/starcoder` - Modelo especializado en programación
- **DeepSeek Coder:** `deepseek-ai/deepseek-coder-6.7b-base` - Modelo código eficiente

## 📊 Monitoreo y Logs

### Ver Logs
```bash
# Logs de la API
docker-compose -f vllm-api/docker-compose.yml logs -f

# Logs del modelo vLLM
docker-compose -f vllm-models/olmocr/docker-compose.yml logs -f

# Logs de todos los servicios
docker-compose -f vllm-api/docker-compose.yml logs
docker-compose -f vllm-models/olmocr/docker-compose.yml logs
```

### Estado de Recursos GPU
```bash
# Uso detallado de GPU
nvidia-smi

# Uso de GPU en tiempo real
nvidia-smi -l 1

# Procesos usando GPU
nvidia-smi pmon -i 0

# Estadísticas de contenedores
docker stats

# Información del sistema
docker system df

# Memoria GPU por proceso
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
```

## 🚀 Despliegue en Producción

### Recomendaciones
1. **GPUs Obligatorias**: Esta solución requiere GPUs NVIDIA para funcionar
2. **Configurar límites de recursos**: En docker-compose.yml según capacidad de GPU
3. **Monitoreo**: Implementar logging y métricas de GPU
4. **Backups**: Configurar volumen persistente para cache de modelos
5. **Actualizaciones**: Mantener modelos y vLLM actualizados
6. **Múltiples GPUs**: Considerar configuración multi-GPU para mayor throughput

### Ejemplo de Producción
```yaml
# docker-compose.production.yml
version: "3.8"
services:
  multi-model-api:
    # ... configuración existente ...
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G

  vllm-olmocr:
    # ... configuración existente ...
    deploy:
      resources:
        limits:
          memory: 16G
        reservations:
          memory: 8G
```

## 🤝 Contribuir

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📚 Recursos Adicionales

- [Documentación oficial de vLLM](https://docs.vllm.ai/)
- [Modelos disponibles en Hugging Face](https://huggingface.co/models)
- [Guía de instalación GPU](https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 📄 Licencia

Este proyecto se proporciona como está para fines educativos y de desarrollo.

## 🆘 Soporte

Para problemas y preguntas:
1. Revisa la sección de solución de problemas
2. Consulta la documentación de Docker y vLLM
3. Abre un issue con información detallada del problema

---

**¡Gracias por usar VLLMs Docker API - La plataforma multi-modelo exclusiva para GPU!** 🚀