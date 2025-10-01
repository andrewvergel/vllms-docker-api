# vLLM Models Directory

Este directorio contiene configuraciones para diferentes modelos que pueden ser ejecutados con vLLM.

## 📁 Estructura

```
vllm-models/
├── README.md                    # Este archivo
├── model_cache/                 # Volumen Docker para cache de modelos
└── [model-name]/               # Carpeta por modelo
    ├── docker-compose.yml      # Configuración específica del modelo
    ├── README.md              # Documentación del modelo
    └── .env                   # Variables de entorno del modelo
```

## 🚀 Uso Rápido

### Para un modelo específico:

```bash
cd vllm-models/[model-name]

# Configurar variables de entorno según tu GPU
nano .env  # o editar con tu editor preferido

# Iniciar el servidor vLLM
docker-compose up -d

# Ver logs en tiempo real
docker-compose logs -f

# Detener el servidor
docker-compose down
```

## ⚙️ Configuración por Variables de Entorno

Cada modelo se configura mediante un archivo `.env` que permite ajustar fácilmente todos los parámetros según las capacidades de tu GPU:

### Variables Principales

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

### Para GPUs con Diferentes Capacidades

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

## ➕ Agregar Nuevo Modelo

1. Crear directorio:
   ```bash
   mkdir vllm-models/nuevo-modelo
   ```

2. Copiar archivos plantilla:
   ```bash
   cp -r vllm-models/template/* vllm-models/nuevo-modelo/
   ```

3. Personalizar configuración:
   - Editar `.env` con el modelo específico
   - Actualizar `README.md` con documentación
   - Ajustar parámetros según GPU disponible

4. Probar el nuevo modelo:
   ```bash
   cd vllm-models/nuevo-modelo
   docker-compose up -d
   docker-compose logs -f
   ```

## 🔧 Configuración GPU

Todos los modelos están configurados para usar GPU por defecto. Asegúrate de:

- Tener drivers NVIDIA instalados
- Docker con soporte GPU habilitado
- Suficiente memoria GPU disponible

## 📊 Modelos Disponibles

### OlmOCR
- **Modelo:** `allenai/olmOCR-7B-0825-FP8`
- **Uso:** OCR y procesamiento de documentos
- **Estado:** ✅ Configurado y listo

## 🔍 Solución de Problemas

### Problemas Comunes

1. **Error de GPU:**
   ```bash
   # Verificar instalación GPU
   nvidia-smi
   docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
   ```

2. **Memoria insuficiente:**
   - Editar `.env` con parámetros más bajos
   - Reiniciar: `docker-compose down && docker-compose up -d`

3. **Modelo no encontrado:**
   - Verificar conexión a internet
   - Limpiar cache: `docker volume rm vllm-models_model_cache`
   - Reiniciar: `docker-compose down && docker-compose up -d`

## 📚 Recursos

- [Documentación vLLM](https://docs.vllm.ai/)
- [Modelos disponibles](https://huggingface.co/models)
- [Guía instalación GPU](https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html)