# Guía de Instalación de Docker en Ubuntu

## 🚀 Instalación Rápida (Copiar y Pegar)

### ⚡ Inicio Súper Rápido (con modelo pre-cargado):
```bash
# 1. Pre-descargar modelo (solo primera vez - ~5 minutos)
cd vllm-docker-api
./preload_model.sh

# 2. Iniciar servicios (carga desde disco local - ~30 segundos)
docker compose up --build
```

### 🆕 Inicio Tradicional (descarga desde HuggingFace - ~10-15 minutos):
```bash
# Solo si NO usas el script de pre-carga
cd vllm-docker-api
docker compose up --build
```

### 🚨 **IMPORTANTE - Error CUDA Solucionado:**
Hemos aplicado una configuración **conservadora pero estable** que resuelve el error:
- ❌ **Error anterior**: `CUDA error: operation not permitted`
- ✅ **Solución aplicada**: Parámetros más conservadores y configuración simplificada

## Instalación de Docker y Docker Compose

**⚠️ ADVERTENCIA: Ejecuta todos estos comandos como usuario root o con sudo**

```bash
# Todos los comandos de instalación en uno - COPIAR Y PEGAR COMPLETO
sudo apt update && \
sudo apt install -y apt-transport-https ca-certificates curl gnupg lsb-release && \
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg && \
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null && \
sudo apt update && \
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin && \
sudo systemctl start docker && \
sudo systemctl enable docker && \
sudo usermod -aG docker $USER && \
echo "✅ Docker instalado correctamente. Cierra sesión y vuelve a iniciarla para usar docker sin sudo"
```

## Instalación de Docker y Docker Compose

### Comandos de instalación

```bash
# Actualizar el índice de paquetes
sudo apt update

# Instalar paquetes necesarios para que apt use HTTPS
sudo apt install apt-transport-https ca-certificates curl gnupg lsb-release

# Agregar la clave GPG oficial de Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Agregar el repositorio de Docker a las fuentes de apt
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Actualizar el índice de paquetes nuevamente
sudo apt update

# Instalar Docker Engine, CLI, containerd y docker-compose-plugin
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Iniciar y habilitar el servicio Docker
sudo systemctl start docker
sudo systemctl enable docker

# Verificar que Docker esté funcionando correctamente
sudo docker run hello-world

# Agregar el usuario actual al grupo docker (para usar docker sin sudo)
sudo usermod -aG docker $USER

# Cerrar sesión y volver a iniciarla para que los cambios del grupo tengan efecto
# O ejecutar: newgrp docker
```

### Notas importantes

1. **Docker Compose**: En las versiones más recientes de Docker (20.10+), Docker Compose viene incluido como plugin (`docker-compose-plugin`), por lo que no necesitas instalarlo por separado.

2. **Grupo Docker**: Después de agregar tu usuario al grupo `docker`, necesitarás cerrar sesión y volver a iniciarla para que los cambios tengan efecto. Como alternativa, puedes ejecutar `newgrp docker` en la terminal actual.

3. **Verificación**: El comando `sudo docker run hello-world` descargará una imagen de prueba y verificará que Docker esté funcionando correctamente.

4. **Versiones**: Estos comandos instalarán las versiones más recientes disponibles en el repositorio oficial de Docker.

### Comandos útiles después de la instalación

```bash
# Verificar versión de Docker
docker --version

# Verificar versión de Docker Compose
docker compose version

# Listar imágenes
docker images

# Listar contenedores en ejecución
docker ps

# Listar todos los contenedores
docker ps -a

# Detener el servicio Docker
sudo systemctl stop docker

# Reiniciar el servicio Docker
sudo systemctl restart docker
```

### Solución de problemas comunes

**Si Docker no inicia automáticamente:**
```bash
sudo systemctl enable docker
sudo systemctl start docker
```

**Si tienes problemas con permisos:**
```bash
sudo usermod -aG docker $USER
newgrp docker
```

**Para verificar que el servicio esté activo:**
```bash
sudo systemctl status docker
```

---

## 🚀 Configuración de Alto Rendimiento para tu Máquina

### Especificaciones de tu hardware:
- **GPU**: 48GB VRAM, 71.5 TFLOPS
- **CPU**: Xeon® E5-2686 v4 (36 cores/72 threads)
- **RAM**: 128.9 GB
- **Storage**: SSD rápido (1948 MB/s)
- **Red**: 367.7 Mbps

### Parámetros optimizados aplicados:

#### vLLM Server (docker-compose.yml):
```yaml
--max-model-len 16384              # Contexto estándar (estable)
--gpu-memory-utilization 0.90      # Utilización conservadora de GPU
--max-num-batched-tokens 32768     # Throughput optimizado
--max-num-seqs 128                 # Requests concurrentes
--tensor-parallel-size 1           # Single GPU (estable)
--block-size 16                    # Bloques estándar para 48GB VRAM
--swap-space 8                     # Espacio de swap estándar
--max-parallel-loading-workers 2   # Carga estándar de modelos
--disable-log-stats                # Reduce overhead de estadísticas
--disable-custom-all-reduce        # Deshabilita all-reduce personalizado (P2P no disponible)
```

#### API (main.py):
```python
DEFAULT_GPU_UTIL = "0.90"      # Utilización conservadora
DEFAULT_MAX_LEN = "16384"      # Contexto estándar
```
```

### Comandos para monitoreo de rendimiento:

```bash
# Monitoreo de GPU en tiempo real
watch -n 1 nvidia-smi

# Uso de CPU y memoria
htop

# Logs del contenedor vLLM
docker compose logs -f vllm-server

# Métricas de rendimiento de la API
curl http://localhost:8000/health

# Información detallada de GPU
nvidia-smi -q

# Uso de memoria del sistema
free -h

# I/O del disco
iostat -x 1
```

### Ajustes adicionales recomendados:

#### Para workloads muy intensivos:
```bash
# Aumentar límites del sistema si es necesario
sudo sysctl -w vm.max_map_count=262144
sudo sysctl -w fs.file-max=1000000
```

#### Para debugging de rendimiento:
```bash
# Profiling detallado (si necesitas diagnosticar)
docker compose logs -f vllm-server | grep -E "(throughput|latency|memory)"

# Métricas de latencia
curl -X POST "http://localhost:8000/process" \
  -H "Content-Type: application/json" \
  -d '{"file": "test.pdf", "format": "benchmark"}'
```

### Resultados esperados:
- **Configuración estable**: Sin errores de CUDA graphs
- **Funcionamiento garantizado**: Compatible con vLLM nightly
- **Uso eficiente de recursos**: 90% de utilización de GPU (estable)
- **Comunicación GPU optimizada** (single GPU para evitar problemas P2P)
- **Sin problemas de permisos CUDA**: Configuración conservadora pero funcional

### Sistema de Cache de Modelos

Para evitar la descarga lenta del modelo desde HuggingFace cada vez que inicias vLLM, hemos implementado un sistema de cache local:

#### **1. Pre-descargar el modelo (primera vez):**
```bash
cd vllm-docker-api
./preload_model.sh
```

#### **2. Volúmenes de cache configurados:**
```yaml
volumes:
  - ./model_cache:/root/.cache/huggingface    # Cache de HuggingFace
  - /mnt/hf_models:/mnt/hf_models             # Modelos locales
```

#### **3. Modelo cargado desde disco local:**
```bash
--model /mnt/hf_models/olmOCR-7B-0825-FP8   # Carga desde disco local
--download-dir /mnt/hf_models               # Descargas a disco local
```

### Troubleshooting aplicado basado en logs:

**Problemas críticos identificados y solucionados:**
- ✅ **Error de CUDA "operation not permitted"**: Solucionado con configuración conservadora
- ✅ **Tensor Parallel Size**: Cambiado de 2 a 1 (P2P no disponible)
- ✅ **Custom All-Reduce**: Deshabilitado explícitamente
- ✅ **Parámetros de memoria**: Reducidos para estabilidad (90% vs 96%)
- ✅ **Block size**: Reducido de 32 a 16 para mayor compatibilidad
- ✅ **Log Stats**: Deshabilitado para reducir overhead
- ✅ **Variables de entorno conflictivas**: Removidas para evitar problemas CUDA

### Beneficios del Sistema de Cache:

#### **⏱️ Velocidad de Carga:**
- **Sin cache**: ~10-15 minutos (descarga desde HuggingFace)
- **Con cache**: ~30 segundos (carga desde disco local SSD)

#### **💾 Ahorro de Ancho de Banda:**
- **Sin cache**: ~8GB descargados por inicio
- **Con cache**: Cero descarga después del primer uso

#### **🔄 Reutilización:**
- El modelo se comparte entre múltiples sesiones
- Persistente entre reinicios de contenedores
- Más rápido en subsiguientes ejecuciones

#### **📊 Verificación del Cache:**
```bash
# Verificar que el modelo esté en disco local
ls -la /mnt/hf_models/olmOCR-7B-0825-FP8/

# Ver logs de carga rápida
docker compose logs vllm-server | grep "Loading model"

# Ver uso del disco
df -h /mnt/hf_models
```