# Guía de Instalación de Docker en Ubuntu

## 🚀 Instalación Rápida (Copiar y Pegar)

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
sudo docker run hello-world && \
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
--max-model-len 20480              # Aumentado de 16384 (+25% contexto)
--gpu-memory-utilization 0.96      # Alta utilización de GPU (segura)
--max-num-batched-tokens 49152     # Mayor throughput por batch
--max-num-seqs 192                 # Más requests concurrentes
--tensor-parallel-size 1           # Single GPU para máxima eficiencia
--block-size 32                    # Bloques más grandes para 48GB VRAM
--swap-space 12                    # Espacio de swap optimizado
--max-parallel-loading-workers 4   # Carga más rápida de modelos
--enable-chunked-prefill           # Optimiza procesamiento de prompts largos
--disable-log-stats                # Reduce overhead de estadísticas
--disable-custom-all-reduce        # Deshabilita all-reduce personalizado (P2P no disponible)
```

#### API (main.py):
```python
DEFAULT_GPU_UTIL = "0.96"      # Alta utilización (segura)
DEFAULT_MAX_LEN = "20480"      # Más contexto disponible

### Variables de entorno adicionales:
```yaml
TORCH_CUDA_ARCH_LIST=8.6          # Arquitectura CUDA específica (evita warnings)
OMP_NUM_THREADS=1                 # Controla paralelismo PyTorch
NCCL_DEBUG=INFO                   # Debug de comunicación GPU
CUDA_VISIBLE_DEVICES=0            # Usa solo primera GPU
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
- **Mayor throughput**: +25-50% más tokens por segundo
- **Mayor concurrencia**: Hasta 192 requests simultáneos
- **Mejor latencia**: Procesamiento más rápido de documentos largos
- **Uso eficiente de recursos**: 96% de utilización de GPU (estable)
- **Comunicación GPU optimizada** (single GPU para evitar problemas P2P)

### Troubleshooting aplicado basado en logs:

**Problemas identificados y solucionados:**
- ✅ **Tensor Parallel Size**: Cambiado de 2 a 1 (P2P no disponible)
- ✅ **Custom All-Reduce**: Deshabilitado explícitamente
- ✅ **OMP Threads**: Controlado con variable de entorno
- ✅ **CUDA Architecture**: Especificada para evitar warnings
- ✅ **Log Stats**: Deshabilitado para reducir overhead