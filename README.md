# VLLMs Docker API

A complete production-ready platform for running multiple AI models using vLLM (Very Large Language Models) with a microservices architecture. This project allows you to connect multiple vLLM servers to process different types of AI models efficiently and at scale.

> **⚠️ IMPORTANT: This solution requires NVIDIA GPU. Not compatible with CPU-only systems.**

## 🌟 Key Features

- 🚀 **Flexible Multi-Model**: Support for language models, vision, OCR, text generation, and more
- 📁 **Versatile Processing**: Handles text, images, PDF documents, and other formats
- 🔄 **RESTful API**: Simple HTTP endpoints for easy integration
- 🐳 **Containerized**: Easy deployment with Docker Compose
- ⚡ **GPU Accelerated**: NVIDIA CUDA support for optimal performance (required)
- 📊 **Health Monitoring**: Built-in health checks and status endpoints
- 🔧 **Configurable**: Flexible GPU memory and model parameter settings
- 🎯 **High Performance**: Optimized for NVIDIA GPUs with maximum efficiency

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   VLLMs Docker API                          │
│                   (Port 8000)                              │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP/REST
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  vLLM Servers                              │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│  │      OCR        │ │    Language     │ │     Vision      │ │
│  │   (OlmOCR)      │ │    (Llama)      │ │    (LLaVA)      │ │
│  │   (Port 8001)   │ │   (Port 8002)   │ │   (Port N)      │ │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🤖 Supported Model Types

This platform can run various types of AI models:

### 📝 Language Models (LLM)
- **Text Generation**: Llama, GPT, Falcon, Mistral
- **Machine Translation**: NLLB, M2M100
- **Text Summarization**: PEGASUS, BART
- **Sentiment Analysis**: BERT, RoBERTa

### 👁️ Vision-Language Models
- **OCR & Documents**: OlmOCR, TrOCR, Donut
- **Image Description**: LLaVA, BLIP, OpenFlamingo
- **Visual Analysis**: GPT-4V, Claude-3
- **Object Detection**: OWL-ViT, GLIP

### 🎯 Specialized Models
- **Code**: CodeLlama, StarCoder, DeepSeek Coder
- **Mathematics**: MATH models, Minerva
- **Science**: Galactica, scientific models
- **Audio**: Wav2CLIP, AudioLDM

## 📋 Prerequisites

### Required Software
- **Docker** and **Docker Compose**
- **Git** (for cloning the repository)
- **NVIDIA GPU** with CUDA support (required)
- **NVIDIA drivers** installed
- **Docker with GPU support** enabled
- **Python 3.8+** (for local development)

### Recommended GPU Specifications
- **Minimum:** GPU with 8GB VRAM (RTX 3070, RTX 4060 Ti or better)
- **Recommended:** GPU with 12GB+ VRAM for larger models
- **Optimal:** GPU with 24GB+ VRAM for maximum processing capacity

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd vllms-docker-api
```

### 2. GPU Setup

#### Configure Available Models
```bash
# Configure OlmOCR model
cd vllm-models/olmocr
nano .env  # Adjust parameters for your GPU

# Start OlmOCR model
docker-compose up -d

# Return to root directory
cd ../..

# Configure API to connect with the model
nano vllm-api/.env
```

### 3. Verify Installation
```bash
# Check API health
curl http://localhost:8000/health

# List available models
curl http://localhost:8000/models

# Service information
curl http://localhost:8000/
```

### 4. Usage Examples

#### Process Document with OCR
```bash
# Process PDF document with OCR model
curl -X POST "http://localhost:8000/process?model=olmocr" \
  -F "file=@document.pdf" \
  -F "output_format=markdown"
```

#### Generate Text with LLM
```bash
# Generate text with language model
curl -X POST "http://localhost:8000/process?model=llama" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum physics simply", "max_tokens": 100}'
```

#### Describe Image with Vision Model
```bash
# Describe image with vision-language model
curl -X POST "http://localhost:8000/process?model=llava" \
  -F "file=@image.jpg" \
  -F "prompt=Describe this image in detail"
```

## 📁 Project Structure

```
vllms-docker-api/
├── README.md                    # This file
├── vllm-api/                   # Main multi-model API
│   ├── app/                    # FastAPI application
│   │   ├── main.py            # API endpoints and logic
│   │   ├── requirements.txt   # Python dependencies
│   │   └── models/            # Models and configuration
│   ├── Dockerfile             # API container definition
│   ├── docker-compose.yml     # GPU configuration
│   ├── .env                   # API environment variables
│   └── README.md              # Detailed API documentation
├── vllm-models/               # vLLM model configurations
│   ├── README.md              # Models guide
│   ├── model_cache/           # Model cache (Docker volume)
│   ├── olmocr/               # OCR model (allenai/olmOCR-7B-0825-FP8)
│   │   ├── docker-compose.yml # Specific vLLM configuration
│   │   ├── .env.example       # Model environment variables
│   │   └── README.md          # Model documentation
│   └── llama-7b/             # Language model (meta-llama/Llama-2-7b)
│       ├── docker-compose.yml # Specific vLLM configuration
│       ├── .env.example       # Model environment variables
│       └── README.md          # Model documentation
└── workspace/                 # Processing workspace
```

## ⚙️ Configuration

### Main Environment Variables

#### API Configuration (`vllm-api/.env`)
```bash
# General configuration
DEFAULT_GPU_UTIL=0.90
DEFAULT_MAX_LEN=18608

# vLLM servers (format: VLLM_SERVER_MODEL_NAME=URL)
VLLM_SERVER_OLMOCR=http://vllm-olmocr:8001
VLLM_SERVER_LLAMA=http://vllm-llama:8002
VLLM_SERVER_LLAVA=http://vllm-llava:8003
```

#### Model Configuration (`vllm-models/olmocr/.env`)
```bash
# Specific model
VLLM_MODEL=allenai/olmOCR-7B-0825-FP8
MODEL_NAME=olmocr

# GPU configuration
CUDA_VISIBLE_DEVICES=0
GPU_MEMORY_UTILIZATION=0.9

# Server
HOST=0.0.0.0
PORT=8001

# Performance
MAX_MODEL_LEN=8192
TENSOR_PARALLEL_SIZE=1
MAX_NUM_BATCHED_TOKENS=32768
```

## 🔧 GPU Tuning

### Different GPU Capacities

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

## 📡 API Endpoints

### GET /
Service information and available models
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
Health check for all configured vLLM servers
```json
{
  "status": "healthy",
  "servers_health": {
    "olmocr (http://vllm-olmocr:8001)": true,
    "llama (http://vllm-llama:8002)": true
  },
  "total_servers": 2,
  "healthy_servers": 2
}
```

### GET /models
List all available models and their status
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
Process a PDF or image file with a specific model

**Parameters:**
- `file` (UploadFile): PDF or image file to process
- `model` (str): **Required** - Model name to use
- `output_format` (str): Output format (`markdown` or `text`)
- `gpu_util` (float): GPU memory utilization (0.0-1.0)
- `max_len` (int): Maximum model length

## 🛠️ Local Development

### 1. Install Dependencies
```bash
cd vllm-api/app
pip install -r requirements.txt
```

### 2. Start vLLM Servers

#### OCR Server (OlmOCR)
```bash
# Start vLLM server with OCR model
docker run -d --name vllm-ocr \
  --gpus all \
  -p 8001:8001 \
  vllm/vllm-openai:nightly \
  --model allenai/olmOCR-7B-0825-FP8 \
  --host 0.0.0.0 --port 8001
```

#### LLM Server (Llama)
```bash
# Start vLLM server with language model
docker run -d --name vllm-llama \
  --gpus all \
  -p 8002:8002 \
  vllm/vllm-openai:nightly \
  --model meta-llama/Llama-2-7b-chat-hf \
  --host 0.0.0.0 --port 8002
```

#### Vision-Language Server (LLaVA)
```bash
# Start vLLM server with vision model
docker run -d --name vllm-llava \
  --gpus all \
  -p 8003:8003 \
  vllm/vllm-openai:nightly \
  --model llava-hf/llava-1.5-7b-hf \
  --host 0.0.0.0 --port 8003
```

### 3. Start API
```bash
cd vllm-api/app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## ➕ Add New Model

### 1. Create Model Directory
```bash
cd vllm-models
mkdir new-model
```

### 2. Copy Base Configuration
```bash
# For models based on existing templates
cp -r olmocr/* new-model/
# Or create from scratch
```

### 3. Customize Configuration

#### For Language Model (LLM)
```bash
# new-model/.env
VLLM_MODEL=meta-llama/Llama-2-7b-chat-hf
MODEL_NAME=llama-7b
MAX_MODEL_LEN=4096
GPU_MEMORY_UTILIZATION=0.8
```

#### For Vision Model (VLM)
```bash
# new-model/.env
VLLM_MODEL=llava-hf/llava-1.5-7b-hf
MODEL_NAME=llava
MAX_MODEL_LEN=2048
GPU_MEMORY_UTILIZATION=0.9
```

#### For Specialized Model
```bash
# new-model/.env
VLLM_MODEL=codellama/CodeLlama-7b-hf
MODEL_NAME=codellama
MAX_MODEL_LEN=16384
GPU_MEMORY_UTILIZATION=0.85
```

### 4. Test New Model
```bash
cd vllm-models/new-model
docker-compose up -d
docker-compose logs -f
```

## 🔍 Troubleshooting

### Common GPU Issues
```bash
# Check GPU installation
nvidia-smi

# Verify GPU support in Docker
docker run --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# Verify Docker has GPU access
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# Clear model cache if having issues
docker volume rm vllm-models_model_cache

# Check specific GPU logs
docker logs container-name --tail 50
```

### Memory Issues
```bash
# Reduce parameters in .env files
GPU_MEMORY_UTILIZATION=0.7
MAX_MODEL_LEN=4096
MAX_NUM_BATCHED_TOKENS=8192

# Restart services
docker-compose down
docker-compose up -d
```

### Port Issues
```bash
# Check available ports (Linux/macOS)
lsof -i :8000
lsof -i :8001

# Change ports in .env files if needed
```

## 📊 Available Model Examples

### OCR & Document Models
- **OlmOCR:** `allenai/olmOCR-7B-0825-FP8` - Advanced OCR and document processing
- **TrOCR:** `microsoft/trocr-base-printed` - OCR for printed text
- **Donut:** `naver-clova-ix/donut-base-finetuned-cord-v2` - Document processing

### Language Models (LLM)
- **Llama 2:** `meta-llama/Llama-2-7b-chat-hf` - Text generation and chat
- **Mistral:** `mistralai/Mistral-7B-Instruct-v0.1` - High-quality instruction model
- **Falcon:** `tiiuae/falcon-7b-instruct` - Efficient and fast model

### Vision-Language Models
- **LLaVA:** `llava-hf/llava-1.5-7b-hf` - Image description and analysis
- **BLIP:** `Salesforce/blip-image-captioning-base` - Caption generation
- **OpenFlamingo:** `openflamingo/OpenFlamingo-3B-vitl-mpt1b` - Vision-language model

### Code Models
- **CodeLlama:** `codellama/CodeLlama-7b-hf` - Code generation and understanding
- **StarCoder:** `bigcode/starcoder` - Programming specialized model
- **DeepSeek Coder:** `deepseek-ai/deepseek-coder-6.7b-base` - Efficient code model

## 📊 Monitoring & Logs

### View Logs
```bash
# API logs
docker-compose -f vllm-api/docker-compose.yml logs -f

# vLLM model logs
docker-compose -f vllm-models/olmocr/docker-compose.yml logs -f

# All services logs
docker-compose -f vllm-api/docker-compose.yml logs
docker-compose -f vllm-models/olmocr/docker-compose.yml logs
```

### GPU Resource Status
```bash
# Detailed GPU usage
nvidia-smi

# Real-time GPU usage
nvidia-smi -l 1

# Processes using GPU
nvidia-smi pmon -i 0

# Container statistics
docker stats

# System information
docker system df

# GPU memory per process
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
```

## 🚀 Production Deployment

### Recommendations
1. **GPUs Required**: This solution requires NVIDIA GPUs to function
2. **Set resource limits**: In docker-compose.yml according to GPU capacity
3. **Monitoring**: Implement logging and GPU metrics
4. **Backups**: Set up persistent volume for model cache
5. **Updates**: Keep models and vLLM updated
6. **Multiple GPUs**: Consider multi-GPU setup for higher throughput

### Production Example
```yaml
# docker-compose.production.yml
version: "3.8"
services:
  multi-model-api:
    # ... existing configuration ...
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G

  vllm-olmocr:
    # ... existing configuration ...
    deploy:
      resources:
        limits:
          memory: 16G
        reservations:
          memory: 8G
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 Additional Resources

- [Official vLLM Documentation](https://docs.vllm.ai/)
- [Available Models on Hugging Face](https://huggingface.co/models)
- [GPU Installation Guide](https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 📄 License

This project is provided as-is for educational and development purposes.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review Docker and vLLM documentation
3. Open an issue with detailed problem information

---

**¡Gracias por usar VLLMs Docker API - La plataforma multi-modelo exclusiva para GPU!** 🚀