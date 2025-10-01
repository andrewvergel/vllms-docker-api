# Multi-Model OCR API

A production-ready OCR (Optical Character Recognition) service that can connect to multiple vLLM servers for processing different models. Built with a microservices architecture using vLLM for high-performance inference and FastAPI for the REST API layer.

## Overview

This project provides a containerized OCR solution that combines:
- **Multi-Model Support**: Connect to multiple vLLM servers via environment variables
- **Dynamic Routing**: Automatically route requests to appropriate vLLM server based on model
- **FastAPI Interface**: REST API for document processing with file upload support
- **Flexible Configuration**: Easy model management through environment variables

## Architecture

```
┌─────────────────┐    ┌──────────────────┐
│   FastAPI       │◄──►│   VLLM Server    │
│   Container     │    │   Container      │
│   (Port 8000)   │    │   (Port 8001)    │
└─────────────────┘    └──────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌──────────────────┐
│   File Upload   │    │   OlmOCR Model   │
│   Processing    │    │   Inference      │
└─────────────────┘    └──────────────────┘
```

## Features

- 🚀 **High-Performance OCR**: State-of-the-art vision-language model for accurate text extraction
- 📁 **Multi-Format Support**: Process PDF documents and image files
- 🔄 **RESTful API**: Simple HTTP endpoints for easy integration
- 🐳 **Containerized**: Easy deployment with Docker Compose
- ⚡ **GPU Accelerated**: NVIDIA CUDA support for optimal performance
- 📊 **Health Monitoring**: Built-in health checks and status endpoints
- 🔧 **Configurable**: Adjustable GPU memory utilization and model parameters

## Prerequisites

- **Docker** and **Docker Compose**
- **NVIDIA GPU** with CUDA support (recommended for production)
- **Python 3.8+** (for local development)
- **Git** (for cloning the repository)

## Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd vllm-docker-api
```

### 2. Start the Services

#### For Systems with NVIDIA GPU:
```bash
docker compose up -d
```

#### For Systems without GPU (CPU-only):
```bash
# Using the CPU-optimized configuration
./start-cpu.sh
```

Or manually:
```bash
docker compose -f docker-compose.cpu.yml up -d
```

This will start both containers:
- `olmocr-api` on `http://localhost:8000`
- `vllm-server` on `http://localhost:8001`

### 3. Check Status and Available Models
```bash
# Check API health
curl http://localhost:8000/health

# List available models
curl http://localhost:8000/models
```

### 4. Process a Document
```bash
# List available models first
curl http://localhost:8000/models

# Process with a specific model
curl -X POST "http://localhost:8000/process?model=olmocr" \
  -F "file=@document.pdf" \
  -F "output_format=markdown"
```

## API Endpoints

### GET /
Service information and available models
```json
{
  "service": "Multi-Model OCR API",
  "vllm_servers": {
    "olmocr": "http://vllm-olmocr:8001",
    "model2": "http://vllm-model2:8002"
  },
  "available_models": ["olmocr", "model2"],
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
    "model2 (http://vllm-model2:8002)": true
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
- `model` (str): **Required** - Model name to use (must match VLLM_SERVER_* config)
- `output_format` (str): Output format (`markdown` or `text`)
- `gpu_util` (float): GPU memory utilization (0.0-1.0)
- `max_len` (int): Maximum model length

**Response:**
```json
{
  "filename": "document.pdf",
  "model_used": "olmocr",
  "server_used": "http://vllm-olmocr:8001",
  "served_model_name": "olmocr",
  "markdown_base64": "base64-encoded-markdown-content",
  "processing_time_seconds": 45,
  "total_input_tokens": 5256,
  "total_output_tokens": 2569,
  "pages_processed": 3
}
```

## Configuration

### Environment Variables

#### Core API Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_GPU_UTIL` | `0.90` | Default GPU memory utilization |
| `DEFAULT_MAX_LEN` | `18608` | Default maximum model length |

#### Multi-Model vLLM Configuration
| Variable Pattern | Example | Description |
|------------------|---------|-------------|
| `VLLM_SERVER_[MODEL]` | `VLLM_SERVER_OLMOCR=http://vllm-olmocr:8001` | vLLM server for specific model |
| `VLLM_MODEL_[MODEL]` | `VLLM_MODEL_OLMOCR=olmocr` | Served model name (optional) |

#### Device Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `OLMOCR_DEVICE` | `cpu` | Device type for processing |

### Example Configuration

```bash
# .env file
DEFAULT_GPU_UTIL=0.90
DEFAULT_MAX_LEN=18608

# Model servers
VLLM_SERVER_OLMOCR=http://vllm-olmocr:8001
VLLM_SERVER_MODEL2=http://vllm-model2:8002

# Optional: Custom served names
VLLM_MODEL_OLMOCR=olmocr
VLLM_MODEL_MODEL2=model2
```

## Development

### Local Setup

1. **Install Dependencies**
```bash
cd app
pip install -r requirements.txt
```

2. **Start VLLM Server**
```bash
docker run -d --name vllm-server \
  --gpus all \
  -p 8001:8001 \
  vllm/vllm-openai:nightly \
  --model allenai/olmOCR-7B-0825-FP8 \
  --host 0.0.0.0 --port 8001
```

3. **Start API Server**
```bash
cd app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Project Structure

```
vllm-docker-api/
├── app/                    # FastAPI application
│   ├── main.py            # API endpoints and logic
│   └── requirements.txt   # Python dependencies
├── workspace/             # File processing workspace (optional)
│   └── warmup/           # Model warmup files (not currently used)
├── Dockerfile            # API container definition
├── docker-compose.yml    # GPU-accelerated setup
├── docker-compose.cpu.yml # CPU-only setup (Mac/Windows/Linux)
├── start-cpu.sh          # Startup script for CPU-only mode
├── entrypoint.sh         # Container startup script
├── .gitignore           # Git ignore patterns
└── README.md            # This file
```

## Performance Tuning

### GPU Memory Optimization
- Adjust `--gpu-memory-utilization` (0.0-1.0) based on available VRAM
- Lower values increase batch size but may reduce throughput

### Model Parameters
- `--max-model-len`: Maximum sequence length (default: 8192)
- Higher values support longer documents but use more memory

### Batch Processing
For multiple files, consider implementing client-side batching to optimize GPU utilization.

## CPU-Only Setup

### System Requirements

The `docker-compose.cpu.yml` configuration works on any system without GPU requirements:

- **CPU-Only Mode**: Runs without GPU acceleration for maximum compatibility
- **Memory Limits**: Conservative memory allocation (8GB for VLLM, 2GB for API)
- **Reduced Model Length**: Lower `max-model-len` (4096) for stability
- **No GPU Dependencies**: Works on any system with Docker support

### Supported Platforms

This configuration works on:
- **Mac (Intel/Apple Silicon)** without GPU support
- **Linux systems** without NVIDIA GPUs
- **Windows** with Docker Desktop (CPU mode)
- **Cloud instances** without GPU access
- **Development environments** with limited resources

### Performance Expectations

- **Processing Speed**: Slower than GPU-accelerated version (CPU-only)
- **Memory Usage**: More predictable and stable
- **Model Loading**: First startup may take 5-10 minutes for model download
- **Resource Usage**: Lower memory footprint, suitable for development

### Troubleshooting for CPU-Only Setup

1. **Docker Issues**
   ```bash
   # Check Docker status
   docker info

   # Restart Docker if needed
   ```

2. **Memory Issues**
   ```bash
   # Check available memory
   docker system df

   # Clean up unused containers/images
   docker system prune -a
   ```

3. **Port Conflicts**
   ```bash
   # Check port usage (Linux/macOS)
   lsof -i :8000
   lsof -i :8001

   # Or on Windows:
   # netstat -ano | findstr :8000

   # Kill process using port
   kill -9 $(lsof -ti:8000)
   ```

4. **Model Download Issues**
   ```bash
   # Clear model cache
   docker volume rm vllm-docker-api_model_cache

   # Check download progress
   docker compose -f docker-compose.cpu.yml logs vllm-server
   ```

### Logs (CPU Setup)
```bash
# View all logs
docker compose -f docker-compose.cpu.yml logs

# View specific service logs
docker compose -f docker-compose.cpu.yml logs olmocr-api-cpu
docker compose -f docker-compose.cpu.yml logs vllm-server-cpu

# Follow logs in real-time
docker compose -f docker-compose.cpu.yml logs -f
```

## General Troubleshooting

### Common Issues

1. **GPU Not Detected (Linux)**
   ```bash
   # Check GPU availability
   nvidia-smi
   # Ensure Docker has GPU access
   docker run --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
   ```

2. **Port Conflicts**
   ```bash
   # Check port usage (Linux)
   netstat -tlnp | grep :8000
   # Modify ports in docker-compose.yml if needed
   ```

3. **Model Download Issues**
   ```bash
   # Clear model cache
   docker volume rm vllm-docker-api_model_cache
   ```

### Logs (Linux)
```bash
# View all logs
docker compose logs

# View specific service logs
docker compose logs olmocr-api
docker compose logs vllm-server
```

## License

This project is provided as-is for educational and development purposes.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review Docker and VLLM documentation
3. Open an issue with detailed information