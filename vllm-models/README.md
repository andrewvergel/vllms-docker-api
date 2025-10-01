# vLLM Models Directory

This directory contains configurations for different models that can be run with vLLM.

## 📁 Structure

```
vllm-models/
├── README.md                    # This file
├── model_cache/                 # Docker volume for model cache
└── [model-name]/               # Folder per model
    ├── docker-compose.yml      # Model-specific configuration
    ├── README.md              # Model documentation
    └── .env                   # Model environment variables
```

## 🚀 Quick Usage

### For a specific model:

```bash
cd vllm-models/[model-name]

# Configure environment variables for your GPU
nano .env  # or edit with your preferred editor

# Start the vLLM server
docker-compose up -d

# View logs in real-time
docker-compose logs -f

# Stop the server
docker-compose down
```

## ⚙️ Environment Variable Configuration

Each model is configured through a `.env` file that allows you to easily adjust all parameters according to your GPU capabilities:

### Main Variables

```bash
# Model
VLLM_MODEL=allenai/olmOCR-7B-0825-FP8
MODEL_NAME=olmocr

# GPU
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

### For Different GPU Capacities

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

## ➕ Add New Model

1. Create directory:
   ```bash
   mkdir vllm-models/new-model
   ```

2. Copy template files:
   ```bash
   cp -r vllm-models/template/* vllm-models/new-model/
   ```

3. Customize configuration:
   - Edit `.env` with the specific model
   - Update `README.md` with documentation
   - Adjust parameters according to available GPU

4. Test the new model:
   ```bash
   cd vllm-models/new-model
   docker-compose up -d
   docker-compose logs -f
   ```

## 🔧 GPU Configuration

All models are configured to use GPU by default. Make sure you have:

- NVIDIA drivers installed
- Docker with GPU support enabled
- Sufficient GPU memory available

## 📊 Available Models

### OlmOCR
- **Model:** `allenai/olmOCR-7B-0825-FP8`
- **Usage:** OCR and document processing
- **Status:** ✅ Configured and ready

## 🔍 Troubleshooting

### Common Issues

1. **GPU Error:**
   ```bash
   # Check GPU installation
   nvidia-smi
   docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
   ```

2. **Insufficient Memory:**
   - Edit `.env` with lower parameters
   - Restart: `docker-compose down && docker-compose up -d`

3. **Model Not Found:**
   - Check internet connection
   - Clear cache: `docker volume rm vllm-models_model_cache`
   - Restart: `docker-compose down && docker-compose up -d`

## 📚 Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [Available Models](https://huggingface.co/models)
- [GPU Installation Guide](https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html)