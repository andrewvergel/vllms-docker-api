from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import requests
from typing import Dict

from .models import ModelFactory, ProcessingResult

app = FastAPI(title="Multi-Model API")

# Configuración desde variables de entorno
DEFAULT_GPU_UTIL = float(os.getenv("DEFAULT_GPU_UTIL", "0.90"))
DEFAULT_MAX_LEN = int(os.getenv("DEFAULT_MAX_LEN", "18608"))


def check_vllm_health(server_url: str) -> bool:
    """
    Verifica si un servidor vLLM específico está disponible.
    """
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def check_all_servers_health() -> Dict[str, bool]:
    """
    Verifica el estado de todos los servidores vLLM configurados.
    """
    health_status = {}
    configs = ModelFactory.get_available_models()

    for model_name, config in configs.items():
        health_status[f"{model_name} ({config.server_url})"] = check_vllm_health(config.server_url)

    return health_status


@app.get("/")
def root():
    """
    Root endpoint with API information.
    """
    all_healthy = all(check_all_servers_health().values())
    configs = ModelFactory.get_available_models()

    return {
        "service": "Multi-Model Processing API",
        "version": "1.0.0",
        "available_models": list(configs.keys()),
        "status": "running" if all_healthy else "some_servers_unavailable",
        "endpoints": {
            "health": "/health",
            "models": "/v1/models",
            "process": "/v1/process",
            "documentation": "/docs"
        }
    }


@app.get("/health")
def health_check():
    """
    Health check para verificar que los servidores vLLM estén disponibles.
    """
    health_status = check_all_servers_health()
    all_healthy = all(health_status.values())

    return {
        "status": "healthy" if all_healthy else "unhealthy",
        "servers_health": health_status,
        "total_servers": len(health_status),
        "healthy_servers": sum(health_status.values())
    }


# API v1 Routes - Multi-Model API endpoints
@app.get("/v1/models")
def list_models():
    """
    List all available models and their configurations.
    """
    configs = ModelFactory.get_available_models()
    models_info = {}

    for model_name, config in configs.items():
        is_healthy = check_vllm_health(config.server_url)

        models_info[model_name] = {
            "server_url": config.server_url,
            "served_name": config.served_name,
            "healthy": is_healthy
        }

    return {
        "available_models": models_info,
        "total_models": len(models_info)
    }




@app.post("/v1/process")
async def process_document(
    file: UploadFile = File(...),
    output_format: str = "markdown",
    model: str,  # Required model - must be configured
    gpu_util: float = DEFAULT_GPU_UTIL,
    max_len: int = DEFAULT_MAX_LEN,
):
    """
    Process a document using the specified model.

    The 'model' parameter is required and must correspond to a model
    configured in the environment variables (VLLM_SERVER_[model]).
    """
    try:
        # Get model instance
        model_instance = ModelFactory.get_model(model)

        # Process the file using the model
        result: ProcessingResult = model_instance.process_file(
            file=file,
            output_format=output_format,
            gpu_util=gpu_util,
            max_len=max_len
        )

        # Convert result to response format
        return {
            "filename": result.filename,
            "model_used": result.model_used,
            "server_used": result.server_used,
            "served_model_name": result.served_model_name,
            "content_base64": result.content_base64,
            "processing_time_seconds": result.processing_time_seconds,
            "total_input_tokens": result.total_input_tokens,
            "total_output_tokens": result.total_output_tokens,
            "pages_processed": result.pages_processed,
            "metadata": result.metadata or {}
        }

    except ValueError as e:
        # Model not configured
        available_models = list(ModelFactory.get_available_models().keys())
        raise HTTPException(
            status_code=400,
            detail={
                "error": str(e),
                "available_models": available_models
            }
        )

    except RuntimeError as e:
        # Processing failed
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "model": model
            }
        )

    except Exception as e:
        # Unexpected error
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Unexpected error: {str(e)}",
                "model": model
            }
        )


