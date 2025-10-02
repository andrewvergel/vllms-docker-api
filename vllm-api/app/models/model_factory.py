"""
Model factory for creating and managing model instances.

This module implements the Factory pattern to centralize model creation
and ensure proper configuration loading from environment variables.
"""

import os
from typing import Dict, Optional
from .base_model import BaseModel, ModelConfig


class ModelFactory:
    """
    Factory class for creating and managing model instances.

    This class loads model configurations from environment variables and
    provides a centralized way to access model instances.
    """

    _models: Dict[str, BaseModel] = {}
    _configs: Dict[str, ModelConfig] = {}

    @classmethod
    def load_model_configs(cls) -> Dict[str, ModelConfig]:
        """
        Load model configurations from environment variables.

        Looks for patterns like:
        - VLLM_SERVER_[MODEL_NAME]
        - VLLM_MODEL_[MODEL_NAME] (optional)

        Returns:
            Dictionary of model configurations
        """
        configs = {}

        for key, value in os.environ.items():
            if key.startswith("VLLM_SERVER_"):
                model_name = key.replace("VLLM_SERVER_", "").lower()

                # Get served name (defaults to model name)
                served_name_key = f"VLLM_MODEL_{model_name.upper()}"
                served_name = os.getenv(served_name_key, model_name)

                configs[model_name] = ModelConfig(
                    name=model_name,
                    server_url=value,
                    served_name=served_name
                )

        return configs

    @classmethod
    def get_model(cls, model_name: str, model_class: type = None) -> BaseModel:
        """
        Get or create a model instance.

        Args:
            model_name: Name of the model to retrieve
            model_class: Specific model class to use (optional)

        Returns:
            Configured model instance

        Raises:
            ValueError: If model is not configured
        """
        if model_name not in cls._configs:
            cls._configs = cls.load_model_configs()

        if model_name not in cls._configs:
            available = list(cls._configs.keys())
            raise ValueError(f"Model '{model_name}' not configured. Available models: {available}")

        config = cls._configs[model_name]

        if model_name not in cls._models:
            if model_class is None:
                # Select model class based on model name
                if 'dotsocr' in model_name.lower():
                    from .dotsocr_model import DotsOCRModel
                    model_class = DotsOCRModel
                else:
                    # Default to OlmOCR for backward compatibility
                    from .olmocr_model import OlmOCRModel
                    model_class = OlmOCRModel

            cls._models[model_name] = model_class(config)

        return cls._models[model_name]

    @classmethod
    def get_available_models(cls) -> Dict[str, ModelConfig]:
        """
        Get all available model configurations.

        Returns:
            Dictionary of model configurations
        """
        if not cls._configs:
            cls._configs = cls.load_model_configs()
        return cls._configs.copy()

    @classmethod
    def is_model_available(cls, model_name: str) -> bool:
        """
        Check if a model is configured and available.

        Args:
            model_name: Name of the model to check

        Returns:
            True if model is configured
        """
        if not cls._configs:
            cls._configs = cls.load_model_configs()
        return model_name in cls._configs

    @classmethod
    def clear_cache(cls):
        """Clear the model cache (useful for testing)."""
        cls._models.clear()
        cls._configs.clear()