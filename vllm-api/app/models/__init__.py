"""
Models package for Multi-Model OCR API.

This package contains model-specific implementations following Clean Code principles.
Each model has its own context and processing logic while maintaining consistent
input/output interfaces.
"""

from .base_model import BaseModel, ModelConfig, ProcessingResult
from .model_factory import ModelFactory
from .olmocr_model import OlmOCRModel
from .pdf_converter import (
    PDFConverter,
    DefaultPDFConverter,
    PDFFileManager,
    PDFConverterError,
    PDFDependencyError,
    PDFConversionError,
    convert_pdf_to_images,
    check_pdf_support,
    get_pdf_conversion_help
)

__all__ = [
    'BaseModel',
    'ModelConfig',
    'ProcessingResult',
    'ModelFactory',
    'OlmOCRModel',
    # PDF Converter Library
    'PDFConverter',
    'DefaultPDFConverter',
    'PDFFileManager',
    'PDFConverterError',
    'PDFDependencyError',
    'PDFConversionError',
    'convert_pdf_to_images',
    'check_pdf_support',
    'get_pdf_conversion_help'
]

# Initialize model factory on import
try:
    ModelFactory.load_model_configs()
except Exception:
    # Ignore errors during import - will be handled at runtime
    pass