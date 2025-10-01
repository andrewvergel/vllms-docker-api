"""
PDF Converter Utility Library

A clean, reusable library for converting PDF files to images.
This module provides a simple interface for PDF processing operations
with proper error handling and system dependency management.

Author: VLLM API Team
Version: 1.0.0
"""

import os
import tempfile
import uuid
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# System dependencies check
try:
    from pdf2image import convert_from_path
    from PIL import Image
    _PDF_SUPPORT = True
except ImportError:
    _PDF_SUPPORT = False


@dataclass
class PDFConversionResult:
    """Result of a PDF conversion operation."""
    success: bool
    image_paths: List[str]
    temp_directory: str
    pages_converted: int
    error_message: Optional[str] = None


@dataclass
class PDFSupportStatus:
    """Status of PDF processing capabilities."""
    python_packages_available: bool
    system_dependencies_available: bool
    fully_supported: bool
    installation_instructions: str


class PDFConverterError(Exception):
    """Base exception for PDF conversion errors."""
    pass


class PDFDependencyError(PDFConverterError):
    """Exception raised when required dependencies are missing."""
    pass


class PDFConversionError(PDFConverterError):
    """Exception raised when PDF conversion fails."""
    pass


class PDFConverter(ABC):
    """
    Abstract base class for PDF converters.

    This allows for different conversion strategies and easy testing.
    """

    @abstractmethod
    def convert_to_images(self, pdf_path: str) -> PDFConversionResult:
        """
        Convert PDF to images.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            PDFConversionResult with conversion details
        """
        pass

    @abstractmethod
    def check_support(self) -> PDFSupportStatus:
        """
        Check if PDF processing is supported.

        Returns:
            PDFSupportStatus with support details
        """
        pass


class DefaultPDFConverter(PDFConverter):
    """
    Default PDF converter implementation using pdf2image and poppler-utils.

    This class handles the actual PDF to image conversion with proper
    error handling and system dependency management.
    """

    def __init__(self):
        """Initialize the PDF converter."""
        if not _PDF_SUPPORT:
            raise PDFDependencyError(
                "PDF processing not available. Install required packages:\n"
                "pip install pdf2image Pillow\n"
                "\nAlso install poppler-utils system package:\n"
                "- Ubuntu/Debian: sudo apt-get install poppler-utils\n"
                "- macOS: brew install poppler\n"
                "- Windows: Download from https://blog.alivate.com.au/poppler-windows/"
            )

    def convert_to_images(self, pdf_path: str) -> PDFConversionResult:
        """
        Convert PDF pages to images.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            PDFConversionResult with conversion details

        Raises:
            PDFConversionError: If conversion fails
        """
        if not os.path.exists(pdf_path):
            raise PDFConversionError(f"PDF file not found: {pdf_path}")

        if not self._is_valid_pdf(pdf_path):
            raise PDFConversionError(f"Invalid PDF file: {pdf_path}")

        temp_dir = None
        image_paths = []

        try:
            # Create unique temporary directory for images
            unique_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID for uniqueness
            temp_dir = tempfile.mkdtemp(prefix=f"pdf_convert_{unique_id}_")
            print(f"Converting PDF to images in unique temporary directory: {temp_dir}")

            # Convert PDF pages to PIL Images
            images = convert_from_path(pdf_path)

            if not images:
                raise PDFConversionError("No pages found in PDF file")

            # Save each page as an image with unique naming
            for i, image in enumerate(images):
                # Create unique filename with timestamp and page number
                timestamp = str(int(time.time()))[-6:]  # Last 6 digits of timestamp
                unique_suffix = f"{unique_id}_{timestamp}"
                image_path = os.path.join(temp_dir, f"page_{i+1:03d}_{unique_suffix}.png")
                image.save(image_path, "PNG")
                image_paths.append(image_path)
                print(f"Saved page {i+1} as {image_path}")

            return PDFConversionResult(
                success=True,
                image_paths=image_paths,
                temp_directory=temp_dir,
                pages_converted=len(image_paths)
            )

        except Exception as e:
            error_msg = str(e).lower()
            if "poppler" in error_msg or "unable to get page count" in error_msg:
                raise PDFConversionError(
                    f"PDF processing failed - poppler-utils not installed: {str(e)}\n\n"
                    "Install poppler-utils system package:\n"
                    "- Ubuntu/Debian: sudo apt-get install poppler-utils\n"
                    "- macOS: brew install poppler\n"
                    "- Windows: Download from https://blog.alivate.com.au/poppler-windows/\n\n"
                    "After installation, restart your application."
                )
            else:
                raise PDFConversionError(f"Failed to convert PDF to images: {str(e)}")

    def check_support(self) -> PDFSupportStatus:
        """
        Check if PDF processing is properly configured.

        Returns:
            PDFSupportStatus with support details
        """
        status = PDFSupportStatus(
            python_packages_available=_PDF_SUPPORT,
            system_dependencies_available=False,
            fully_supported=False,
            installation_instructions=""
        )

        if not _PDF_SUPPORT:
            status.installation_instructions = (
                "Install Python packages:\n"
                "pip install pdf2image Pillow\n\n"
                "Install poppler-utils system package:\n"
                "- Ubuntu/Debian: sudo apt-get install poppler-utils\n"
                "- macOS: brew install poppler\n"
                "- Windows: Download from https://blog.alivate.com.au/poppler-windows/"
            )
            return status

        # Try to detect if poppler is installed by attempting a simple conversion
        try:
            # Create a dummy PDF for testing
            test_pdf = self._create_test_pdf()

            try:
                convert_from_path(test_pdf.name, first_page=1, last_page=1)
                status.system_dependencies_available = True
                status.fully_supported = True
            except Exception:
                status.installation_instructions = (
                    "Install poppler-utils system package:\n"
                    "- Ubuntu/Debian: sudo apt-get install poppler-utils\n"
                    "- macOS: brew install poppler\n"
                    "- Windows: Download from https://blog.alivate.com.au/poppler-windows/"
                )
            finally:
                os.unlink(test_pdf.name)

        except Exception:
            status.installation_instructions = (
                "Install poppler-utils system package:\n"
                "- Ubuntu/Debian: sudo apt-get install poppler-utils\n"
                "- macOS: brew install poppler\n"
                "- Windows: Download from https://blog.alivate.com.au/poppler-windows/"
            )

        return status

    def cleanup_images(self, image_paths: List[str]) -> None:
        """
        Clean up temporary image files and their unique directories.

        Args:
            image_paths: List of unique image file paths to clean up
        """
        cleaned_dirs = set()

        for image_path in image_paths:
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
                    print(f"Cleaned up unique image: {image_path}")
            except Exception as e:
                print(f"Warning: Failed to cleanup {image_path}: {e}")

            # Clean up temp directory if it's empty and unique to this operation
            temp_dir = os.path.dirname(image_path)
            if (os.path.exists(temp_dir) and
                temp_dir not in cleaned_dirs and
                'pdf_convert_' in temp_dir):  # Only clean up our unique directories

                try:
                    if not os.listdir(temp_dir):
                        os.rmdir(temp_dir)
                        cleaned_dirs.add(temp_dir)
                        print(f"Cleaned up unique temp directory: {temp_dir}")
                    else:
                        print(f"Directory {temp_dir} not empty, keeping it")
                except Exception as e:
                    print(f"Warning: Failed to cleanup directory {temp_dir}: {e}")

    def _is_valid_pdf(self, pdf_path: str) -> bool:
        """Check if file is a valid PDF."""
        try:
            with open(pdf_path, 'rb') as f:
                header = f.read(8)
                return header.startswith(b'%PDF-')
        except Exception:
            return False

    def _create_test_pdf(self) -> tempfile.NamedTemporaryFile:
        """Create a minimal test PDF for dependency checking with unique name."""
        unique_id = str(uuid.uuid4())[:8]
        test_pdf = tempfile.NamedTemporaryFile(suffix=f'_{unique_id}.pdf', delete=False)
        test_pdf.write(b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n72 720 Td\n/F0 12 Tf\n(Hello World) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000200 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n274\n%%EOF")
        test_pdf.close()
        return test_pdf


class PDFFileManager:
    """
    Context manager for handling PDF file operations with automatic cleanup.

    This class provides a clean interface for PDF processing operations
    with automatic resource management.
    """

    def __init__(self, pdf_path: str, converter: Optional[PDFConverter] = None):
        """
        Initialize PDF file manager.

        Args:
            pdf_path: Path to the PDF file
            converter: PDF converter instance (creates default if None)
        """
        self.pdf_path = pdf_path
        self.converter = converter or DefaultPDFConverter()
        self._conversion_result: Optional[PDFConversionResult] = None

    def __enter__(self) -> List[str]:
        """Enter context and convert PDF to images."""
        self._conversion_result = self.converter.convert_to_images(self.pdf_path)
        if not self._conversion_result.success:
            raise PDFConversionError(self._conversion_result.error_message or "Conversion failed")
        return self._conversion_result.image_paths

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and cleanup resources."""
        if self._conversion_result:
            self.converter.cleanup_images(self._conversion_result.image_paths)


# Convenience functions for easy usage
def convert_pdf_to_images(pdf_path: str) -> List[str]:
    """
    Convenience function to convert PDF to images.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of image file paths

    Raises:
        PDFConverterError: If conversion fails
    """
    with PDFFileManager(pdf_path) as image_paths:
        return image_paths.copy()  # Return copy to avoid cleanup issues


def check_pdf_support() -> Dict[str, Any]:
    """
    Check PDF processing support status.

    Returns:
        Dictionary with support status and instructions
    """
    try:
        converter = DefaultPDFConverter()
        status = converter.check_support()

        return {
            "python_packages": status.python_packages_available,
            "poppler_installed": status.system_dependencies_available,
            "fully_supported": status.fully_supported,
            "installation_instructions": status.installation_instructions
        }
    except Exception as e:
        return {
            "python_packages": False,
            "poppler_installed": False,
            "fully_supported": False,
            "installation_instructions": f"Error checking PDF support: {str(e)}"
        }


def get_pdf_conversion_help() -> str:
    """
    Get help text for PDF processing capabilities.

    Returns:
        String with PDF processing information
    """
    return (
        "PDF files are fully supported! Each page is automatically converted to an image,\n"
        "processed individually, and the results are concatenated into a single response.\n"
        "No manual conversion needed - just upload your PDF directly."
    )