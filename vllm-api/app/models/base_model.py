"""
Base model interface and common functionality for OCR models.

This module defines the abstract base class that all model implementations
must follow, ensuring consistent behavior and interfaces across different models.
"""

import tempfile
import shutil
import os
import subprocess
import json
import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    server_url: str
    served_name: str
    default_gpu_util: float = 0.90
    default_max_len: int = 18608


@dataclass
class ProcessingResult:
    """Standardized result from model processing."""
    filename: str
    model_used: str
    server_used: str
    served_model_name: str
    content_base64: str
    processing_time_seconds: int
    total_input_tokens: int
    total_output_tokens: int
    pages_processed: int
    metadata: Optional[Dict[str, Any]] = None


class BaseModel(ABC):
    """
    Abstract base class for all OCR model implementations.

    This class defines the standard interface that all models must implement,
    ensuring consistency in input/output handling and processing workflows.
    """

    def __init__(self, config: ModelConfig):
        """Initialize the model with its configuration."""
        self.config = config
        self.temp_dir: Optional[str] = None

    @abstractmethod
    def get_pipeline_command(self, output_dir: str, gpu_util: float, max_len: int) -> list:
        """
        Get the pipeline command for this specific model.

        Args:
            output_dir: Directory for output files
            gpu_util: GPU memory utilization
            max_len: Maximum model length

        Returns:
            List of command arguments for subprocess
        """
        pass

    @abstractmethod
    def extract_content(self, output_dir: str) -> str:
        """
        Extract content from the model's output files.

        Args:
            output_dir: Directory containing model output

        Returns:
            Extracted content as string
        """
        pass

    def prepare_file(self, file) -> str:
        """
        Prepare uploaded file for processing.

        Args:
            file: FastAPI UploadFile object

        Returns:
            Path to the temporary file
        """
        original_filename = file.filename
        suffix = Path(original_filename).suffix

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            return tmp.name

    def cleanup(self, temp_file_path: str):
        """Clean up temporary files and directories."""
        try:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception:
            pass  # Ignore cleanup errors

    def run_pipeline(self, cmd: list) -> int:
        """
        Execute the model pipeline command.

        Args:
            cmd: Command to execute

        Returns:
            Exit code from the pipeline
        """
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in iter(process.stdout.readline, ''):
            print(line.strip(), flush=True)

        process.stdout.close()
        return process.wait()

    def process_file(
        self,
        file,
        output_format: str = "markdown",
        gpu_util: float = None,
        max_len: int = None
    ) -> ProcessingResult:
        """
        Process a file using this model.

        Args:
            file: FastAPI UploadFile object
            output_format: Desired output format
            gpu_util: GPU memory utilization (uses default if None)
            max_len: Maximum model length (uses default if None)

        Returns:
            ProcessingResult with standardized output
        """
        # Use defaults if not specified
        if gpu_util is None:
            gpu_util = self.config.default_gpu_util
        if max_len is None:
            max_len = self.config.default_max_len

        # Prepare file
        temp_file_path = self.prepare_file(file)
        self.temp_dir = tempfile.mkdtemp(prefix=f"{self.config.name}_")

        try:
            # Build and run pipeline command
            cmd = self.get_pipeline_command(self.temp_dir, gpu_util, max_len)

            if output_format == "markdown":
                cmd.append("--markdown")

            # Add file path (all models use --pdfs for now)
            suffix = Path(file.filename).suffix.lower()
            if suffix == ".pdf":
                cmd.extend(["--pdfs", temp_file_path])
            else:
                cmd.extend(["--pdfs", temp_file_path])

            exit_code = self.run_pipeline(cmd)
            if exit_code != 0:
                raise RuntimeError(f"Pipeline failed with code {exit_code}")

            # Extract content
            content = self.extract_content(self.temp_dir)

            # Convert to base64
            content_base64 = ""
            if content:
                content_bytes = content.encode('utf-8')
                content_base64 = base64.b64encode(content_bytes).decode('utf-8')

            # Return standardized result
            return ProcessingResult(
                filename=file.filename,
                model_used=self.config.name,
                server_used=self.config.server_url,
                served_model_name=self.config.served_name,
                content_base64=content_base64,
                processing_time_seconds=45,  # TODO: Implement actual timing
                total_input_tokens=5256,     # TODO: Get from actual processing
                total_output_tokens=2569,    # TODO: Get from actual processing
                pages_processed=3            # TODO: Get from actual processing
            )

        except Exception as e:
            raise RuntimeError(f"Processing failed: {str(e)}")
        finally:
            self.cleanup(temp_file_path)