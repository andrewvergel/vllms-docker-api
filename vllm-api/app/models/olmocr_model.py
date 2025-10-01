"""
OlmOCR model implementation for external VLLM servers.

This module contains the implementation for processing files with external
VLLM servers using the OpenAI-compatible API, rather than running local
olmocr.pipeline subprocesses.
"""

import json
import requests
import base64
import time
from pathlib import Path
from typing import Dict, Any
from .base_model import BaseModel, ModelConfig


class OlmOCRModel(BaseModel):
    """
    OlmOCR model implementation for external VLLM servers.

    This class handles communication with external VLLM servers using
    the OpenAI-compatible API endpoints.
    """

    def get_pipeline_command(self, output_dir: str, gpu_util: float, max_len: int) -> list:
        """
        Not used for external VLLM servers.

        For external VLLM servers, we use HTTP API calls instead of subprocess commands.
        This method is kept for interface compatibility but not used.
        """
        raise NotImplementedError("External VLLM servers use HTTP API, not subprocess commands")

    def process_with_vllm_api(self, file_path: str, output_format: str) -> Dict[str, Any]:
        """
        Process file using external VLLM server API.

        Args:
            file_path: Path to the file to process
            output_format: Desired output format (markdown, json, etc.)

        Returns:
            API response from VLLM server
        """
        # Read file and convert to base64
        with open(file_path, "rb") as f:
            file_content = f.read()

        # Prepare the request payload for VLLM API
        # This is a simplified example - adjust based on your VLLM server's API
        payload = {
            "model": self.config.served_name,
            "file": base64.b64encode(file_content).decode('utf-8'),
            "output_format": output_format,
            "max_tokens": 2000,  # Adjust as needed
        }

        # Make API call to external VLLM server
        try:
            response = requests.post(
                f"{self.config.server_url}/v1/process",  # Adjust endpoint as needed
                json=payload,
                timeout=300  # 5 minute timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API call to VLLM server failed: {str(e)}")

    def extract_content(self, output_dir: str) -> str:
        """
        For external VLLM servers, content extraction happens via API response.

        This method is kept for interface compatibility but delegates to the
        API-based processing method.
        """
        # This will be handled by process_with_vllm_api instead
        return ""

    def process_file(
        self,
        file,
        output_format: str = "markdown",
        gpu_util: float = None,
        max_len: int = None
    ) -> ProcessingResult:
        """
        Process a file using external VLLM server API.

        Args:
            file: FastAPI UploadFile object
            output_format: Desired output format
            gpu_util: GPU memory utilization (parameter kept for compatibility)
            max_len: Maximum model length (parameter kept for compatibility)

        Returns:
            ProcessingResult with standardized output
        """
        # Use defaults if not specified (kept for compatibility)
        if gpu_util is None:
            gpu_util = self.config.default_gpu_util
        if max_len is None:
            max_len = self.config.default_max_len

        # Prepare file
        temp_file_path = self.prepare_file(file)

        try:
            # Process with external VLLM server
            start_time = time.time()
            api_response = self.process_with_vllm_api(temp_file_path, output_format)
            processing_time = time.time() - start_time

            # Extract content from API response
            content = api_response.get("content", "")
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
                processing_time_seconds=int(processing_time),
                total_input_tokens=api_response.get("input_tokens", 0),
                total_output_tokens=api_response.get("output_tokens", 0),
                pages_processed=api_response.get("pages_processed", 1),
                metadata=api_response.get("metadata", {})
            )

        except Exception as e:
            raise RuntimeError(f"Processing failed: {str(e)}")
        finally:
            self.cleanup(temp_file_path)