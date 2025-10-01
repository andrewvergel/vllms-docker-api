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
from .base_model import BaseModel, ModelConfig, ProcessingResult


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

    def process_with_vllm_api(self, file_path: str, output_format: str, prompt: str = None) -> Dict[str, Any]:
        """
        Process file using external VLLM server API with OpenAI-compatible chat completions.

        Args:
            file_path: Path to the file to process
            output_format: Desired output format (markdown, json, etc.)
            prompt: Custom prompt to use for processing. If None, uses default prompt.

        Returns:
            API response from VLLM server
        """
        # Read file and convert to base64
        with open(file_path, "rb") as f:
            file_content = f.read()

        # Convert file to base64 data URL
        file_extension = Path(file_path).suffix.lower().lstrip('.')
        if file_extension in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
            mime_type = f"image/{file_extension}"
            if file_extension == 'jpg':
                mime_type = "image/jpeg"
        else:
            mime_type = "application/octet-stream"

        file_b64 = base64.b64encode(file_content).decode('utf-8')
        image_data_url = f"data:{mime_type};base64,{file_b64}"

        # Default prompt if none provided
        if prompt is None:
            prompt = (
                "Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.\n\n"
                "1. Bbox format: [x1, y1, x2, y2]\n\n"
                "2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].\n\n"
                "3. Text Extraction & Formatting Rules:\n"
                "    - Picture: For the 'Picture' category, the text field should be omitted.\n"
                "    - Formula: Format its text as LaTeX.\n"
                "    - Table: Format its text as HTML.\n"
                "    - All Others (Text, Title, etc.): Format their text as Markdown.\n\n"
                "4. Constraints:\n"
                "    - The output text must be the original text from the image, with no translation.\n"
                "    - All layout elements must be sorted according to human reading order.\n\n"
                "5. Final Output: The entire output must be a single JSON object.\n"
                f"Output format: {output_format}"
            )

        # Prepare the request payload for OpenAI-compatible chat completions API
        payload = {
            "model": self.config.served_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.7
        }

        # Make API call to external VLLM server
        try:
            response = requests.post(
                f"{self.config.server_url}/v1/chat/completions",
                json=payload,
                timeout=300  # 5 minute timeout
            )

            # Log the full response for debugging
            print(f"VLLM API Response Status: {response.status_code}")
            print(f"VLLM API Response Headers: {response.headers}")
            print(f"VLLM API Response Body: {response.text}")

            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            # Enhanced error handling to see the actual server response
            if hasattr(e, 'response') and e.response is not None:
                error_msg = f"API call to VLLM server failed: {response.status_code} {response.reason} for url: {response.url}"
                error_msg += f"\nResponse body: {response.text}"
                error_msg += f"\nRequest payload: {json.dumps(payload, indent=2)}"
                raise RuntimeError(error_msg)
            else:
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
        max_len: int = None,
        prompt: str = None
    ) -> ProcessingResult:
        """
        Process a file using external VLLM server API.

        Args:
            file: FastAPI UploadFile object
            output_format: Desired output format
            gpu_util: GPU memory utilization (parameter kept for compatibility)
            max_len: Maximum model length (parameter kept for compatibility)
            prompt: Custom prompt to use for processing. If None, uses default prompt.

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
            api_response = self.process_with_vllm_api(temp_file_path, output_format, prompt)
            processing_time = time.time() - start_time

            # Extract content from chat completions response
            content = ""
            if api_response.get("choices") and len(api_response["choices"]) > 0:
                content = api_response["choices"][0].get("message", {}).get("content", "")

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
                total_input_tokens=api_response.get("usage", {}).get("prompt_tokens", 0),
                total_output_tokens=api_response.get("usage", {}).get("completion_tokens", 0),
                pages_processed=1,  # Single image processing
                metadata={
                    "model": api_response.get("model", ""),
                    "finish_reason": api_response.get("choices", [{}])[0].get("finish_reason", ""),
                    "response_format": output_format
                }
            )

        except Exception as e:
            raise RuntimeError(f"Processing failed: {str(e)}")
        finally:
            self.cleanup(temp_file_path)