"""
DotsOCR model implementation for external VLLM servers.

This module contains the implementation for processing files with external
VLLM servers using the OpenAI-compatible API, rather than running local
dotsocr.pipeline subprocesses.
"""

import json
import requests
import base64
import time
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Optional
from .base_model import BaseModel, ModelConfig, ProcessingResult
from .pdf_converter import (
    PDFFileManager,
    check_pdf_support,
    get_pdf_conversion_help,
    PDFConverterError
)


class DotsOCRModel(BaseModel):
    """
    DotsOCR model implementation for external VLLM servers.

    This class handles communication with external VLLM servers using
    the OpenAI-compatible API endpoints.

    Supported file formats:
    - Images: JPG, JPEG, PNG, GIF, WEBP
    - PDFs: Automatically converts each page to images and processes them

    Requirements for PDF processing:
    - Python packages: pdf2image, Pillow
    - System package: poppler-utils (Ubuntu/Debian: sudo apt-get install poppler-utils)

    For PDF files, each page is converted to an image, processed individually,
    and the results are concatenated into a single response with page separators.
    """

    @classmethod
    def get_supported_formats(cls) -> list:
        """
        Get list of supported file formats.

        Returns:
            List of supported file extensions
        """
        return ['jpg', 'jpeg', 'png', 'gif', 'webp', 'pdf']

    def convert_pdf_to_images(self, pdf_path: str) -> List[str]:
        """
        Convert PDF pages to images using the PDF utility library.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of paths to generated image files

        Raises:
            RuntimeError: If PDF processing is not available or conversion fails
        """
        try:
            # Use the PDF utility library for conversion
            image_paths = []
            with PDFFileManager(pdf_path) as images:
                image_paths = images.copy()  # Return copy to avoid cleanup issues
            return image_paths

        except PDFConverterError as e:
            raise RuntimeError(str(e))
        except Exception as e:
            raise RuntimeError(f"Unexpected error during PDF conversion: {str(e)}")

    def process_pdf_pages(self, pdf_path: str, output_format: str, prompt: str = None) -> List[Dict[str, Any]]:
        """
        Process each page of a PDF by converting to images and calling the model with concurrent batching.

        Args:
            pdf_path: Path to the PDF file
            output_format: Desired output format
            prompt: Custom prompt (optional)

        Returns:
            List of API responses, one per page in correct order
        """
        # Use PDF utility library for conversion and automatic cleanup
        try:
            with PDFFileManager(pdf_path) as image_paths:
                return self._process_pdf_pages_with_images(image_paths, output_format, prompt)
        except PDFConverterError as e:
            raise RuntimeError(f"PDF processing failed: {str(e)}")

    def _process_pdf_pages_with_images(self, image_paths: List[str], output_format: str, prompt: str = None) -> List[Dict[str, Any]]:
        """
        Process PDF pages using true batch processing (max 5 concurrent per batch).

        Args:
            image_paths: List of image file paths
            output_format: Desired output format
            prompt: Custom prompt (optional)

        Returns:
            List of API responses, one per page in correct order
        """
        print(f"Processing {len(image_paths)} pages using batch processing (max 5 per batch)")

        # Prepare all page data with their indices to maintain order
        pages_data = []
        for i, image_path in enumerate(image_paths):
            print(f"Preparing PDF page {i+1}/{len(image_paths)}")

            # Update prompt to include page information
            page_prompt = prompt or self._get_default_prompt(output_format)
            if len(image_paths) > 1:
                page_prompt = f"This is page {i+1} of {len(image_paths)}.\n\n{page_prompt}"

            pages_data.append({
                'image_path': image_path,
                'page_index': i,
                'prompt': page_prompt,
                'output_format': output_format
            })

        # Process pages in batches of maximum 5 concurrent requests
        all_results = [None] * len(image_paths)  # Pre-allocate for ordered results
        batch_size = 5

        for batch_start in range(0, len(pages_data), batch_size):
            batch_end = min(batch_start + batch_size, len(pages_data))
            batch = pages_data[batch_start:batch_end]

            print(f"Processing batch {batch_start//batch_size + 1}: pages {batch_start+1}-{batch_end}")

            # Process this batch concurrently and wait for ALL responses
            batch_results = self._process_batch_concurrently(batch)

            # Place results in correct positions maintaining order
            for page_data, result in zip(batch, batch_results):
                all_results[page_data['page_index']] = result

        return all_results

    def _process_batch_concurrently(self, batch_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of pages concurrently, waiting for ALL to complete.

        Args:
            batch_pages: List of page data dictionaries to process

        Returns:
            List of API responses in the same order as input
        """
        # Use ThreadPoolExecutor to process all pages in the batch concurrently
        with ThreadPoolExecutor(max_workers=len(batch_pages)) as executor:
            # Submit all tasks in the batch
            future_to_page_data = {
                executor.submit(self._process_single_page, page_data): page_data
                for page_data in batch_pages
            }

            # Wait for ALL futures to complete before returning
            # This ensures the entire batch is processed before continuing
            batch_results = []
            for future in as_completed(future_to_page_data):
                page_data = future_to_page_data[future]
                try:
                    result = future.result()
                    batch_results.append((page_data, result))
                except Exception as e:
                    raise RuntimeError(f"Failed to process page {page_data['page_index'] + 1}: {str(e)}")

            # Sort results by page index to maintain order within the batch
            batch_results.sort(key=lambda x: x[0]['page_index'])
            return [result for _, result in batch_results]

    def _process_single_page(self, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single page using the synchronous API method."""
        return self.process_with_vllm_api(
            page_data['image_path'],
            page_data['output_format'],
            page_data['prompt']
        )


    def _get_default_prompt(self, output_format: str) -> str:
        """
        Get the default prompt for processing by reading from the prompt file.

        Args:
            output_format: Desired output format

        Returns:
            Default prompt string loaded from dotocr_prompt.md and formatted for API
        """
        try:
            # Read the prompt from the markdown file
            prompt_file_path = os.path.join(os.path.dirname(__file__), 'dotocr_prompt.md')
            with open(prompt_file_path, 'r', encoding='utf-8') as file:
                prompt_content = file.read()

            # Clean and format the prompt content for API use
            # Remove markdown headers and decorative elements
            lines = prompt_content.split('\n')
            cleaned_lines = []

            for line in lines:
                # Skip decorative lines and headers
                if line.strip() in ['⸻', '---', '***', '---', '===']:
                    continue
                if line.strip().startswith('📝 PROMPT:') or line.strip().startswith('📌 EXPECTED OUTPUT'):
                    continue
                if line.strip().startswith('🧾 PROMPT:') or line.strip().startswith('🚀'):
                    continue
                # Keep content lines
                if line.strip() and not line.strip().startswith('#'):
                    cleaned_lines.append(line.strip())

            # Join cleaned content
            cleaned_content = '\n'.join(cleaned_lines)

            # Create the final prompt
            formatted_prompt = f"""
You are an expert OCR system specialized in document processing.

{prompt_content}

IMPORTANT REQUIREMENTS:
- Process the document image and extract all textual content
- Replicate the original document structure and hierarchy in {output_format.upper()} format
- Preserve formatting, lists, tables, and layout as much as possible
- Include all visible text, numbers, and structural elements
- If something is unclear, indicate it clearly

OUTPUT FORMAT: {output_format.upper()}
"""
            return formatted_prompt.strip()

        except FileNotFoundError:
            # Fallback to a simple default prompt if file is not found
            return (
                "Please extract and structure all text content from this document image. "
                "Preserve the original layout, formatting, and hierarchy. "
                f"Output the result in {output_format.upper()} format with proper structure."
            )
        except Exception as e:
            # Fallback for any other error
            return (
                "Please extract and structure the document content from this image. "
                f"Output the result in {output_format.upper()} format."
            )

    @classmethod
    def check_pdf_support(cls) -> Dict[str, Any]:
        """
        Check if PDF processing is properly configured using the PDF utility library.

        Returns:
            Dictionary with support status and installation instructions
        """
        return check_pdf_support()

    @classmethod
    def get_pdf_conversion_help(cls) -> str:
        """
        Get help text for PDF processing capabilities using the PDF utility library.

        Returns:
            String with PDF processing information
        """
        return get_pdf_conversion_help()

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
        supported_image_formats = ['jpg', 'jpeg', 'png', 'gif', 'webp']

        if file_extension in supported_image_formats:
            mime_type = f"image/{file_extension}"
            if file_extension == 'jpg':
                mime_type = "image/jpeg"
        elif file_extension == 'pdf':
            # PDFs are handled at the process_file level, not here
            mime_type = "application/pdf"
        else:
            mime_type = "application/octet-stream"

        file_b64 = base64.b64encode(file_content).decode('utf-8')
        image_data_url = f"data:{mime_type};base64,{file_b64}"

        # Default prompt if none provided
        if prompt is None:
            prompt = self._get_default_prompt(output_format)

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
            "temperature": 0.9
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

        For PDF files, processes each page and concatenates the results.
        For image files, processes the single image.

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
        file_extension = Path(file.filename).suffix.lower().lstrip('.') if file.filename else ""

        try:
            # Process with external VLLM server
            start_time = time.time()

            if file_extension == 'pdf':
                # Process PDF pages
                page_responses = self.process_pdf_pages(temp_file_path, output_format, prompt)
                processing_time = time.time() - start_time

                # Collect all page data into a structured format
                all_pages_data = []
                total_input_tokens = 0
                total_output_tokens = 0
                pages_processed = len(page_responses)

                for i, api_response in enumerate(page_responses):
                    # Extract content from each page response
                    page_content = ""
                    if api_response.get("choices") and len(api_response["choices"]) > 0:
                        page_content = api_response["choices"][0].get("message", {}).get("content", "")

                    # Try to parse the content as JSON to get structured data
                    page_elements = []
                    if page_content:
                        try:
                            # The content should be a JSON array of elements with bbox, category, and text
                            page_elements = json.loads(page_content)
                        except json.JSONDecodeError:
                            # If it's not valid JSON, treat it as raw text
                            page_elements = [{"text": page_content, "category": "Text", "bbox": [0, 0, 0, 0]}]

                    # Add page information to each element
                    for element in page_elements:
                        element["page_number"] = i + 1
                        element["document_page"] = i + 1

                    all_pages_data.extend(page_elements)

                    # Accumulate token counts
                    total_input_tokens += api_response.get("usage", {}).get("prompt_tokens", 0)
                    total_output_tokens += api_response.get("usage", {}).get("completion_tokens", 0)

                # Create final JSON structure
                final_content = json.dumps(all_pages_data, ensure_ascii=False, indent=2)

            else:
                # Process single image using batch processing (batch of 1)
                batch_pages = [{
                    'image_path': temp_file_path,
                    'page_index': 0,
                    'prompt': prompt or self._get_default_prompt(output_format),
                    'output_format': output_format
                }]

                batch_results = self._process_batch_concurrently(batch_pages)
                api_response = batch_results[0]
                processing_time = time.time() - start_time

                # Extract and parse content from single response
                content = ""
                if api_response.get("choices") and len(api_response["choices"]) > 0:
                    content = api_response["choices"][0].get("message", {}).get("content", "")

                # Parse JSON content for structured data
                if content:
                    try:
                        page_elements = json.loads(content)
                        # Add page information to each element
                        for element in page_elements:
                            element["page_number"] = 1
                            element["document_page"] = 1
                        final_content = json.dumps(page_elements, ensure_ascii=False, indent=2)
                    except json.JSONDecodeError:
                        # If it's not valid JSON, wrap it in a simple structure
                        final_content = json.dumps([{"text": content, "category": "Text", "bbox": [0, 0, 0, 0], "page_number": 1}], ensure_ascii=False, indent=2)
                else:
                    final_content = json.dumps([], ensure_ascii=False, indent=2)

                total_input_tokens = api_response.get("usage", {}).get("prompt_tokens", 0)
                total_output_tokens = api_response.get("usage", {}).get("completion_tokens", 0)
                pages_processed = 1

            content_base64 = ""
            if final_content:
                content_bytes = final_content.encode('utf-8')
                content_base64 = base64.b64encode(content_bytes).decode('utf-8')

            # Return standardized result
            return ProcessingResult(
                filename=file.filename,
                model_used=self.config.name,
                server_used=self.config.server_url,
                served_model_name=self.config.served_name,
                content_base64=content_base64,
                processing_time_seconds=int(processing_time),
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                pages_processed=pages_processed,
                metadata={
                    "model": api_response.get("model", "") if 'api_response' in locals() else "",
                    "finish_reason": api_response.get("choices", [{}])[0].get("finish_reason", "") if 'api_response' in locals() else "",
                    "response_format": "json",  # Always return JSON structure for DotsOCR
                    "file_type": "pdf" if file_extension == 'pdf' else "image",
                    "pages_data": all_pages_data if file_extension == 'pdf' else None
                }
            )

        except Exception as e:
            raise RuntimeError(f"Processing failed: {str(e)}")
        finally:
            self.cleanup(temp_file_path)