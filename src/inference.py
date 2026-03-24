"""Inference engine for Qwen3-VL and InternVL3 models."""

import torch
from PIL import Image
from typing import Union, List, Dict, Any, Optional
import structlog
import json

from .model_loader import ModelLoader
from config.settings import settings

logger = structlog.get_logger()


class InferenceEngine:
    """Handles inference with Qwen3-VL or InternVL3 model."""

    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader
        self.model_type = settings.model.model_type
        self.processor = model_loader.get_processor
        self.model = model_loader.get_model

    def _ensure_loaded(self) -> None:
        """Ensure model is loaded."""
        if not self.model_loader.is_loaded:
            self.model_loader.load()

    @torch.no_grad()
    def process_image(
        self,
        image: Union[Image.Image, str, List[Image.Image]],
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """
        Process an image (or batch of images) with a text prompt.

        Args:
            image: PIL Image, path to image, or list of images
            prompt: Text prompt to guide the model
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        self._ensure_loaded()

        if self.model_type == "internvl3":
            return self._process_internvl(image, prompt, max_tokens)
        else:
            return self._process_qwen(image, prompt, max_tokens, temperature)

    def _process_internvl(
        self,
        image: Union[Image.Image, str, List[Image.Image]],
        prompt: str,
        max_tokens: int = 1024,
    ) -> str:
        """Process image with InternVL3-1B."""
        # InternVLEngine stores itself as the model
        engine = self.model()
        if isinstance(image, (str, Image.Image)):
            image = [image]

        # Load images from paths if needed
        images = []
        for img in image:
            if isinstance(img, str):
                images.append(Image.open(img))
            else:
                images.append(img)

        # Process each image - InternVL doesn't support true batching
        results = []
        for img in images:
            result = engine.process_image(img, prompt, max_tokens)
            results.append(result)

        return results[0] if len(results) == 1 else results

    def _process_qwen(
        self,
        image: Union[Image.Image, str, List[Image.Image]],
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """Process image with Qwen3-VL."""
        # Handle single image
        if isinstance(image, (str, Image.Image)):
            image = [image]

        # Load images from paths if needed
        images = []
        for img in image:
            if isinstance(img, str):
                images.append(Image.open(img))
            else:
                images.append(img)

        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in images],
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template
        text = self.processor().apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor()(
            text=text,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        # Move to device
        inputs = {k: v.to(self.model().device) for k, v in inputs.items()}

        # Generate
        outputs = self.model().generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

        # Decode (remove input tokens)
        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        response = self.processor().batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        logger.debug("inference_completed", response_length=len(response))
        return response

    @torch.no_grad()
    def process_batch(
        self,
        images: List[Image.Image],
        prompt: str,
        max_tokens: int = 1024,
    ) -> List[str]:
        """
        Process a batch of images.

        Args:
            images: List of PIL Images
            prompt: Text prompt to guide the model
            max_tokens: Maximum tokens to generate

        Returns:
            List of generated text responses
        """
        self._ensure_loaded()

        if self.model_type == "internvl3":
            # InternVL doesn't support true batch processing
            engine = self.model()
            results = []
            for img in images:
                result = engine.process_image(img, prompt, max_tokens)
                results.append(result)
            return results
        else:
            return self._process_batch_qwen(images, prompt, max_tokens)

    def _process_batch_qwen(
        self,
        images: List[Image.Image],
        prompt: str,
        max_tokens: int = 1024,
    ) -> List[str]:
        """Process batch with Qwen3-VL."""
        # Prepare messages with all images
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in images],
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template
        text = self.processor().apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor()(
            text=text,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        # Move to device
        inputs = {k: v.to(self.model().device) for k, v in inputs.items()}

        # Generate
        outputs = self.model().generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=False,
        )

        # Decode
        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        responses = self.processor().batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        return responses


class InferenceEngineLite:
    """
    Lightweight inference engine that creates its own model instance.
    Useful for single-use or stateless inference.
    """

    def __init__(self):
        self._loader = ModelLoader()
        self._engine = None

    def __enter__(self):
        self._loader.load()
        self._engine = InferenceEngine(self._loader)
        return self._engine

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._loader.unload()
        return False

    @torch.no_grad()
    def quick_extract(
        self,
        image_path: str,
        prompt: str,
        expected_format: str = "json",
    ) -> Dict[str, Any]:
        """
        Quick single-image extraction.

        Args:
            image_path: Path to the image file
            prompt: Extraction prompt
            expected_format: Expected output format (json or text)

        Returns:
            Parsed response
        """
        with self:
            response = self._engine.process_image(image_path, prompt)

            if expected_format == "json":
                # Try to parse as JSON
                try:
                    # Handle potential markdown code blocks
                    cleaned = response.strip()
                    if cleaned.startswith("```json"):
                        cleaned = cleaned[7:]
                    if cleaned.startswith("```"):
                        cleaned = cleaned[3:]
                    if cleaned.endswith("```"):
                        cleaned = cleaned[:-3]

                    return json.loads(cleaned.strip())
                except json.JSONDecodeError as e:
                    logger.warning("json_parse_failed", error=str(e), response=response)
                    return {"raw_response": response, "parse_error": str(e)}

            return {"response": response}
