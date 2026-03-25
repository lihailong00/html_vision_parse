"""InternVL3-1B inference engine wrapper for the extraction pipeline."""

import time
from typing import List, Optional, Tuple
import numpy as np
import torch
from PIL import Image
from loguru import logger
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, GenerationConfig

from config.settings import settings




class InternVLEngine:
    """
    Inference engine wrapper for InternVL3-1B model.

    This model is much smaller (~1B params) than Qwen3-VL-2B (~2B params)
    and offers significantly faster inference while maintaining reasonable accuracy.
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path or "/home/longcoding/dev/models/InternVL3-1B"
        self.model = None
        self.tokenizer = None
        self.config = None
        self.device = "cuda"
        self.image_size = 448  # InternVL3-1B fixed input size

    def load(self):
        """Load the InternVL3-1B model and tokenizer."""
        logger.info("loading_internvl_model", path=self.model_path)

        t0 = time.time()

        self.config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=self.config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).cuda()

        load_time = time.time() - t0
        logger.info(
            "internvl_model_loaded",
            load_time=load_time,
            device=self.device,
            dtype="bfloat16"
        )

    def unload(self):
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()
            logger.info("internvl_model_unloaded")

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for InternVL3-1B.

        Uses the standard InternVL preprocessing:
        1. Resize so shorter side is image_size (448)
        2. Center crop to image_size x image_size
        3. Normalize with ImageNet mean/std
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        width, height = image.size

        # Resize so shorter side is image_size
        if width < height:
            new_width = self.image_size
            new_height = int(height * self.image_size / width)
        else:
            new_height = self.image_size
            new_width = int(width * self.image_size / height)

        image = image.resize((new_width, new_height), Image.LANCZOS)

        # Center crop
        left = (new_width - self.image_size) // 2
        top = (new_height - self.image_size) // 2
        image = image.crop((left, top, left + self.image_size, top + self.image_size))

        # Convert to tensor and normalize
        image_array = np.array(image).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std

        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()

        return image_tensor

    def _build_prompt(self, question: str) -> str:
        """Build the prompt for extraction."""
        return question

    def process_image(
        self,
        image: Image.Image,
        prompt: str,
        max_tokens: int = 4096,
    ) -> str:
        """
        Process a single image with the given prompt.

        Args:
            image: PIL Image of the screenshot
            prompt: Text prompt for extraction
            max_tokens: Maximum tokens to generate

        Returns:
            Model's text response
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Preprocess image
        pixel_values = self._preprocess_image(image)
        pixel_values = pixel_values.unsqueeze(0).to(self.device, dtype=torch.bfloat16)

        # Generation config
        gen_config = {
            "max_new_tokens": max_tokens,
            "do_sample": False,
        }

        # Generate
        response = self.model.chat(
            tokenizer=self.tokenizer,
            pixel_values=pixel_values,
            question=prompt,
            generation_config=gen_config,
            verbose=False
        )

        return response

    def process_batch(
        self,
        images: List[Image.Image],
        prompt: str,
    ) -> List[str]:
        """
        Process multiple images in batch.

        Note: InternVL3-1B doesn't support true batch processing in the same way
        as Qwen3-VL. This processes images sequentially.
        """
        results = []
        for image in images:
            result = self.process_image(image, prompt)
            results.append(result)
        return results


def create_internvl_engine() -> InternVLEngine:
    """Factory function to create an InternVL engine."""
    return InternVLEngine()
