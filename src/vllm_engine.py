"""vLLM-based inference engine for high-performance inference."""

from typing import List, Dict, Any, Union, Optional
from PIL import Image
import io
import base64
from loguru import logger



# Check if vllm is available
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    # Only log debug info, not warning - vLLM is optional
    logger.debug("vllm_not_installed_skipping")


class VLLMEngine:
    """
    High-performance inference engine using vLLM.

    vLLM provides 5-10x speedup through PagedAttention and KV cache management.
    """

    def __init__(self, model_name: str, tensor_parallel_size: int = 1):
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM not installed. Run: pip install vllm>=0.6.0"
            )

        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.llm = None
        self._loaded = False

    def load(self) -> None:
        """Load the model with vLLM."""
        if self._loaded:
            return

        logger.info("loading_model_vllm", model=self.model_name)

        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=8192,  # Adjust based on needs
        )

        self._loaded = True
        logger.info("model_loaded_vllm", model=self.model_name)

    def unload(self) -> None:
        """Unload the model."""
        if self.llm is not None:
            del self.llm
            self.llm = None
            self._loaded = False
            logger.info("model_unloaded_vllm")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """Encode PIL image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def _prepare_multimodal_prompt(
        self,
        images: List[Image.Image],
        prompt: str,
    ) -> str:
        """
        Prepare prompt for vLLM with images.

        Note: vLLM multimodal support for Qwen2.5-VL requires specific handling.
        This is a simplified version - check vLLM docs for latest API.
        """
        # For Qwen VL models in vLLM, use the multimodal prefix
        image_data = [self._encode_image_to_base64(img) for img in images]

        # Build prompt with image tokens
        prompt_parts = []
        for img_b64 in image_data:
            prompt_parts.append(f"<img>{img_b64}</img>")
        prompt_parts.append(prompt)

        return "".join(prompt_parts)

    def process_image(
        self,
        image: Union[Image.Image, str],
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """
        Process single image with vLLM.

        Args:
            image: PIL Image or path to image
            prompt: Text prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        self.load()

        # Load image if path
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # Prepare prompt
        full_prompt = self._prepare_multimodal_prompt([image], prompt)

        # Sampling params
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0.001,
            stop=["</s>", "```"],
        )

        # Generate
        outputs = self.llm.generate([full_prompt], sampling_params)
        return outputs[0].outputs[0].text

    def process_batch(
        self,
        images: List[Image.Image],
        prompt: str,
        max_tokens: int = 1024,
    ) -> List[str]:
        """
        Process batch of images with vLLM.

        For batch processing, vLLM can process multiple requests in parallel
        with significant throughput gains.

        Args:
            images: List of PIL Images
            prompt: Text prompt (same for all images)
            max_tokens: Max tokens to generate

        Returns:
            List of generated texts
        """
        self.load()

        # Prepare prompts for each image
        prompts = [
            self._prepare_multimodal_prompt([img], prompt)
            for img in images
        ]

        # Sampling params
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.1,
            stop=["</s>", "```"],
        )

        # Generate batch
        outputs = self.llm.generate(prompts, sampling_params)

        return [output.outputs[0].text for output in outputs]


class VLLMEngineLite:
    """Context manager for easy single-use inference with vLLM."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        self.model_name = model_name
        self._engine = None

    def __enter__(self):
        self._engine = VLLMEngine(self.model_name)
        self._engine.load()
        return self._engine

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._engine:
            self._engine.unload()
        return False
