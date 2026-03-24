"""Model loader for Qwen3-VL and InternVL3."""

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from typing import Optional
import structlog

from config.settings import settings

logger = structlog.get_logger()


class ModelLoader:
    """Handles loading and initialization of Qwen3-VL or InternVL3 model."""

    def __init__(self):
        self.model = None
        self.processor = None
        self._loaded = False
        self._model_type = settings.model.model_type

    def load(self) -> None:
        """Load the model and processor based on model_type."""
        if self._loaded:
            logger.info("model_already_loaded")
            return

        if self._model_type == "internvl3":
            self._load_internvl()
        else:
            self._load_qwen3vl()

        self._loaded = True

    def _load_qwen3vl(self) -> None:
        """Load Qwen3-VL model."""
        model_name = settings.model.name
        logger.info("loading_qwen3vl_model", model_name=model_name)

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=settings.model.trust_remote_code,
        )

        # Configure model loading based on quantization
        load_kwargs = {
            "trust_remote_code": settings.model.trust_remote_code,
        }

        # Use dtype from model config instead of torch_dtype
        load_kwargs["torch_dtype"] = getattr(torch, settings.model.dtype)

        if settings.model.quantization == "int8":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        elif settings.model.quantization == "int4":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, settings.model.dtype),
                bnb_4bit_use_double_quant=True,
            )

        # Load Qwen3VL model with specific class
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            **load_kwargs,
        )

        self.model.eval()
        logger.info("model_loaded_successfully", model=model_name)

    def _load_internvl(self) -> None:
        """Load InternVL3-1B model."""
        from .internvl_engine import InternVLEngine

        model_path = settings.model.internvl_model_path
        logger.info("loading_internvl_model", model_path=model_path)

        engine = InternVLEngine(model_path=model_path)
        engine.load()
        self.model = engine  # Store the engine as the "model"

        logger.info("internvl_model_loaded_successfully", model_path=model_path)

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            if self._model_type == "internvl3":
                # For InternVL, call unload on the engine
                self.model.unload()
            else:
                del self.model
                del self.processor
                self.processor = None
            self.model = None
            self._loaded = False
            torch.cuda.empty_cache()
            logger.info("model_unloaded")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def get_model(self):
        """Get the loaded model instance."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.model

    def get_processor(self):
        """Get the loaded processor instance."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.processor
