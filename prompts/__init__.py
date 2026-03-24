"""Prompts module."""

from .extraction_prompt import (
    get_extraction_prompt,
    get_layout_detection_prompt,
    get_validation_prompt,
)

__all__ = [
    "get_extraction_prompt",
    "get_layout_detection_prompt",
    "get_validation_prompt",
]
