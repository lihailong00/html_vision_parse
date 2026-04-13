"""API clients for cloud LLM services (Claude, GPT, Gemini)."""

import base64
import io
import os
from abc import ABC, abstractmethod
from typing import Optional

from loguru import logger
from PIL import Image


class LLMAPIClient(ABC):
    """Abstract base class for LLM API clients."""

    def __init__(self, api_key: str, model: Optional[str] = None):
        """
        Initialize the API client.

        Args:
            api_key: API key for the service. If None, reads from environment variable.
            model: Optional model name override.
        """
        self.api_key = api_key or self._get_api_key_from_env()
        self.model = model

    @abstractmethod
    def chat_completion(self, image_base64: str, prompt: str) -> str:
        """
        Send an image and prompt to the API and get a text response.

        Args:
            image_base64: Base64-encoded image (with or without data URL prefix).
            prompt: Text prompt to send with the image.

        Returns:
            Text response from the API.

        Raises:
            APIError: If the API call fails.
        """
        pass

    @staticmethod
    def image_to_base64(image: Image.Image) -> str:
        """
        Convert a PIL Image to a base64 string.

        Args:
            image: PIL Image to convert.

        Returns:
            Base64-encoded image string (with image/png data URL prefix).
        """
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        return f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"


class ClaudeAPIClient(LLMAPIClient):
    """Anthropic Claude API client using the Messages API."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the Claude API client.

        Args:
            api_key: Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.
            model: Model name (default: claude-sonnet-4-20250514).
        """
        super().__init__(api_key, model or "claude-sonnet-4-20250514")
        self._client = None

    def _get_api_key_from_env(self) -> str:
        """Get API key from ANTHROPIC_API_KEY environment variable."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        return api_key

    @property
    def client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def chat_completion(self, image_base64: str, prompt: str) -> str:
        """
        Send an image and prompt to Claude using the Messages API.

        Args:
            image_base64: Base64-encoded image (with or without data URL prefix).
            prompt: Text prompt to send with the image.

        Returns:
            Text response from Claude.
        """
        import anthropic

        # Extract base64 data if it's a data URL
        if image_base64.startswith("data:"):
            image_data = image_base64
        else:
            image_data = f"data:image/png;base64,{image_base64}"

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_data.split(",", 1)[1]
                                    if "," in image_data
                                    else image_base64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            return response.content[0].text
        except anthropic.APIError as e:
            logger.error("claude_api_error", error=str(e))
            raise


class GPTAPIClient(LLMAPIClient):
    """OpenAI GPT API client using the Vision API."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the GPT API client.

        Args:
            api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
            model: Model name (default: gpt-4o).
        """
        super().__init__(api_key, model or "gpt-4o")
        self._client = None

    def _get_api_key_from_env(self) -> str:
        """Get API key from OPENAI_API_KEY environment variable."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        return api_key

    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            import openai

            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    def chat_completion(self, image_base64: str, prompt: str) -> str:
        """
        Send an image and prompt to GPT-4o using the Vision API.

        Args:
            image_base64: Base64-encoded image (with or without data URL prefix).
            prompt: Text prompt to send with the image.

        Returns:
            Text response from GPT-4o.
        """
        import openai

        # Ensure proper data URL format
        if image_base64.startswith("data:"):
            image_url = image_base64
        else:
            image_url = f"data:image/png;base64,{image_base64}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except openai.APIError as e:
            logger.error("openai_api_error", error=str(e))
            raise


class GeminiAPIClient(LLMAPIClient):
    """Google Gemini API client."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the Gemini API client.

        Args:
            api_key: Google API key. Defaults to GOOGLE_API_KEY env var.
            model: Model name (default: gemini-2.0-flash).
        """
        super().__init__(api_key, model or "gemini-2.0-flash")
        self._client = None

    def _get_api_key_from_env(self) -> str:
        """Get API key from GOOGLE_API_KEY environment variable."""
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        return api_key

    @property
    def client(self):
        """Lazy initialization of Gemini client."""
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def chat_completion(self, image_base64: str, prompt: str) -> str:
        """
        Send an image and prompt to Gemini.

        Args:
            image_base64: Base64-encoded image (with or without data URL prefix).
            prompt: Text prompt to send with the image.

        Returns:
            Text response from Gemini.
        """
        from google import genai

        # Extract base64 data if it's a data URL
        if "," in image_base64:
            base64_data = image_base64.split(",", 1)[1]
        else:
            base64_data = image_base64

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    {
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": base64_data,
                                }
                            },
                            {"text": prompt},
                        ]
                    }
                ],
            )
            return response.text
        except genai.APIError as e:
            logger.error("gemini_api_error", error=str(e))
            raise
