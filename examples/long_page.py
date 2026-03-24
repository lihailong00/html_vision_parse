"""
Example: Extract long page content by splitting screenshot into multiple parts.

For pages with几万字 content, single screenshot may not capture everything.
This script demonstrates splitting a long page into multiple viewport-height
chunks and extracting content from each.
"""

from PIL import Image
from src.model_loader import ModelLoader
from src.inference import InferenceEngine
from prompts.extraction_prompt import get_extraction_prompt
import re


class LongPageExtractor:
    """Extract content from long pages by splitting into chunks."""

    def __init__(self, viewport_height=1080):
        self.viewport_height = viewport_height
        self.loader = None
        self.engine = None

    def _load_model(self):
        if self.loader is None:
            self.loader = ModelLoader()
            self.loader.load()
            self.engine = InferenceEngine(self.loader)

    def _unload_model(self):
        if self.loader:
            self.loader.unload()
            self.loader = None
            self.engine = None

    def _split_image(self, full_page_img: Image.Image) -> list:
        """Split full page screenshot into viewport-height chunks."""
        width, total_height = full_page_img.size
        chunks = []

        y = 0
        while y < total_height:
            chunk_height = min(self.viewport_height, total_height - y)
            chunk = full_page_img.crop((0, y, width, y + chunk_height))
            chunks.append(chunk)
            y += self.viewport_height

        return chunks

    def _extract_from_chunk(self, img: Image.Image, prompt: str) -> str:
        """Extract text from a single chunk."""
        response = self.engine.process_image(img, prompt, max_tokens=2048)

        # Try to parse JSON
        try:
            import json
            data = json.loads(response)
            return data.get("content", "")
        except:
            # If parsing fails, try to extract content manually
            content_match = re.search(r'"content"\s*:\s*"(.*)"', response, re.DOTALL)
            if content_match:
                return content_match.group(1)
            return ""

    def extract(self, full_page_img: Image.Image) -> dict:
        """
        Extract content from a long page.

        Args:
            full_page_img: Full page screenshot (PIL Image)

        Returns:
            dict with title, content (merged), confidence
        """
        self._load_model()
        prompt = get_extraction_prompt()

        # Split into chunks
        chunks = self._split_image(full_page_img)
        print(f"Split into {len(chunks)} chunks")

        # Extract from each chunk
        contents = []
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            content = self._extract_from_chunk(chunk, prompt)
            if content:
                contents.append(content)

        # Merge contents
        merged = "\n\n".join(contents)

        return {
            "content": merged,
            "num_chunks": len(chunks),
            "content_length": len(merged)
        }

    def extract_from_file(self, image_path: str) -> dict:
        """Extract from a saved screenshot file."""
        img = Image.open(image_path).convert("RGB")
        return self.extract(img)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python long_page.py <screenshot.png>")
        sys.exit(1)

    extractor = LongPageExtractor()
    result = extractor.extract_from_file(sys.argv[1])

    print(f"\n=== RESULT ===")
    print(f"Chunks processed: {result['num_chunks']}")
    print(f"Total content length: {result['content_length']} chars")
    print(f"\nContent preview (first 500 chars):")
    print(result['content'][:500])

    extractor._unload_model()
