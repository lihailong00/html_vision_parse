"""HTML to screenshot rendering using Playwright."""

import asyncio
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, Union
from PIL import Image
import structlog

from .browser import BrowserContext
from config.settings import settings

logger = structlog.get_logger()


class HTMLRenderer:
    """
    Render HTML content (string or file) to screenshot for extraction.

    Usage:
        renderer = HTMLRenderer()

        # From HTML string
        image = renderer.render_from_html("<html><body><h1>Hello</h1></body></html>")

        # From HTML file
        image = renderer.render_from_file("/path/to/file.html")

        # Save screenshot
        renderer.render_from_html(html, save_path="output.png")
    """

    def __init__(self):
        pass

    async def render_from_html(
        self,
        html_content: str,
        screenshot_path: Optional[str] = None,
        viewport_width: Optional[int] = None,
        viewport_height: Optional[int] = None,
    ) -> Image.Image:
        """
        Render HTML content string to screenshot.

        Args:
            html_content: HTML source as string
            screenshot_path: Optional path to save screenshot
            viewport_width: Override viewport width (default from settings)
            viewport_height: Override viewport height (default from settings)

        Returns:
            PIL Image of the rendered HTML
        """
        logger.info("rendering_html", content_length=len(html_content))

        # Build HTML with proper viewport meta tag if custom size
        if viewport_width or viewport_height:
            width = viewport_width or settings.browser.viewport_width
            height = viewport_height or settings.browser.viewport_height
            viewport_meta = f'<meta name="viewport" content="width={width}, height={height}">'
            html_content = html_content.replace('<head>', f'<head>{viewport_meta}', 1)

        # Use data URL to render HTML content
        encoded = base64.b64encode(html_content.encode('utf-8')).decode('ascii')
        data_url = f"data:text/html;base64,{encoded}"

        async with BrowserContext() as page:
            # Navigate to data URL
            await page.goto(
                data_url,
                wait_until=settings.browser.wait_until,
                timeout=settings.browser.wait_timeout,
            )

            # Wait for content to settle
            await page.wait_for_timeout(1000)

            # Capture screenshot
            screenshot_bytes = await page.screenshot(
                full_page=settings.browser.full_page,
                type="png",
            )

        # Convert bytes to PIL Image
        image = Image.open(BytesIO(screenshot_bytes)).convert("RGB")

        if screenshot_path:
            image.save(screenshot_path)
            logger.info("screenshot_saved", path=screenshot_path)

        return image

    async def render_from_file(
        self,
        html_file_path: Union[str, Path],
        screenshot_path: Optional[str] = None,
    ) -> Image.Image:
        """
        Render HTML file to screenshot.

        Args:
            html_file_path: Path to HTML file
            screenshot_path: Optional path to save screenshot

        Returns:
            PIL Image of the rendered HTML
        """
        html_path = Path(html_file_path)
        if not html_path.exists():
            raise FileNotFoundError(f"HTML file not found: {html_file_path}")

        logger.info("rendering_html_file", path=str(html_path))

        # Read HTML content
        html_content = html_path.read_text(encoding='utf-8')

        return await self.render_from_html(html_content, screenshot_path)

    def render_from_html_sync(
        self,
        html_content: str,
        screenshot_path: Optional[str] = None,
        viewport_width: Optional[int] = None,
        viewport_height: Optional[int] = None,
    ) -> Image.Image:
        """Synchronous wrapper for render_from_html."""
        return asyncio.run(self.render_from_html(
            html_content, screenshot_path, viewport_width, viewport_height
        ))

    def render_from_file_sync(
        self,
        html_file_path: Union[str, Path],
        screenshot_path: Optional[str] = None,
    ) -> Image.Image:
        """Synchronous wrapper for render_from_file."""
        return asyncio.run(self.render_from_file(html_file_path, screenshot_path))
