"""Screenshot capture from web pages using Playwright."""

import asyncio
from typing import Optional, Union, List
from pathlib import Path

from loguru import logger
from PIL import Image

from .browser import BrowserContext
from config.settings import settings




class ScreenshotCapture:
    """
    Captures screenshots from web pages using Playwright.

    Example:
        capture = ScreenshotCapture()
        image = await capture.capture("https://example.com")
        image.save("screenshot.png")
    """

    def __init__(self):
        self._page = None

    async def capture(
        self,
        url: str,
        full_page: bool = None,
        wait_until: str = None,
        wait_timeout: int = None,
        screenshot_path: Optional[str] = None,
    ) -> Image.Image:
        """
        Capture screenshot from a URL.

        Args:
            url: Target URL
            full_page: Capture full scrollable page (default from config)
            wait_until: When to consider load complete (default from config)
            wait_timeout: Timeout in ms (default from config)
            screenshot_path: Optional path to save screenshot

        Returns:
            PIL Image of the page
        """
        full_page = full_page if full_page is not None else settings.browser.full_page
        wait_until = wait_until or settings.browser.wait_until
        wait_timeout = wait_timeout or settings.browser.wait_timeout

        logger.info("capturing_screenshot", url=url, full_page=full_page)

        async with BrowserContext() as page:
            # Set up console message logging
            page.on("console", lambda msg: logger.debug("browser_console", type=msg.type, text=msg.text))

            # Navigate to URL
            await page.goto(
                url,
                wait_until=wait_until,
                timeout=wait_timeout,
            )

            # Wait a bit for dynamic content
            await page.wait_for_timeout(1000)

            # Capture screenshot
            screenshot_bytes = await page.screenshot(
                full_page=full_page,
                type="png",
            )

        # Convert bytes to PIL Image
        from io import BytesIO
        image = Image.open(BytesIO(screenshot_bytes)).convert("RGB")

        # Save if path specified
        if screenshot_path:
            image.save(screenshot_path)
            logger.info("screenshot_saved", path=screenshot_path)

        return image

    async def capture_multiple(
        self,
        urls: List[str],
        full_page: bool = None,
    ) -> List[Image.Image]:
        """
        Capture screenshots from multiple URLs.

        Args:
            urls: List of target URLs
            full_page: Capture full scrollable page

        Returns:
            List of PIL Images
        """
        logger.info("capturing_multiple", count=len(urls))

        images = []
        for url in urls:
            try:
                image = await self.capture(url, full_page=full_page)
                images.append(image)
            except Exception as e:
                logger.error("screenshot_failed", url=url, error=str(e))
                # Add None for failed captures
                images.append(None)

        return images

    def capture_sync(
        self,
        url: str,
        full_page: bool = None,
        wait_until: str = None,
        wait_timeout: int = None,
        screenshot_path: Optional[str] = None,
    ) -> Image.Image:
        """
        Synchronous wrapper for capture.

        Note: This blocks the event loop. Use async capture() in production.
        """
        return asyncio.run(
            self.capture(
                url=url,
                full_page=full_page,
                wait_until=wait_until,
                wait_timeout=wait_timeout,
                screenshot_path=screenshot_path,
            )
        )


class ScreenshotCaptureLite:
    """
    Quick screenshot capture with minimal setup.

    Example:
        image = ScreenshotCaptureLite().capture("https://example.com")
    """

    @staticmethod
    def capture(
        url: str,
        full_page: bool = True,
        screenshot_path: Optional[str] = None,
    ) -> Image.Image:
        """
        Capture screenshot in one line.

        Args:
            url: Target URL
            full_page: Capture full page
            screenshot_path: Optional save path

        Returns:
            PIL Image
        """
        capture = ScreenshotCapture()
        return capture.capture_sync(
            url=url,
            full_page=full_page,
            screenshot_path=screenshot_path,
        )
