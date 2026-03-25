"""Playwright browser management for web scraping."""

import asyncio
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from loguru import logger

from config.settings import settings



# Check if playwright is available
try:
    from playwright.async_api import async_playwright, Browser, Page, Playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("playwright_not_installed")


class BrowserManager:
    """
    Manages Playwright browser lifecycle.

    Usage:
        async with BrowserManager() as browser:
            page = await browser.new_page()
            await page.goto("https://example.com")
    """

    def __init__(self):
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "Playwright not installed. Run: uv pip install playwright && playwright install chromium"
            )
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
        return False

    async def start(self) -> None:
        """Start the browser."""
        logger.info("starting_browser", type=settings.browser.type, headless=settings.browser.headless)

        self._playwright = await async_playwright().start()

        browser_type = getattr(self._playwright, settings.browser.type)
        self._browser = await browser_type.launch(
            headless=settings.browser.headless,
        )

        logger.info("browser_started")

    async def stop(self) -> None:
        """Stop the browser."""
        if self._browser:
            await self._browser.close()
            self._browser = None

        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

        logger.info("browser_stopped")

    @property
    def browser(self) -> Browser:
        """Get the browser instance."""
        if not self._browser:
            raise RuntimeError("Browser not started. Use 'async with' context manager.")
        return self._browser

    async def new_page(self) -> Page:
        """Create a new page with configured viewport and user agent."""
        page = await self.browser.new_page(
            viewport={"width": settings.browser.viewport_width, "height": settings.browser.viewport_height},
            user_agent=settings.browser.user_agent,
        )
        return page


class BrowserContext:
    """
    Context for a single page operation.

    Example:
        async with BrowserContext() as page:
            await page.goto("https://example.com")
            screenshot = await page.screenshot()
    """

    def __init__(self):
        self._manager: Optional[BrowserManager] = None
        self._page: Optional[Page] = None

    async def __aenter__(self) -> Page:
        self._manager = BrowserManager()
        await self._manager.start()
        self._page = await self._manager.new_page()
        return self._page

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._page:
            await self._page.close()
        if self._manager:
            await self._manager.stop()
        return False

    @property
    def page(self) -> Page:
        """Get the page instance."""
        if not self._page:
            raise RuntimeError("Context not entered. Use 'async with'.")
        return self._page
