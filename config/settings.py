"""Simplified configuration settings for web screenshot parser."""

from pathlib import Path
from pydantic import BaseModel


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


# Browser configuration
class BrowserConfig(BaseModel):
    type: str = "chromium"
    headless: bool = True
    viewport_width: int = 1920
    viewport_height: int = 1080
    full_page: bool = True
    wait_until: str = "networkidle"
    wait_timeout: int = 30000  # ms
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


# API configuration
class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 18765
    provider: str = "claude"  # "claude", "gpt", "gemini"
    api_key: str = ""         # from environment if empty
    model: str = ""           # provider-specific default


# All configurations (simplified)
class Settings(BaseModel):
    browser: BrowserConfig = BrowserConfig()
    api: APIConfig = APIConfig()


# Global settings instance
settings = Settings()
