"""YAML configuration loader with environment variable override support."""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Literal
import yaml
import structlog

from config.settings import settings, Settings

logger = structlog.get_logger()


class ConfigLoader:
    """
    Load configuration from YAML files with environment variable overrides.

    Priority (highest to lowest):
    1. Environment variables (e.g., EXTRACTION_METHOD=ocr)
    2. YAML config file
    3. Default values in settings.py

    Usage:
        loader = ConfigLoader()

        # Load from default config.yaml in project root
        loader.load()

        # Load from custom path
        loader.load("/path/to/config.yaml")

        # Override specific values
        loader.load(overrides={"extraction": {"method": "ocr"}})

        # Get merged settings
        cfg = loader.get_settings()
    """

    # Mapping of YAML config keys to environment variables
    ENV_MAPPING = {
        "model.name": "MODEL_NAME",
        "model.model_type": "MODEL_TYPE",
        "model.inference_framework": "INFERENCE_FRAMEWORK",
        "extraction.extraction_method": "EXTRACTION_METHOD",
        "extraction.field_methods": "FIELD_METHODS",
        "extraction.min_confidence": "MIN_CONFIDENCE",
        "browser.type": "BROWSER_TYPE",
        "browser.headless": "BROWSER_HEADLESS",
        "browser.viewport_width": "VIEWPORT_WIDTH",
        "browser.viewport_height": "VIEWPORT_HEIGHT",
        "api.host": "API_HOST",
        "api.port": "API_PORT",
        "api.workers": "API_WORKERS",
        "ocr.enabled": "OCR_ENABLED",
        "ocr.min_confidence": "OCR_MIN_CONFIDENCE",
        "batch.batch_size": "BATCH_SIZE",
        "batch.num_workers": "BATCH_NUM_WORKERS",
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the config loader.

        Args:
            config_path: Path to YAML config file. If None, looks for config.yaml
                     in the project root.
        """
        self._config_path = config_path
        self._yaml_config: Dict[str, Any] = {}
        self._overrides: Dict[str, Any] = {}
        self._loaded = False

    def load(
        self,
        config_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "ConfigLoader":
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file
            overrides: Dict of values to override after loading YAML

        Returns:
            self for chaining
        """
        path = config_path or self._config_path

        # Default to config.yaml in project root
        if path is None:
            project_root = Path(__file__).parent.parent
            default_path = project_root / "config.yaml"
            if default_path.exists():
                path = str(default_path)

        if path:
            path = Path(path)
            if path.exists():
                logger.info("loading_config", path=str(path))
                with open(path, 'r', encoding='utf-8') as f:
                    self._yaml_config = yaml.safe_load(f) or {}
                logger.info("config_loaded", path=str(path))
            else:
                logger.warning("config_file_not_found", path=str(path))
                self._yaml_config = {}

        # Apply overrides
        if overrides:
            self._overrides = overrides
            logger.info("config_overrides_applied", overrides=list(overrides.keys()))

        self._loaded = True
        return self

    def _apply_yaml_config(self):
        """Apply YAML configuration to settings."""
        if not self._yaml_config:
            return

        # Model config
        if "model" in self._yaml_config:
            model_cfg = self._yaml_config["model"]
            if "name" in model_cfg:
                settings.model.name = model_cfg["name"]
            if "model_type" in model_cfg:
                settings.model.model_type = model_cfg["model_type"]
            if "inference_framework" in model_cfg:
                settings.model.inference_framework = model_cfg["inference_framework"]
            if "device" in model_cfg:
                settings.model.device = model_cfg["device"]
            if "dtype" in model_cfg:
                settings.model.dtype = model_cfg["dtype"]

        # Extraction config
        if "extraction" in self._yaml_config:
            ext_cfg = self._yaml_config["extraction"]
            if "extraction_method" in ext_cfg:
                settings.extraction.extraction_method = ext_cfg["extraction_method"]
            if "field_methods" in ext_cfg:
                settings.extraction.field_methods = ext_cfg["field_methods"]
            if "min_confidence" in ext_cfg:
                settings.extraction.min_confidence = ext_cfg["min_confidence"]
            if "enable_cross_validation" in ext_cfg:
                settings.extraction.enable_cross_validation = ext_cfg["enable_cross_validation"]
            if "max_retries" in ext_cfg:
                settings.extraction.max_retries = ext_cfg["max_retries"]
            if "timeout_seconds" in ext_cfg:
                settings.extraction.timeout_seconds = ext_cfg["timeout_seconds"]

        # Browser config
        if "browser" in self._yaml_config:
            browser_cfg = self._yaml_config["browser"]
            if "type" in browser_cfg:
                settings.browser.type = browser_cfg["type"]
            if "headless" in browser_cfg:
                settings.browser.headless = browser_cfg["headless"]
            if "viewport_width" in browser_cfg:
                settings.browser.viewport_width = browser_cfg["viewport_width"]
            if "viewport_height" in browser_cfg:
                settings.browser.viewport_height = browser_cfg["viewport_height"]
            if "full_page" in browser_cfg:
                settings.browser.full_page = browser_cfg["full_page"]
            if "wait_until" in browser_cfg:
                settings.browser.wait_until = browser_cfg["wait_until"]
            if "user_agent" in browser_cfg:
                settings.browser.user_agent = browser_cfg["user_agent"]

        # API config
        if "api" in self._yaml_config:
            api_cfg = self._yaml_config["api"]
            if "host" in api_cfg:
                settings.api.host = api_cfg["host"]
            if "port" in api_cfg:
                settings.api.port = api_cfg["port"]
            if "workers" in api_cfg:
                settings.api.workers = api_cfg["workers"]
            if "max_concurrent_requests" in api_cfg:
                settings.api.max_concurrent_requests = api_cfg["max_concurrent_requests"]

        # OCR config
        if "ocr" in self._yaml_config:
            ocr_cfg = self._yaml_config["ocr"]
            if "enabled" in ocr_cfg:
                settings.ocr.enabled = ocr_cfg["enabled"]
            if "min_confidence" in ocr_cfg:
                settings.ocr.min_confidence = ocr_cfg["min_confidence"]
            if "use_easyocr" in ocr_cfg:
                settings.ocr.use_easyocr = ocr_cfg["use_easyocr"]
            if "fallback_to_vl" in ocr_cfg:
                settings.ocr.fallback_to_vl = ocr_cfg["fallback_to_vl"]
            if "ocr_timeout_seconds" in ocr_cfg:
                settings.ocr.ocr_timeout_seconds = ocr_cfg["ocr_timeout_seconds"]

        # Batch config
        if "batch" in self._yaml_config:
            batch_cfg = self._yaml_config["batch"]
            if "batch_size" in batch_cfg:
                settings.batch.batch_size = batch_cfg["batch_size"]
            if "num_workers" in batch_cfg:
                settings.batch.num_workers = batch_cfg["num_workers"]
            if "prefetch_factor" in batch_cfg:
                settings.batch.prefetch_factor = batch_cfg["prefetch_factor"]

    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        for config_key, env_var in self.ENV_MAPPING.items():
            env_value = os.environ.get(env_var)
            if env_value is None:
                continue

            # Parse value based on expected type
            parts = config_key.split(".")
            if len(parts) != 2:
                continue

            section, key = parts

            # Get the current value type from settings
            if section == "model":
                current = getattr(settings.model, key, None)
            elif section == "extraction":
                current = getattr(settings.extraction, key, None)
            elif section == "browser":
                current = getattr(settings.browser, key, None)
            elif section == "api":
                current = getattr(settings.api, key, None)
            elif section == "ocr":
                current = getattr(settings.ocr, key, None)
            elif section == "batch":
                current = getattr(settings.batch, key, None)
            else:
                continue

            # Convert string to appropriate type
            if isinstance(current, bool):
                parsed_value = env_value.lower() in ("true", "1", "yes")
            elif isinstance(current, int):
                parsed_value = int(env_value)
            elif isinstance(current, float):
                parsed_value = float(env_value)
            elif isinstance(current, dict):
                # Parse as JSON for dicts
                import json
                try:
                    parsed_value = json.loads(env_value)
                except json.JSONDecodeError:
                    logger.warning("env_var_json_parse_failed", env_var=env_var)
                    continue
            else:
                parsed_value = env_value

            # Apply the override
            if section == "model":
                setattr(settings.model, key, parsed_value)
            elif section == "extraction":
                setattr(settings.extraction, key, parsed_value)
            elif section == "browser":
                setattr(settings.browser, key, parsed_value)
            elif section == "api":
                setattr(settings.api, key, parsed_value)
            elif section == "ocr":
                setattr(settings.ocr, key, parsed_value)
            elif section == "batch":
                setattr(settings.batch, key, parsed_value)

            logger.debug("env_override_applied", key=config_key, env_var=env_var, value=parsed_value)

    def _apply_overrides(self):
        """Apply direct overrides."""
        if not self._overrides:
            return

        for section, values in self._overrides.items():
            if not isinstance(values, dict):
                continue

            for key, value in values.items():
                section_obj = getattr(settings, section, None)
                if section_obj is None:
                    continue

                setattr(section_obj, key, value)
                logger.debug("override_applied", section=section, key=key, value=value)

    def get_settings(self) -> Settings:
        """
        Get the merged settings.

        Loads config if not already loaded.

        Returns:
            Merged Settings object
        """
        if not self._loaded:
            self.load()

        self._apply_yaml_config()
        self._apply_env_overrides()
        self._apply_overrides()

        return settings

    def get_config_dict(self) -> Dict[str, Any]:
        """
        Get configuration as a dictionary (for inspection/debugging).

        Returns:
            Dict representation of current configuration
        """
        if not self._loaded:
            self.load()

        return {
            "model": settings.model.model_dump(),
            "extraction": settings.extraction.model_dump(),
            "browser": settings.browser.model_dump(),
            "api": settings.api.model_dump(),
            "ocr": settings.ocr.model_dump(),
            "batch": settings.batch.model_dump(),
        }


def create_default_config(output_path: str = "config.yaml"):
    """
    Create a default configuration file.

    Args:
        output_path: Path where to write the default config
    """
    default_config = {
        "model": {
            "model_type": "qwen3_vl",
            "name": "/home/longcoding/dev/models/Qwen3-VL-2B-Instruct",
            "device": "cuda",
            "dtype": "bfloat16",
            "max_batch_size": 4,
            "quantization": "int4",
            "inference_framework": "transformers",
            "internvl_model_path": "/home/longcoding/dev/models/InternVL3-1B",
        },
        "extraction": {
            "min_confidence": 0.85,
            "enable_cross_validation": False,
            "max_retries": 2,
            "timeout_seconds": 30,
            "extraction_method": "vl",
            "field_methods": {},
        },
        "browser": {
            "type": "chromium",
            "headless": True,
            "viewport_width": 1920,
            "viewport_height": 1080,
            "full_page": True,
            "wait_until": "networkidle",
            "wait_timeout": 30000,
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 1,
            "max_concurrent_requests": 10,
        },
        "ocr": {
            "enabled": True,
            "min_confidence": 0.75,
            "use_easyocr": True,
            "fallback_to_vl": True,
            "ocr_timeout_seconds": 10,
        },
        "batch": {
            "batch_size": 4,
            "num_workers": 4,
            "prefetch_factor": 2,
        },
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)

    logger.info("default_config_created", path=output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Config loader utilities")
    parser.add_argument("--create-default", action="store_true", help="Create default config.yaml")
    parser.add_argument("--output", default="config.yaml", help="Output path for default config")
    parser.add_argument("--show", action="store_true", help="Show current configuration")

    args = parser.parse_args()

    if args.create_default:
        create_default_config(args.output)
        print(f"Default configuration written to: {args.output}")

    if args.show:
        loader = ConfigLoader()
        config = loader.get_config_dict()
        print(yaml.dump(config, default_flow_style=False))
