"""
Configuration management for Aerovision-V1-Server.
"""

import os
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Application
    app_name: str = "Aerovision-V1-Server"
    version: str = "2.0.0"
    environment: Literal["development", "test", "production"] = "development"
    debug: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False  # Set to True only for development

    # CORS
    cors_origins: list[str] = ["*"]
    cors_methods: list[str] = ["GET", "POST"]
    cors_headers: list[str] = ["*"]

    # Inference
    model_dir: str = "models"
    device: str = "cuda"
    preload_models: bool = True

    # OCR
    ocr_mode: str = "auto"
    ocr_lang: str = "ch"
    use_angle_cls: bool = True
    qwen_model: str = "qwen3-vl-flash"
    ocr_timeout: int = 30

    # Quality thresholds
    quality_pass_threshold: float = 0.6
    sharpness_weight: float = 0.3
    exposure_weight: float = 0.2
    composition_weight: float = 0.15
    noise_weight: float = 0.2
    color_weight: float = 0.15

    # New class detection
    new_class_similarity_threshold: float = 0.7

    # Limits
    max_image_size_mb: int = 20
    max_batch_size: int = 50
    request_timeout_seconds: int = 300

    # Redis (for shared statistics across workers)
    redis_url: str = "redis://localhost:6379/0"
    redis_enabled: bool = True

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # json or text


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Force reload settings from environment."""
    global _settings
    _settings = Settings()
    return _settings
