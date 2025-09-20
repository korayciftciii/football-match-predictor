"""Configuration management for Football Match Predictor."""

import os
from typing import Optional
from pydantic import field_validator, ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable validation."""
    
    # API Keys
    api_football_key: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    openrouter_api_key: str = ""
    huggingface_api_key: Optional[str] = None
    
    # Service Configuration
    redis_url: str = "redis://localhost:6379"
    log_level: str = "INFO"
    log_format: str = "text"
    cache_ttl: int = 3600
    
    # ML Configuration
    model_confidence_threshold: float = 0.7
    prediction_cache_hours: int = 6
    
    # API Configuration
    api_football_base_url: str = "https://v3.football.api-sports.io"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    
    # Environment
    environment: str = "development"
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        protected_namespaces=('settings_',)
    )
    
    @field_validator('api_football_key')
    @classmethod
    def validate_api_football_key(cls, v):
        # Skip validation in test environment
        if os.getenv('ENVIRONMENT') == 'test':
            return v
        if not v:
            raise ValueError('API_FOOTBALL_KEY is required')
        return v
    
    @field_validator('telegram_bot_token')
    @classmethod
    def validate_telegram_bot_token(cls, v):
        # Skip validation in test environment
        if os.getenv('ENVIRONMENT') == 'test':
            return v
        if not v:
            raise ValueError('TELEGRAM_BOT_TOKEN is required')
        return v
    
    @field_validator('telegram_chat_id')
    @classmethod
    def validate_telegram_chat_id(cls, v):
        # Skip validation in test environment
        if os.getenv('ENVIRONMENT') == 'test':
            return v
        if not v:
            raise ValueError('TELEGRAM_CHAT_ID is required')
        return v
    
    @field_validator('openrouter_api_key')
    @classmethod
    def validate_openrouter_api_key(cls, v):
        # Skip validation in test environment
        if os.getenv('ENVIRONMENT') == 'test':
            return v
        if not v:
            raise ValueError('OPENROUTER_API_KEY is required')
        return v


# Global settings instance - lazy initialization
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Get settings instance with lazy initialization."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

# For backward compatibility
settings = get_settings()