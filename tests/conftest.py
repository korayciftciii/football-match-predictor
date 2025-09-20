"""Pytest configuration and fixtures for Football Match Predictor tests."""

import os
import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables for all tests."""
    # Set test environment
    os.environ['ENVIRONMENT'] = 'test'
    
    # Set dummy API keys for testing
    os.environ['API_FOOTBALL_KEY'] = 'test_api_football_key'
    os.environ['TELEGRAM_BOT_TOKEN'] = 'test_telegram_bot_token'
    os.environ['TELEGRAM_CHAT_ID'] = 'test_telegram_chat_id'
    os.environ['OPENROUTER_API_KEY'] = 'test_openrouter_api_key'
    os.environ['HUGGINGFACE_API_KEY'] = 'test_huggingface_api_key'
    
    # Set test-specific configurations
    os.environ['REDIS_URL'] = 'redis://localhost:6379/0'
    os.environ['LOG_LEVEL'] = 'DEBUG'
    os.environ['CACHE_TTL'] = '60'
    
    yield
    
    # Cleanup after test
    test_env_vars = [
        'ENVIRONMENT', 'API_FOOTBALL_KEY', 'TELEGRAM_BOT_TOKEN',
        'TELEGRAM_CHAT_ID', 'OPENROUTER_API_KEY', 'HUGGINGFACE_API_KEY',
        'REDIS_URL', 'LOG_LEVEL', 'CACHE_TTL'
    ]
    for var in test_env_vars:
        if var in os.environ:
            del os.environ[var]


@pytest.fixture
def mock_redis():
    """Mock Redis connection for tests."""
    with patch('redis.asyncio.Redis') as mock_redis:
        mock_redis.return_value.ping.return_value = True
        mock_redis.return_value.get.return_value = None
        mock_redis.return_value.set.return_value = True
        mock_redis.return_value.delete.return_value = True
        yield mock_redis


@pytest.fixture
def mock_httpx():
    """Mock HTTP client for tests."""
    with patch('httpx.AsyncClient') as mock_client:
        mock_response = mock_client.return_value.get.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": [],
            "results": 0
        }
        yield mock_client


@pytest.fixture
def mock_openai():
    """Mock OpenAI client for tests."""
    with patch('openai.AsyncOpenAI') as mock_openai:
        mock_response = mock_openai.return_value.chat.completions.create.return_value
        mock_response.choices = [type('obj', (object,), {
            'message': type('obj', (object,), {
                'content': 'Test AI response'
            })()
        })()]
        yield mock_openai


@pytest.fixture
def mock_telegram_bot():
    """Mock Telegram bot for tests."""
    with patch('telegram.Bot') as mock_bot:
        mock_bot.return_value.send_message.return_value = type('obj', (object,), {
            'message_id': 123,
            'text': 'Test message'
        })()
        yield mock_bot


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "asyncio: marks tests as async tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add asyncio marker to async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)
        
        # Add unit marker to tests in unit test files
        if "test_" in item.nodeid and "integration" not in item.nodeid:
            item.add_marker(pytest.mark.unit)


# Import asyncio for the collection modifier
import asyncio
