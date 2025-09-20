#!/bin/bash

echo "Installing test dependencies..."
pip install pytest pytest-asyncio pytest-mock joblib pydantic-settings

echo "Setting environment variables..."
export API_FOOTBALL_KEY=test_key
export TELEGRAM_BOT_TOKEN=test_token
export TELEGRAM_CHAT_ID=test_chat_id
export OPENROUTER_API_KEY=test_openrouter_key
export REDIS_URL=redis://localhost:6379
export LOG_LEVEL=ERROR

echo "Running tests..."
python -m pytest tests/ -v --tb=short --asyncio-mode=auto --disable-warnings