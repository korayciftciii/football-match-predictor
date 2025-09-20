@echo off
echo Installing test dependencies...
pip install pytest pytest-asyncio pytest-mock joblib pydantic-settings

echo Setting environment variables...
set API_FOOTBALL_KEY=test_key
set TELEGRAM_BOT_TOKEN=test_token
set TELEGRAM_CHAT_ID=test_chat_id
set OPENROUTER_API_KEY=test_openrouter_key
set REDIS_URL=redis://localhost:6379
set LOG_LEVEL=ERROR

echo Running tests...
python -m pytest tests/ --tb=short --asyncio-mode=auto --disable-warnings -q

pause