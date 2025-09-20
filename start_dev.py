#!/usr/bin/env python3
"""Development server starter for Football Match Predictor."""

import os
import sys
from pathlib import Path

# Add app directory to Python path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))

# Set development environment variables if not set
if not os.getenv("LOG_LEVEL"):
    os.environ["LOG_LEVEL"] = "DEBUG"

if not os.getenv("CACHE_TTL"):
    os.environ["CACHE_TTL"] = "1800"  # 30 minutes for dev

# Import and run
from app.main import run_dev_server

if __name__ == "__main__":
    print("🏈 Starting Football Match Predictor - Development Server")
    print("📊 Features: ML Predictions + AI Analysis + Telegram Bot")
    print("🔗 API Docs: http://localhost:8000/docs")
    print("🤖 Bot: Make sure to set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
    print("=" * 60)
    
    run_dev_server()