#!/usr/bin/env python3
"""Production server starter for Football Match Predictor."""

import os
import sys
from pathlib import Path

# Add app directory to Python path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))

# Set production environment variables if not set
if not os.getenv("LOG_LEVEL"):
    os.environ["LOG_LEVEL"] = "INFO"

if not os.getenv("CACHE_TTL"):
    os.environ["CACHE_TTL"] = "3600"  # 1 hour for prod

# Import and run
from app.main import run_prod_server

if __name__ == "__main__":
    print("🏈 Starting Football Match Predictor - Production Server")
    print("📊 Features: ML Predictions + AI Analysis + Telegram Bot")
    print("🔗 Health Check: http://localhost:8000/health")
    print("📈 Metrics: http://localhost:8000/metrics")
    print("=" * 60)
    
    run_prod_server()