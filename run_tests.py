#!/usr/bin/env python3
"""Test runner script for Football Match Predictor."""

import subprocess
import sys
import os

def run_tests():
    """Run all tests with proper configuration."""
    
    # Set environment variables for testing
    os.environ["API_FOOTBALL_KEY"] = "test_key"
    os.environ["TELEGRAM_BOT_TOKEN"] = "test_token"
    os.environ["TELEGRAM_CHAT_ID"] = "test_chat_id"
    os.environ["OPENROUTER_API_KEY"] = "test_openrouter_key"
    os.environ["REDIS_URL"] = "redis://localhost:6379"
    os.environ["LOG_LEVEL"] = "ERROR"  # Reduce log noise during tests
    
    # Run pytest with configuration
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--asyncio-mode=auto",
        "--disable-warnings"
    ]
    
    print("Running tests...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)