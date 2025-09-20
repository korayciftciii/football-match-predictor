"""FastAPI main application entry point for Football Match Predictor."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn

from app.config import settings
from app.routes.matches import router as matches_router
from app.services.fetch_data import APIFootballError
from app.services.analyzer import AnalysisError
from app.bots.telegram_bot import bot_instance
from app.utils.logger import setup_logging, get_logger, set_correlation_id
from app.utils.cache import cache_manager
from app.utils.monitoring import metrics_collector
from app.services.workflow import workflow
import uuid

# Setup logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("Starting Football Match Predictor application")
    
    try:
        # Initialize cache connection
        cache_connected = await cache_manager.connect()
        if cache_connected:
            logger.info("Cache connection established")
        else:
            logger.warning("Cache connection failed, continuing without cache")
        
        # Initialize Telegram bot
        try:
            await bot_instance.initialize()
            logger.info("Telegram bot initialized")
            
            # Start bot polling in background
            asyncio.create_task(bot_instance.start_polling())
            logger.info("Telegram bot polling started")
            
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            logger.warning("Continuing without Telegram bot")
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Football Match Predictor application")
    
    try:
        # Stop Telegram bot
        if bot_instance.is_running:
            await bot_instance.stop_polling()
            logger.info("Telegram bot stopped")
        
        # Disconnect cache
        await cache_manager.disconnect()
        logger.info("Cache disconnected")
        
        logger.info("Application shutdown completed")
        
    except Exception as e:
        logger.error(f"Error during application shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title="Football Match Predictor API",
    description="""
    🏈 **Football Match Predictor API**
    
    Bu API futbol maçları için ML tabanlı tahminler sağlar:
    
    * 📊 **Skor tahminleri** - Makine öğrenmesi ile skor tahmini
    * ⚽ **Gol atacak oyuncu** - Oyuncu bazlı gol tahminleri  
    * 🟨 **Sarı kart sayısı** - İstatistiksel kart tahminleri
    * ⛳ **Korner sayısı** - Korner tahminleri
    * 🕐 **İlk yarı sonucu** - İlk yarı tahminleri
    * 🤖 **AI Analizi** - Türkçe AI destekli açıklamalar
    * 📱 **Telegram Bot** - Bot arayüzü ile kolay erişim
    
    **Özellikler:**
    - Gerçek zamanlı API-Football entegrasyonu
    - Redis cache ile performans optimizasyonu
    - OpenRouter AI ile Türkçe açıklamalar
    - Comprehensive error handling
    - Request correlation tracking
    """,
    version="1.0.0",
    contact={
        "name": "Football Match Predictor",
        "url": "https://github.com/your-repo/football-match-predictor",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)


# Request correlation ID middleware
@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """Add correlation ID to all requests for tracking."""
    correlation_id = str(uuid.uuid4())[:8]
    set_correlation_id(correlation_id)
    
    # Add to response headers
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    
    return response


# Global exception handlers
@app.exception_handler(APIFootballError)
async def api_football_error_handler(request: Request, exc: APIFootballError):
    """Handle API Football specific errors."""
    logger.error(f"API Football error: {exc}")
    return JSONResponse(
        status_code=503,
        content={
            "detail": f"External API error: {str(exc)}",
            "type": "api_football_error",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )


@app.exception_handler(AnalysisError)
async def analysis_error_handler(request: Request, exc: AnalysisError):
    """Handle analysis specific errors."""
    logger.error(f"Analysis error: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": f"Analysis failed: {str(exc)}",
            "type": "analysis_error",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request validation failed",
            "errors": exc.errors(),
            "type": "validation_error",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "type": "http_error",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "type": "internal_error",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )


# Include routers
app.include_router(matches_router)


# Root endpoint
@app.get("/", tags=["root"])
async def root() -> Dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "message": "🏈 Football Match Predictor API",
        "version": "1.0.0",
        "description": "ML-powered football match predictions with Turkish AI analysis",
        "features": [
            "Score predictions",
            "Goal scorer predictions", 
            "Yellow card predictions",
            "Corner predictions",
            "First half predictions",
            "AI-powered Turkish summaries",
            "Telegram bot interface"
        ],
        "endpoints": {
            "matches": "/matches/today",
            "predictions": "/matches/{match_id}/predictions",
            "analysis": "/matches/{match_id}/analyze",
            "health": "/matches/health",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "timestamp": datetime.now().isoformat(),
        "status": "operational"
    }


# Health check endpoint
@app.get("/health", tags=["monitoring"])
async def health_check() -> Dict[str, Any]:
    """Application health check endpoint."""
    try:
        # Check cache health
        cache_healthy = await cache_manager.is_healthy()
        
        # Check bot health
        bot_health = await bot_instance.health_check()
        
        # Check workflow health
        workflow_health = await workflow.get_workflow_health()
        
        # Overall status
        services_healthy = (
            cache_healthy and 
            bot_health.get("status") == "healthy" and
            workflow_health.get("status") in ["healthy", "degraded"]
        )
        overall_status = "healthy" if services_healthy else "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "services": {
                "cache": "healthy" if cache_healthy else "unhealthy",
                "telegram_bot": bot_health.get("status", "unknown"),
                "workflow": workflow_health.get("status", "unknown"),
                "api": "healthy"
            },
            "uptime": "running",
            "environment": "production" if settings.log_level == "INFO" else "development"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


# Comprehensive workflow health check endpoint
@app.get("/health/workflow", tags=["monitoring"])
async def workflow_health_check() -> Dict[str, Any]:
    """Comprehensive workflow health check endpoint."""
    return await workflow.run_health_check()


# Comprehensive system health check endpoint
@app.get("/health/comprehensive", tags=["monitoring"])
async def comprehensive_health_check() -> Dict[str, Any]:
    """Run comprehensive health checks on all system components."""
    from app.utils.health_checks import health_manager
    
    return await health_manager.get_overall_health()


# Circuit breaker status endpoint
@app.get("/health/circuit-breakers", tags=["monitoring"])
async def circuit_breaker_status() -> Dict[str, Any]:
    """Get status of all circuit breakers."""
    from app.utils.circuit_breaker import get_all_circuit_breaker_stats
    
    stats = await get_all_circuit_breaker_stats()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "circuit_breakers": {
            name: {
                "state": cb_stats.state.value,
                "failure_count": cb_stats.failure_count,
                "success_count": cb_stats.success_count,
                "total_requests": cb_stats.total_requests,
                "failure_rate": cb_stats.failure_rate,
                "success_rate": cb_stats.success_rate,
                "last_failure": cb_stats.last_failure_time.isoformat() if cb_stats.last_failure_time else None,
                "last_success": cb_stats.last_success_time.isoformat() if cb_stats.last_success_time else None
            }
            for name, cb_stats in stats.items()
        }
    }


# Reset circuit breakers endpoint (for admin use)
@app.post("/admin/circuit-breakers/reset", tags=["admin"])
async def reset_circuit_breakers() -> Dict[str, Any]:
    """Reset all circuit breakers (admin endpoint)."""
    from app.utils.circuit_breaker import reset_all_circuit_breakers
    
    await reset_all_circuit_breakers()
    
    return {
        "message": "All circuit breakers have been reset",
        "timestamp": datetime.now().isoformat()
    }


# Metrics endpoint
@app.get("/metrics", tags=["monitoring"])
async def metrics() -> Dict[str, Any]:
    """Application metrics endpoint."""
    try:
        # Get cache stats
        cache_stats = await cache_manager.get_stats()
        
        # Get bot stats
        bot_health = await bot_instance.health_check()
        
        # Get workflow stats
        workflow_health = await workflow.get_workflow_health()
        
        # Get monitoring metrics
        monitoring_metrics = metrics_collector.get_metrics_summary()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "application": {
                "name": "Football Match Predictor",
                "version": "1.0.0",
                "status": "running"
            },
            "cache": cache_stats,
            "telegram_bot": {
                "status": bot_health.get("status", "unknown"),
                "active_sessions": bot_health.get("active_sessions", 0)
            },
            "workflow": workflow_health,
            "monitoring": monitoring_metrics,
            "system": {
                "log_level": settings.log_level,
                "cache_ttl": settings.cache_ttl,
                "prediction_cache_hours": settings.prediction_cache_hours
            }
        }
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "status": "metrics_unavailable"
        }


# OpenRouter API test endpoint
@app.get("/test/openrouter", tags=["testing"])
async def test_openrouter() -> Dict[str, Any]:
    """Test OpenRouter API connection."""
    try:
        from app.bots.telegram_bot import test_openrouter_api
        
        # Test direct API call
        direct_test = await test_openrouter_api()
        
        # Test through AI summary service
        ai_test = await bot_instance.analyzer.ai_summary.test_connection()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "direct_api_test": direct_test,
            "ai_service_test": ai_test,
            "model_used": "openai/gpt-oss-120b:free",
            "base_url": settings.openrouter_base_url
        }
        
    except Exception as e:
        logger.error(f"OpenRouter test failed: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "error": str(e)
        }


# API-Football debug endpoint
@app.get("/test/api-football", tags=["testing"])
async def test_api_football() -> Dict[str, Any]:
    """Test API-Football connection and data."""
    try:
        from app.services.fetch_data import FootballDataFetcher
        from datetime import date
        
        fetcher = FootballDataFetcher()
        
        # Test API status
        status_result = await fetcher.health_check()
        
        # Test today's date
        today = date.today().strftime("%Y-%m-%d")
        
        # Test direct API call for fixtures
        try:
            # Test with a specific league (Premier League = 39)
            test_data = await fetcher._make_request("fixtures", {
                "date": today,
                "league": 39
            })
            
            fixtures_count = len(test_data.get("response", []))
            
            # Also test without league filter to see all matches for today
            all_today_data = await fetcher._make_request("fixtures", {"date": today})
            all_today_count = len(all_today_data.get("response", []))
            
        except Exception as e:
            test_data = {"error": str(e)}
            fixtures_count = 0
            all_today_count = 0
        
        # Test all leagues
        try:
            all_matches = await fetcher.get_todays_matches()
            total_matches = len(all_matches)
        except Exception as e:
            total_matches = 0
            all_matches = {"error": str(e)}
        
        return {
            "timestamp": datetime.now().isoformat(),
            "today_date": today,
            "api_status": status_result,
            "premier_league_fixtures": {
                "count": fixtures_count,
                "data": test_data
            },
            "all_today_fixtures": {
                "count": all_today_count,
                "note": "All matches for today across all leagues"
            },
            "all_matches": {
                "count": total_matches,
                "sample": all_matches[:2] if isinstance(all_matches, list) and all_matches else all_matches
            },
            "api_config": {
                "base_url": settings.api_football_base_url,
                "has_key": bool(settings.api_football_key),
                "key_length": len(settings.api_football_key) if settings.api_football_key else 0
            }
        }
        
    except Exception as e:
        logger.error(f"API-Football test failed: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "error": str(e)
        }


# Development server runner
def run_dev_server():
    """Run development server with hot reload."""
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower(),
        access_log=True
    )


# Production server runner
def run_prod_server():
    """Run production server."""
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker for now due to shared state
        log_level=settings.log_level.lower(),
        access_log=True
    )


if __name__ == "__main__":
    # Run development server by default
    run_dev_server()