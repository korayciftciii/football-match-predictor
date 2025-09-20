"""Graceful degradation utilities for handling service failures."""

import asyncio
from typing import Any, Callable, Optional, Dict, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from app.utils.logger import get_logger
from app.utils.circuit_breaker import CircuitBreakerError

logger = get_logger(__name__)


class DegradationLevel(Enum):
    """Levels of service degradation."""
    FULL_SERVICE = "full_service"
    REDUCED_FEATURES = "reduced_features"
    BASIC_SERVICE = "basic_service"
    EMERGENCY_MODE = "emergency_mode"


@dataclass
class DegradationConfig:
    """Configuration for graceful degradation."""
    max_retries: int = 3
    retry_delay: float = 1.0
    fallback_enabled: bool = True
    cache_fallback: bool = True
    emergency_mode_threshold: int = 5  # Number of consecutive failures


class ServiceDegradationManager:
    """Manages graceful degradation across services."""
    
    def __init__(self):
        self.degradation_level = DegradationLevel.FULL_SERVICE
        self.service_failures: Dict[str, List[datetime]] = {}
        self.last_degradation_change = datetime.now()
        self.degradation_reasons: List[str] = []
    
    def record_service_failure(self, service_name: str, error: str):
        """Record a service failure for degradation tracking."""
        now = datetime.now()
        
        if service_name not in self.service_failures:
            self.service_failures[service_name] = []
        
        # Add failure timestamp
        self.service_failures[service_name].append(now)
        
        # Clean old failures (older than 5 minutes)
        cutoff = now - timedelta(minutes=5)
        self.service_failures[service_name] = [
            failure_time for failure_time in self.service_failures[service_name]
            if failure_time > cutoff
        ]
        
        # Check if we need to degrade service
        self._evaluate_degradation_level()
        
        logger.warning(f"Service failure recorded for {service_name}: {error}")
    
    def record_service_recovery(self, service_name: str):
        """Record a service recovery."""
        if service_name in self.service_failures:
            # Clear recent failures for this service
            self.service_failures[service_name] = []
        
        # Re-evaluate degradation level
        self._evaluate_degradation_level()
        
        logger.info(f"Service recovery recorded for {service_name}")
    
    def _evaluate_degradation_level(self):
        """Evaluate and update the current degradation level."""
        now = datetime.now()
        total_recent_failures = sum(len(failures) for failures in self.service_failures.values())
        
        # Count critical service failures
        critical_services = ["api_football", "redis_cache", "workflow_pipeline"]
        critical_failures = sum(
            len(self.service_failures.get(service, []))
            for service in critical_services
        )
        
        old_level = self.degradation_level
        old_reasons = self.degradation_reasons.copy()
        
        # Determine new degradation level
        if critical_failures >= 5:
            self.degradation_level = DegradationLevel.EMERGENCY_MODE
            self.degradation_reasons = [f"Critical service failures: {critical_failures}"]
        elif total_recent_failures >= 10:
            self.degradation_level = DegradationLevel.BASIC_SERVICE
            self.degradation_reasons = [f"High failure rate: {total_recent_failures} failures"]
        elif critical_failures >= 2:
            self.degradation_level = DegradationLevel.REDUCED_FEATURES
            self.degradation_reasons = [f"Some critical services failing: {critical_failures}"]
        elif total_recent_failures == 0:
            self.degradation_level = DegradationLevel.FULL_SERVICE
            self.degradation_reasons = []
        
        # Log degradation level changes
        if old_level != self.degradation_level:
            self.last_degradation_change = now
            logger.warning(
                f"Service degradation level changed: {old_level.value} -> {self.degradation_level.value}. "
                f"Reasons: {', '.join(self.degradation_reasons)}"
            )
    
    def get_degradation_status(self) -> Dict[str, Any]:
        """Get current degradation status."""
        return {
            "degradation_level": self.degradation_level.value,
            "reasons": self.degradation_reasons,
            "last_change": self.last_degradation_change.isoformat(),
            "service_failures": {
                service: len(failures)
                for service, failures in self.service_failures.items()
            },
            "is_degraded": self.degradation_level != DegradationLevel.FULL_SERVICE
        }
    
    def should_use_fallback(self, service_name: str) -> bool:
        """Check if a service should use fallback mechanisms."""
        if self.degradation_level == DegradationLevel.EMERGENCY_MODE:
            return True
        
        service_failure_count = len(self.service_failures.get(service_name, []))
        return service_failure_count >= 2
    
    def get_allowed_features(self) -> List[str]:
        """Get list of features allowed at current degradation level."""
        if self.degradation_level == DegradationLevel.FULL_SERVICE:
            return [
                "match_predictions", "ai_summaries", "player_predictions",
                "card_predictions", "corner_predictions", "batch_processing",
                "real_time_data", "advanced_analytics"
            ]
        elif self.degradation_level == DegradationLevel.REDUCED_FEATURES:
            return [
                "match_predictions", "basic_summaries", "player_predictions",
                "card_predictions", "corner_predictions"
            ]
        elif self.degradation_level == DegradationLevel.BASIC_SERVICE:
            return [
                "match_predictions", "basic_summaries"
            ]
        else:  # EMERGENCY_MODE
            return [
                "basic_predictions"
            ]


# Global degradation manager
degradation_manager = ServiceDegradationManager()


async def with_graceful_degradation(
    primary_func: Callable,
    fallback_func: Optional[Callable] = None,
    service_name: str = "unknown",
    config: Optional[DegradationConfig] = None
) -> Any:
    """
    Execute a function with graceful degradation support.
    
    Args:
        primary_func: Primary function to execute
        fallback_func: Fallback function if primary fails
        service_name: Name of the service for tracking
        config: Degradation configuration
        
    Returns:
        Result from primary or fallback function
    """
    config = config or DegradationConfig()
    
    # Try primary function with retries
    last_exception = None
    
    for attempt in range(config.max_retries):
        try:
            if asyncio.iscoroutinefunction(primary_func):
                result = await primary_func()
            else:
                result = primary_func()
            
            # Record success
            degradation_manager.record_service_recovery(service_name)
            return result
            
        except CircuitBreakerError as e:
            # Circuit breaker is open, use fallback immediately
            logger.warning(f"Circuit breaker open for {service_name}, using fallback")
            degradation_manager.record_service_failure(service_name, str(e))
            break
            
        except Exception as e:
            last_exception = e
            degradation_manager.record_service_failure(service_name, str(e))
            
            if attempt < config.max_retries - 1:
                delay = config.retry_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Attempt {attempt + 1} failed for {service_name}, retrying in {delay}s: {e}")
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {config.max_retries} attempts failed for {service_name}: {e}")
    
    # Try fallback function if available
    if fallback_func and config.fallback_enabled:
        try:
            logger.info(f"Using fallback for {service_name}")
            
            if asyncio.iscoroutinefunction(fallback_func):
                return await fallback_func()
            else:
                return fallback_func()
                
        except Exception as e:
            logger.error(f"Fallback also failed for {service_name}: {e}")
            degradation_manager.record_service_failure(f"{service_name}_fallback", str(e))
    
    # If we get here, both primary and fallback failed
    if last_exception:
        raise last_exception
    else:
        raise Exception(f"Service {service_name} completely unavailable")


def graceful_degradation(
    service_name: str,
    fallback_func: Optional[Callable] = None,
    config: Optional[DegradationConfig] = None
):
    """Decorator for graceful degradation."""
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            primary_func = lambda: func(*args, **kwargs)
            return await with_graceful_degradation(
                primary_func, fallback_func, service_name, config
            )
        return wrapper
    return decorator


def is_feature_available(feature_name: str) -> bool:
    """Check if a feature is available at the current degradation level."""
    allowed_features = degradation_manager.get_allowed_features()
    return feature_name in allowed_features


def get_degradation_message() -> Optional[str]:
    """Get user-friendly message about current service degradation."""
    status = degradation_manager.get_degradation_status()
    
    if not status["is_degraded"]:
        return None
    
    level = status["degradation_level"]
    
    messages = {
        "reduced_features": "⚠️ Some advanced features may be temporarily unavailable due to service issues.",
        "basic_service": "⚠️ Running in basic mode. Some features are temporarily disabled.",
        "emergency_mode": "🚨 Emergency mode active. Only essential features are available."
    }
    
    return messages.get(level, "⚠️ Service is experiencing issues.")


# Convenience functions for common degradation patterns
async def with_cache_fallback(cache_key: str, fetch_func: Callable, cache_ttl: int = 3600) -> Any:
    """Execute function with cache fallback support."""
    from app.utils.cache import cache_manager
    
    async def primary():
        return await fetch_func()
    
    async def fallback():
        # Try to get stale data from cache
        cached_data = await cache_manager.get(f"stale_{cache_key}")
        if cached_data:
            logger.info(f"Using stale cache data for {cache_key}")
            return cached_data
        
        # If no cache, try a simplified version
        logger.warning(f"No cache available for {cache_key}, using minimal data")
        return None
    
    return await with_graceful_degradation(primary, fallback, "cache_fallback")