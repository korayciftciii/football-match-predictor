"""Redis cache utility for Football Match Predictor."""

import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Any, Optional, Union, Callable, Dict
from functools import wraps
import redis.asyncio as redis
from redis.exceptions import RedisError, ConnectionError

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class CacheError(Exception):
    """Custom exception for cache-related errors."""
    pass


class CacheManager:
    """Redis cache manager with fallback mechanisms."""
    
    def __init__(self):
        self.redis_url = settings.redis_url
        self.default_ttl = settings.cache_ttl
        self._redis_client: Optional[redis.Redis] = None
        self._is_connected = False
        
    async def connect(self) -> bool:
        """
        Connect to Redis server.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self._redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,  # We'll handle encoding ourselves
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self._redis_client.ping()
            self._is_connected = True
            logger.info("Successfully connected to Redis")
            return True
            
        except (ConnectionError, RedisError) as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self._is_connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to Redis: {e}")
            self._is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Redis server."""
        if self._redis_client:
            try:
                await self._redis_client.close()
                logger.info("Disconnected from Redis")
            except Exception as e:
                logger.warning(f"Error disconnecting from Redis: {e}")
            finally:
                self._redis_client = None
                self._is_connected = False
    
    async def is_healthy(self) -> bool:
        """Check if Redis connection is healthy."""
        if not self._is_connected or not self._redis_client:
            return False
        
        try:
            await self._redis_client.ping()
            return True
        except Exception:
            self._is_connected = False
            return False
    
    def _serialize_key(self, key: str) -> str:
        """Create a consistent cache key."""
        # Add prefix for namespacing
        prefixed_key = f"football_predictor:{key}"
        
        # Hash long keys to avoid Redis key length limits
        if len(prefixed_key) > 250:
            hash_key = hashlib.md5(prefixed_key.encode()).hexdigest()
            return f"football_predictor:hash:{hash_key}"
        
        return prefixed_key
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for Redis storage."""
        try:
            # Only use JSON for basic types, use pickle for everything else
            if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
                # Check if it's a simple dict/list without complex objects
                if isinstance(value, (dict, list)):
                    # Try JSON, if it fails, use pickle
                    try:
                        return json.dumps(value).encode('utf-8')
                    except (TypeError, ValueError):
                        return pickle.dumps(value)
                else:
                    return json.dumps(value).encode('utf-8')
            else:
                # Use pickle for complex objects (like Pydantic models)
                return pickle.dumps(value)
        except Exception as e:
            logger.warning(f"Failed to serialize value: {e}")
            # Fallback to pickle
            return pickle.dumps(value)
    
    def _deserialize_value(self, value: bytes) -> Any:
        """Deserialize value from Redis storage."""
        try:
            # Try JSON first
            return json.loads(value.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            try:
                # Fallback to pickle
                return pickle.loads(value)
            except Exception as e:
                logger.error(f"Failed to deserialize value: {e}")
                raise CacheError(f"Failed to deserialize cached value: {e}")
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache with circuit breaker protection.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/error
        """
        if not await self.is_healthy():
            logger.debug(f"Cache unavailable, skipping get for key: {key}")
            return None
        
        try:
            # Import here to avoid circular imports
            from app.utils.circuit_breaker import get_circuit_breaker, REDIS_CIRCUIT
            
            cache_breaker = get_circuit_breaker("redis_cache", REDIS_CIRCUIT)
            
            async def _get_operation():
                cache_key = self._serialize_key(key)
                value = await self._redis_client.get(cache_key)
                
                if value is None:
                    logger.debug(f"Cache miss for key: {key}")
                    return None
                
                logger.debug(f"Cache hit for key: {key}")
                return self._deserialize_value(value)
            
            return await cache_breaker.call(_get_operation)
            
        except Exception as e:
            logger.warning(f"Error getting from cache (key: {key}): {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default: settings.cache_ttl)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not await self.is_healthy():
            logger.debug(f"Cache unavailable, skipping set for key: {key}")
            return False
        
        try:
            cache_key = self._serialize_key(key)
            serialized_value = self._serialize_value(value)
            ttl = ttl or self.default_ttl
            
            await self._redis_client.setex(cache_key, ttl, serialized_value)
            logger.debug(f"Cached value for key: {key} (TTL: {ttl}s)")
            return True
            
        except Exception as e:
            logger.warning(f"Error setting cache (key: {key}): {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not await self.is_healthy():
            return False
        
        try:
            cache_key = self._serialize_key(key)
            result = await self._redis_client.delete(cache_key)
            logger.debug(f"Deleted cache key: {key}")
            return result > 0
            
        except Exception as e:
            logger.warning(f"Error deleting from cache (key: {key}): {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if key exists, False otherwise
        """
        if not await self.is_healthy():
            return False
        
        try:
            cache_key = self._serialize_key(key)
            result = await self._redis_client.exists(cache_key)
            return result > 0
            
        except Exception as e:
            logger.warning(f"Error checking cache existence (key: {key}): {e}")
            return False
    
    async def get_ttl(self, key: str) -> Optional[int]:
        """
        Get remaining TTL for a key.
        
        Args:
            key: Cache key
            
        Returns:
            int: Remaining TTL in seconds, None if key doesn't exist or error
        """
        if not await self.is_healthy():
            return None
        
        try:
            cache_key = self._serialize_key(key)
            ttl = await self._redis_client.ttl(cache_key)
            return ttl if ttl > 0 else None
            
        except Exception as e:
            logger.warning(f"Error getting TTL (key: {key}): {e}")
            return None
    
    async def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching a pattern.
        
        Args:
            pattern: Key pattern (e.g., "team_stats:*")
            
        Returns:
            int: Number of keys deleted
        """
        if not await self.is_healthy():
            return 0
        
        try:
            cache_pattern = self._serialize_key(pattern)
            keys = await self._redis_client.keys(cache_pattern)
            
            if keys:
                deleted = await self._redis_client.delete(*keys)
                logger.info(f"Cleared {deleted} cache keys matching pattern: {pattern}")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.warning(f"Error clearing cache pattern ({pattern}): {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            dict: Cache statistics
        """
        if not await self.is_healthy():
            return {"status": "disconnected"}
        
        try:
            info = await self._redis_client.info()
            return {
                "status": "connected",
                "used_memory": info.get("used_memory_human", "N/A"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0),
                    info.get("keyspace_misses", 0)
                )
            }
            
        except Exception as e:
            logger.warning(f"Error getting cache stats: {e}")
            return {"status": "error", "error": str(e)}
    
    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate percentage."""
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0


# Global cache manager instance
cache_manager = CacheManager()


def cache_key_builder(*args, **kwargs) -> str:
    """Build a cache key from function arguments."""
    key_parts = []
    
    # Add positional arguments
    for arg in args:
        if hasattr(arg, '__dict__'):
            # For objects, use class name and id
            key_parts.append(f"{arg.__class__.__name__}_{id(arg)}")
        else:
            key_parts.append(str(arg))
    
    # Add keyword arguments
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}_{v}")
    
    return ":".join(key_parts)


def cache_key_builder(*args, **kwargs) -> str:
    """Build a cache key from function arguments."""
    try:
        # Convert args and kwargs to a consistent string
        key_parts = []
        
        # Add positional args (skip 'self' if present)
        for i, arg in enumerate(args):
            if i == 0 and hasattr(arg, '__class__'):
                # Skip 'self' parameter
                continue
            key_parts.append(str(arg))
        
        # Add keyword args
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        
        return ":".join(key_parts) if key_parts else "no_args"
        
    except Exception as e:
        logger.warning(f"Error building cache key: {e}")
        return "error_key"


def cached(ttl: Optional[int] = None, key_prefix: str = ""):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Build cache key
            func_name = f"{func.__module__}.{func.__name__}"
            arg_key = cache_key_builder(*args, **kwargs)
            cache_key = f"{key_prefix}:{func_name}:{arg_key}" if key_prefix else f"{func_name}:{arg_key}"
            
            # Try to get from cache
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for function: {func_name}")
                return cached_result
            
            # Execute function
            logger.debug(f"Cache miss for function: {func_name}, executing...")
            result = await func(*args, **kwargs)
            
            # Cache the result
            if result is not None:
                await cache_manager.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


def invalidate_cache_pattern(pattern: str):
    """
    Decorator to invalidate cache patterns after function execution.
    
    Args:
        pattern: Cache key pattern to invalidate
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Invalidate cache pattern
            deleted = await cache_manager.clear_pattern(pattern)
            if deleted > 0:
                logger.info(f"Invalidated {deleted} cache entries for pattern: {pattern}")
            
            return result
        
        return wrapper
    return decorator


# Convenience functions for common cache operations

async def cache_team_stats(team_id: int, stats: Any, ttl: Optional[int] = None) -> bool:
    """Cache team statistics."""
    key = f"team_stats:{team_id}"
    return await cache_manager.set(key, stats, ttl)


async def get_cached_team_stats(team_id: int) -> Optional[Any]:
    """Get cached team statistics."""
    key = f"team_stats:{team_id}"
    return await cache_manager.get(key)


async def cache_match_predictions(match_id: int, predictions: Any, ttl: Optional[int] = None) -> bool:
    """Cache match predictions."""
    key = f"predictions:{match_id}"
    # Use longer TTL for predictions (6 hours by default)
    prediction_ttl = ttl or (settings.prediction_cache_hours * 3600)
    return await cache_manager.set(key, predictions, prediction_ttl)


async def get_cached_match_predictions(match_id: int) -> Optional[Any]:
    """Get cached match predictions."""
    key = f"predictions:{match_id}"
    return await cache_manager.get(key)


async def cache_todays_matches(matches: Any, ttl: Optional[int] = None) -> bool:
    """Cache today's matches."""
    today = datetime.now().strftime("%Y-%m-%d")
    key = f"matches:today:{today}"
    # Cache for 1 hour since match schedules can change
    matches_ttl = ttl or 3600
    return await cache_manager.set(key, matches, matches_ttl)


async def get_cached_todays_matches() -> Optional[Any]:
    """Get cached today's matches."""
    today = datetime.now().strftime("%Y-%m-%d")
    key = f"matches:today:{today}"
    return await cache_manager.get(key)


# Global cache manager instance
cache_manager = CacheManager()