"""Unit tests for cache utility."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import json
import pickle
from datetime import datetime
import sys
import os

# Add the app directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.utils.cache import (
    CacheManager, CacheError, cached, cache_key_builder,
    cache_team_stats, get_cached_team_stats,
    cache_match_predictions, get_cached_match_predictions
)


class TestCacheManager:
    """Test cases for CacheManager."""
    
    @pytest.fixture
    def cache_manager(self):
        """Create a CacheManager instance for testing."""
        return CacheManager()
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_redis.get.return_value = None
        mock_redis.setex.return_value = True
        mock_redis.delete.return_value = 1
        mock_redis.exists.return_value = 1
        mock_redis.ttl.return_value = 3600
        mock_redis.keys.return_value = ["key1", "key2"]
        mock_redis.info.return_value = {
            "used_memory_human": "1.5M",
            "connected_clients": 5,
            "total_commands_processed": 1000,
            "keyspace_hits": 800,
            "keyspace_misses": 200
        }
        return mock_redis
    
    @pytest.mark.asyncio
    async def test_connect_success(self, cache_manager, mock_redis):
        """Test successful Redis connection."""
        with patch('redis.asyncio.from_url', return_value=mock_redis):
            result = await cache_manager.connect()
            
            assert result is True
            assert cache_manager._is_connected is True
            mock_redis.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_failure(self, cache_manager):
        """Test Redis connection failure."""
        with patch('redis.asyncio.from_url') as mock_from_url:
            mock_redis = AsyncMock()
            mock_redis.ping.side_effect = Exception("Connection failed")
            mock_from_url.return_value = mock_redis
            
            result = await cache_manager.connect()
            
            assert result is False
            assert cache_manager._is_connected is False
    
    @pytest.mark.asyncio
    async def test_is_healthy_true(self, cache_manager, mock_redis):
        """Test healthy Redis connection."""
        cache_manager._redis_client = mock_redis
        cache_manager._is_connected = True
        
        result = await cache_manager.is_healthy()
        
        assert result is True
        mock_redis.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_is_healthy_false(self, cache_manager, mock_redis):
        """Test unhealthy Redis connection."""
        cache_manager._redis_client = mock_redis
        cache_manager._is_connected = True
        mock_redis.ping.side_effect = Exception("Connection lost")
        
        result = await cache_manager.is_healthy()
        
        assert result is False
        assert cache_manager._is_connected is False
    
    def test_serialize_key(self, cache_manager):
        """Test cache key serialization."""
        key = "test_key"
        result = cache_manager._serialize_key(key)
        
        assert result == "football_predictor:test_key"
    
    def test_serialize_key_long(self, cache_manager):
        """Test long cache key serialization."""
        long_key = "a" * 300  # Longer than 250 chars
        result = cache_manager._serialize_key(long_key)
        
        assert result.startswith("football_predictor:hash:")
        assert len(result) < 100  # Should be much shorter
    
    def test_serialize_value_json(self, cache_manager):
        """Test value serialization with JSON."""
        test_data = {"key": "value", "number": 42}
        result = cache_manager._serialize_value(test_data)
        
        assert isinstance(result, bytes)
        # Should be JSON serializable
        deserialized = json.loads(result.decode('utf-8'))
        assert deserialized == test_data
    
    def test_serialize_value_pickle(self, cache_manager):
        """Test value serialization with pickle for complex objects."""
        test_data = datetime.now()  # Complex object
        result = cache_manager._serialize_value(test_data)
        
        assert isinstance(result, bytes)
        # Should be pickle serialized
        deserialized = pickle.loads(result)
        assert deserialized == test_data
    
    def test_deserialize_value_json(self, cache_manager):
        """Test value deserialization from JSON."""
        test_data = {"key": "value", "number": 42}
        serialized = json.dumps(test_data).encode('utf-8')
        
        result = cache_manager._deserialize_value(serialized)
        
        assert result == test_data
    
    def test_deserialize_value_pickle(self, cache_manager):
        """Test value deserialization from pickle."""
        test_data = datetime.now()
        serialized = pickle.dumps(test_data)
        
        result = cache_manager._deserialize_value(serialized)
        
        assert result == test_data
    
    @pytest.mark.asyncio
    async def test_get_success(self, cache_manager, mock_redis):
        """Test successful cache get."""
        cache_manager._redis_client = mock_redis
        cache_manager._is_connected = True
        
        test_data = {"key": "value"}
        mock_redis.get.return_value = json.dumps(test_data).encode('utf-8')
        
        result = await cache_manager.get("test_key")
        
        assert result == test_data
        mock_redis.get.assert_called_once_with("football_predictor:test_key")
    
    @pytest.mark.asyncio
    async def test_get_miss(self, cache_manager, mock_redis):
        """Test cache miss."""
        cache_manager._redis_client = mock_redis
        cache_manager._is_connected = True
        mock_redis.get.return_value = None
        
        result = await cache_manager.get("test_key")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_unhealthy(self, cache_manager):
        """Test get when cache is unhealthy."""
        cache_manager._is_connected = False
        
        result = await cache_manager.get("test_key")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_set_success(self, cache_manager, mock_redis):
        """Test successful cache set."""
        cache_manager._redis_client = mock_redis
        cache_manager._is_connected = True
        
        test_data = {"key": "value"}
        result = await cache_manager.set("test_key", test_data, 3600)
        
        assert result is True
        mock_redis.setex.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_set_unhealthy(self, cache_manager):
        """Test set when cache is unhealthy."""
        cache_manager._is_connected = False
        
        result = await cache_manager.set("test_key", {"data": "value"})
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_delete_success(self, cache_manager, mock_redis):
        """Test successful cache delete."""
        cache_manager._redis_client = mock_redis
        cache_manager._is_connected = True
        mock_redis.delete.return_value = 1
        
        result = await cache_manager.delete("test_key")
        
        assert result is True
        mock_redis.delete.assert_called_once_with("football_predictor:test_key")
    
    @pytest.mark.asyncio
    async def test_exists_true(self, cache_manager, mock_redis):
        """Test cache key exists."""
        cache_manager._redis_client = mock_redis
        cache_manager._is_connected = True
        mock_redis.exists.return_value = 1
        
        result = await cache_manager.exists("test_key")
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_get_ttl(self, cache_manager, mock_redis):
        """Test getting TTL for a key."""
        cache_manager._redis_client = mock_redis
        cache_manager._is_connected = True
        mock_redis.ttl.return_value = 1800
        
        result = await cache_manager.get_ttl("test_key")
        
        assert result == 1800
    
    @pytest.mark.asyncio
    async def test_clear_pattern(self, cache_manager, mock_redis):
        """Test clearing keys by pattern."""
        cache_manager._redis_client = mock_redis
        cache_manager._is_connected = True
        mock_redis.keys.return_value = ["key1", "key2", "key3"]
        mock_redis.delete.return_value = 3
        
        result = await cache_manager.clear_pattern("test:*")
        
        assert result == 3
        mock_redis.keys.assert_called_once()
        mock_redis.delete.assert_called_once_with("key1", "key2", "key3")
    
    @pytest.mark.asyncio
    async def test_get_stats(self, cache_manager, mock_redis):
        """Test getting cache statistics."""
        cache_manager._redis_client = mock_redis
        cache_manager._is_connected = True
        
        stats = await cache_manager.get_stats()
        
        assert stats["status"] == "connected"
        assert stats["used_memory"] == "1.5M"
        assert stats["hit_rate"] == 80.0  # 800/(800+200) * 100
    
    def test_calculate_hit_rate(self, cache_manager):
        """Test hit rate calculation."""
        hit_rate = cache_manager._calculate_hit_rate(800, 200)
        assert hit_rate == 80.0
        
        # Test zero division
        hit_rate = cache_manager._calculate_hit_rate(0, 0)
        assert hit_rate == 0.0


class TestCacheDecorators:
    """Test cache decorators and utility functions."""
    
    def test_cache_key_builder(self):
        """Test cache key building from arguments."""
        key = cache_key_builder("arg1", 42, keyword="value")
        
        assert "arg1" in key
        assert "42" in key
        assert "keyword=value" in key
    
    @pytest.mark.asyncio
    async def test_cached_decorator_hit(self):
        """Test cached decorator with cache hit."""
        mock_cache_manager = AsyncMock()
        mock_cache_manager.get.return_value = "cached_result"
        
        with patch('app.utils.cache.cache_manager', mock_cache_manager):
            @cached(ttl=3600)
            async def test_function(arg1, arg2="default"):
                return "function_result"
            
            result = await test_function("test", arg2="value")
            
            assert result == "cached_result"
            mock_cache_manager.get.assert_called_once()
            mock_cache_manager.set.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_cached_decorator_miss(self):
        """Test cached decorator with cache miss."""
        mock_cache_manager = AsyncMock()
        mock_cache_manager.get.return_value = None
        mock_cache_manager.set.return_value = True
        
        with patch('app.utils.cache.cache_manager', mock_cache_manager):
            @cached(ttl=3600)
            async def test_function(arg1, arg2="default"):
                return "function_result"
            
            result = await test_function("test", arg2="value")
            
            assert result == "function_result"
            mock_cache_manager.get.assert_called_once()
            mock_cache_manager.set.assert_called_once()


class TestConvenienceFunctions:
    """Test convenience cache functions."""
    
    @pytest.mark.asyncio
    async def test_cache_team_stats(self):
        """Test caching team statistics."""
        mock_cache_manager = AsyncMock()
        mock_cache_manager.set.return_value = True
        
        with patch('app.utils.cache.cache_manager', mock_cache_manager):
            result = await cache_team_stats(123, {"goals": 2.5}, 7200)
            
            assert result is True
            mock_cache_manager.set.assert_called_once_with("team_stats:123", {"goals": 2.5}, 7200)
    
    @pytest.mark.asyncio
    async def test_get_cached_team_stats(self):
        """Test getting cached team statistics."""
        mock_cache_manager = AsyncMock()
        mock_cache_manager.get.return_value = {"goals": 2.5}
        
        with patch('app.utils.cache.cache_manager', mock_cache_manager):
            result = await get_cached_team_stats(123)
            
            assert result == {"goals": 2.5}
            mock_cache_manager.get.assert_called_once_with("team_stats:123")
    
    @pytest.mark.asyncio
    async def test_cache_match_predictions(self):
        """Test caching match predictions."""
        mock_cache_manager = AsyncMock()
        mock_cache_manager.set.return_value = True
        
        with patch('app.utils.cache.cache_manager', mock_cache_manager):
            with patch('app.utils.cache.settings') as mock_settings:
                mock_settings.prediction_cache_hours = 6
                
                result = await cache_match_predictions(456, {"score": "2-1"})
                
                assert result is True
                # Should use 6 hours * 3600 seconds = 21600
                mock_cache_manager.set.assert_called_once_with("predictions:456", {"score": "2-1"}, 21600)
    
    @pytest.mark.asyncio
    async def test_get_cached_match_predictions(self):
        """Test getting cached match predictions."""
        mock_cache_manager = AsyncMock()
        mock_cache_manager.get.return_value = {"score": "2-1"}
        
        with patch('app.utils.cache.cache_manager', mock_cache_manager):
            result = await get_cached_match_predictions(456)
            
            assert result == {"score": "2-1"}
            mock_cache_manager.get.assert_called_once_with("predictions:456")