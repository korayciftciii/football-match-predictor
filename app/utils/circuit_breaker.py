"""Circuit breaker pattern implementation for external service calls."""

import asyncio
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass
import functools

from app.utils.logger import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, calls are blocked
    HALF_OPEN = "half_open"  # Testing if service is back


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Number of failures to open circuit
    recovery_timeout: int = 60          # Seconds to wait before trying again
    expected_exception: type = Exception # Exception type that triggers circuit
    success_threshold: int = 3          # Successes needed to close circuit in half-open state
    timeout: int = 30                   # Request timeout in seconds


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""
    state: CircuitState
    failure_count: int
    success_count: int
    last_failure_time: Optional[datetime]
    last_success_time: Optional[datetime]
    total_requests: int
    total_failures: int
    total_successes: int
    opened_at: Optional[datetime]
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.total_failures / self.total_requests) * 100
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 100.0
        return (self.total_successes / self.total_requests) * 100


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    
    def __init__(self, message: str, stats: CircuitBreakerStats):
        super().__init__(message)
        self.stats = stats


class CircuitBreaker:
    """Circuit breaker implementation for protecting external service calls."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State management
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        self.opened_at: Optional[datetime] = None
        
        # Statistics
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0
        
        # Thread safety
        self._lock = asyncio.Lock()
        
        logger.info(f"Circuit breaker '{name}' initialized with config: {self.config}")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function call through the circuit breaker.
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
            Original exception: If function fails and circuit allows it
        """
        async with self._lock:
            self.total_requests += 1
            
            # Check if circuit should be opened
            if self.state == CircuitState.CLOSED and self._should_open():
                await self._open_circuit()
            
            # Check if circuit should transition to half-open
            elif self.state == CircuitState.OPEN and self._should_attempt_reset():
                await self._half_open_circuit()
            
            # Block calls if circuit is open
            if self.state == CircuitState.OPEN:
                stats = self.get_stats()
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Failure rate: {stats.failure_rate:.1f}%. "
                    f"Next retry in {self._time_until_retry():.0f}s",
                    stats
                )
        
        # Execute the function call
        try:
            # Add timeout if specified
            if self.config.timeout > 0:
                result = await asyncio.wait_for(
                    func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs),
                    timeout=self.config.timeout
                )
            else:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            # Record success
            await self._record_success()
            return result
            
        except self.config.expected_exception as e:
            # Record failure
            await self._record_failure()
            raise e
        except asyncio.TimeoutError as e:
            # Timeout is also considered a failure
            await self._record_failure()
            raise e
    
    async def _record_success(self):
        """Record a successful call."""
        async with self._lock:
            self.success_count += 1
            self.total_successes += 1
            self.last_success_time = datetime.now()
            
            # Reset failure count on success
            self.failure_count = 0
            
            # Close circuit if enough successes in half-open state
            if (self.state == CircuitState.HALF_OPEN and 
                self.success_count >= self.config.success_threshold):
                await self._close_circuit()
            
            logger.debug(f"Circuit breaker '{self.name}' recorded success. State: {self.state}")
    
    async def _record_failure(self):
        """Record a failed call."""
        async with self._lock:
            self.failure_count += 1
            self.total_failures += 1
            self.last_failure_time = datetime.now()
            
            # Reset success count on failure
            self.success_count = 0
            
            logger.warning(f"Circuit breaker '{self.name}' recorded failure. "
                         f"Failure count: {self.failure_count}/{self.config.failure_threshold}")
    
    def _should_open(self) -> bool:
        """Check if circuit should be opened."""
        return self.failure_count >= self.config.failure_threshold
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt to reset (transition to half-open)."""
        if not self.last_failure_time:
            return False
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout
    
    def _time_until_retry(self) -> float:
        """Calculate seconds until next retry attempt."""
        if not self.last_failure_time:
            return 0.0
        
        time_since_failure = datetime.now() - self.last_failure_time
        return max(0.0, self.config.recovery_timeout - time_since_failure.total_seconds())
    
    async def _open_circuit(self):
        """Open the circuit."""
        self.state = CircuitState.OPEN
        self.opened_at = datetime.now()
        logger.warning(f"Circuit breaker '{self.name}' OPENED due to {self.failure_count} failures")
    
    async def _half_open_circuit(self):
        """Transition circuit to half-open state."""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0  # Reset success count for half-open testing
        logger.info(f"Circuit breaker '{self.name}' transitioned to HALF-OPEN")
    
    async def _close_circuit(self):
        """Close the circuit."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.opened_at = None
        logger.info(f"Circuit breaker '{self.name}' CLOSED after successful recovery")
    
    def get_stats(self) -> CircuitBreakerStats:
        """Get current circuit breaker statistics."""
        return CircuitBreakerStats(
            state=self.state,
            failure_count=self.failure_count,
            success_count=self.success_count,
            last_failure_time=self.last_failure_time,
            last_success_time=self.last_success_time,
            total_requests=self.total_requests,
            total_failures=self.total_failures,
            total_successes=self.total_successes,
            opened_at=self.opened_at
        )
    
    async def reset(self):
        """Manually reset the circuit breaker."""
        async with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.opened_at = None
            logger.info(f"Circuit breaker '{self.name}' manually reset")
    
    def is_healthy(self) -> bool:
        """Check if circuit breaker is in a healthy state."""
        return self.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN]


# Global circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get or create a circuit breaker by name."""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]


def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator to wrap functions with circuit breaker protection."""
    def decorator(func: Callable):
        cb = get_circuit_breaker(name, config)
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await cb.call(func, *args, **kwargs)
        
        return wrapper
    return decorator


async def get_all_circuit_breaker_stats() -> Dict[str, CircuitBreakerStats]:
    """Get statistics for all circuit breakers."""
    return {name: cb.get_stats() for name, cb in _circuit_breakers.items()}


async def reset_all_circuit_breakers():
    """Reset all circuit breakers."""
    for cb in _circuit_breakers.values():
        await cb.reset()
    logger.info("All circuit breakers reset")


# Predefined circuit breakers for common services
API_FOOTBALL_CIRCUIT = CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=30,
    expected_exception=Exception,
    success_threshold=2,
    timeout=15
)

OPENROUTER_CIRCUIT = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60,
    expected_exception=Exception,
    success_threshold=3,
    timeout=30
)

REDIS_CIRCUIT = CircuitBreakerConfig(
    failure_threshold=2,
    recovery_timeout=10,
    expected_exception=Exception,
    success_threshold=1,
    timeout=5
)