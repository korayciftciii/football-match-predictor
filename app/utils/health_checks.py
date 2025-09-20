"""Comprehensive health check system for all application components."""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from app.utils.logger import get_logger
from app.utils.circuit_breaker import get_all_circuit_breaker_stats
from app.utils.monitoring import metrics_collector

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class HealthCheck:
    """Base class for health checks."""
    
    def __init__(self, name: str, timeout: float = 10.0):
        self.name = name
        self.timeout = timeout
    
    async def check(self) -> HealthCheckResult:
        """Perform the health check."""
        start_time = time.time()
        timestamp = datetime.now()
        
        try:
            # Run the check with timeout
            result = await asyncio.wait_for(self._perform_check(), timeout=self.timeout)
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=result.get('status', HealthStatus.UNKNOWN),
                message=result.get('message', 'Check completed'),
                duration_ms=duration_ms,
                timestamp=timestamp,
                details=result.get('details'),
                error=result.get('error')
            )
            
        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout}s",
                duration_ms=duration_ms,
                timestamp=timestamp,
                error="timeout"
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                duration_ms=duration_ms,
                timestamp=timestamp,
                error=str(e)
            )
    
    async def _perform_check(self) -> Dict[str, Any]:
        """Override this method to implement the actual health check."""
        raise NotImplementedError


class RedisHealthCheck(HealthCheck):
    """Health check for Redis cache."""
    
    def __init__(self):
        super().__init__("redis_cache", timeout=5.0)
    
    async def _perform_check(self) -> Dict[str, Any]:
        from app.utils.cache import cache_manager
        
        # Test basic connectivity
        is_connected = await cache_manager.is_healthy()
        
        if not is_connected:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': 'Redis connection failed',
                'details': {'connected': False}
            }
        
        # Test basic operations
        test_key = f"health_check_{int(time.time())}"
        test_value = {"test": True, "timestamp": datetime.now().isoformat()}
        
        # Test SET
        set_success = await cache_manager.set(test_key, test_value, ttl=60)
        if not set_success:
            return {
                'status': HealthStatus.DEGRADED,
                'message': 'Redis SET operation failed',
                'details': {'connected': True, 'set_operation': False}
            }
        
        # Test GET
        retrieved_value = await cache_manager.get(test_key)
        if retrieved_value != test_value:
            return {
                'status': HealthStatus.DEGRADED,
                'message': 'Redis GET operation failed',
                'details': {'connected': True, 'get_operation': False}
            }
        
        # Test DELETE
        delete_success = await cache_manager.delete(test_key)
        
        return {
            'status': HealthStatus.HEALTHY,
            'message': 'Redis is fully operational',
            'details': {
                'connected': True,
                'set_operation': True,
                'get_operation': True,
                'delete_operation': delete_success
            }
        }


class APIFootballHealthCheck(HealthCheck):
    """Health check for API-Football service."""
    
    def __init__(self):
        super().__init__("api_football", timeout=15.0)
    
    async def _perform_check(self) -> Dict[str, Any]:
        from app.services.fetch_data import FootballDataFetcher
        
        fetcher = FootballDataFetcher()
        
        try:
            # Test API status endpoint
            health_result = await fetcher.health_check()
            
            if health_result.get('status') == 'healthy':
                return {
                    'status': HealthStatus.HEALTHY,
                    'message': 'API-Football is operational',
                    'details': health_result
                }
            else:
                return {
                    'status': HealthStatus.DEGRADED,
                    'message': 'API-Football reported issues',
                    'details': health_result
                }
                
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'API-Football check failed: {str(e)}',
                'error': str(e)
            }


class OpenRouterHealthCheck(HealthCheck):
    """Health check for OpenRouter AI service."""
    
    def __init__(self):
        super().__init__("openrouter_ai", timeout=20.0)
    
    async def _perform_check(self) -> Dict[str, Any]:
        from app.services.ai_summary import AISummaryGenerator
        
        ai_service = AISummaryGenerator()
        
        try:
            # Test AI service connection
            test_result = await ai_service.test_connection()
            
            if test_result.get('status') == 'success':
                return {
                    'status': HealthStatus.HEALTHY,
                    'message': 'OpenRouter AI is operational',
                    'details': {
                        'model': test_result.get('model'),
                        'response_received': bool(test_result.get('response'))
                    }
                }
            else:
                return {
                    'status': HealthStatus.UNHEALTHY,
                    'message': 'OpenRouter AI connection failed',
                    'details': test_result
                }
                
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'OpenRouter AI check failed: {str(e)}',
                'error': str(e)
            }


class TelegramBotHealthCheck(HealthCheck):
    """Health check for Telegram bot."""
    
    def __init__(self):
        super().__init__("telegram_bot", timeout=10.0)
    
    async def _perform_check(self) -> Dict[str, Any]:
        from app.bots.telegram_bot import bot_instance
        
        try:
            # Test bot health
            bot_health = await bot_instance.health_check()
            
            if bot_health.get('status') == 'healthy':
                return {
                    'status': HealthStatus.HEALTHY,
                    'message': 'Telegram bot is operational',
                    'details': {
                        'is_running': bot_health.get('is_running', False),
                        'active_sessions': bot_health.get('active_sessions', 0)
                    }
                }
            else:
                return {
                    'status': HealthStatus.DEGRADED,
                    'message': 'Telegram bot has issues',
                    'details': bot_health
                }
                
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'Telegram bot check failed: {str(e)}',
                'error': str(e)
            }


class WorkflowHealthCheck(HealthCheck):
    """Health check for the complete workflow."""
    
    def __init__(self):
        super().__init__("workflow_pipeline", timeout=30.0)
    
    async def _perform_check(self) -> Dict[str, Any]:
        from app.services.workflow import workflow
        
        try:
            # Test workflow health
            workflow_health = await workflow.get_workflow_health()
            
            status_map = {
                'healthy': HealthStatus.HEALTHY,
                'degraded': HealthStatus.DEGRADED,
                'unhealthy': HealthStatus.UNHEALTHY
            }
            
            workflow_status = workflow_health.get('status', 'unknown')
            health_status = status_map.get(workflow_status, HealthStatus.UNKNOWN)
            
            return {
                'status': health_status,
                'message': f'Workflow pipeline is {workflow_status}',
                'details': {
                    'success_rate': workflow_health.get('statistics', {}).get('success_rate_percent', 0),
                    'cache_hit_rate': workflow_health.get('statistics', {}).get('cache_hit_rate_percent', 0),
                    'total_predictions': workflow_health.get('statistics', {}).get('total_predictions', 0)
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'Workflow health check failed: {str(e)}',
                'error': str(e)
            }


class SystemResourcesHealthCheck(HealthCheck):
    """Health check for system resources."""
    
    def __init__(self):
        super().__init__("system_resources", timeout=5.0)
    
    async def _perform_check(self) -> Dict[str, Any]:
        try:
            import psutil
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Determine health based on resource usage
            status = HealthStatus.HEALTHY
            issues = []
            
            if cpu_percent > 90:
                status = HealthStatus.DEGRADED
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory.percent > 90:
                status = HealthStatus.DEGRADED
                issues.append(f"High memory usage: {memory.percent:.1f}%")
            
            if disk.percent > 90:
                status = HealthStatus.DEGRADED
                issues.append(f"High disk usage: {disk.percent:.1f}%")
            
            message = "System resources are healthy"
            if issues:
                message = f"System resource issues: {', '.join(issues)}"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_percent': disk.percent,
                    'disk_free_gb': disk.free / (1024**3)
                }
            }
            
        except ImportError:
            return {
                'status': HealthStatus.UNKNOWN,
                'message': 'psutil not available for system monitoring',
                'details': {'psutil_available': False}
            }
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'System resources check failed: {str(e)}',
                'error': str(e)
            }


class HealthCheckManager:
    """Manages and coordinates all health checks."""
    
    def __init__(self):
        self.health_checks: List[HealthCheck] = [
            RedisHealthCheck(),
            APIFootballHealthCheck(),
            OpenRouterHealthCheck(),
            TelegramBotHealthCheck(),
            WorkflowHealthCheck(),
            SystemResourcesHealthCheck()
        ]
        self.last_check_time: Optional[datetime] = None
        self.last_results: List[HealthCheckResult] = []
    
    async def run_all_checks(self, parallel: bool = True) -> List[HealthCheckResult]:
        """Run all health checks."""
        logger.info("Running comprehensive health checks")
        
        if parallel:
            # Run checks in parallel
            tasks = [check.check() for check in self.health_checks]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    final_results.append(HealthCheckResult(
                        name=self.health_checks[i].name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check failed with exception: {str(result)}",
                        duration_ms=0,
                        timestamp=datetime.now(),
                        error=str(result)
                    ))
                else:
                    final_results.append(result)
        else:
            # Run checks sequentially
            final_results = []
            for check in self.health_checks:
                result = await check.check()
                final_results.append(result)
        
        self.last_check_time = datetime.now()
        self.last_results = final_results
        
        logger.info(f"Health checks completed: {self._get_summary_stats(final_results)}")
        return final_results
    
    def _get_summary_stats(self, results: List[HealthCheckResult]) -> str:
        """Get summary statistics for health check results."""
        healthy = sum(1 for r in results if r.status == HealthStatus.HEALTHY)
        degraded = sum(1 for r in results if r.status == HealthStatus.DEGRADED)
        unhealthy = sum(1 for r in results if r.status == HealthStatus.UNHEALTHY)
        unknown = sum(1 for r in results if r.status == HealthStatus.UNKNOWN)
        
        return f"{healthy} healthy, {degraded} degraded, {unhealthy} unhealthy, {unknown} unknown"
    
    async def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        if not self.last_results:
            results = await self.run_all_checks()
        else:
            results = self.last_results
        
        # Determine overall status
        statuses = [r.status for r in results]
        
        if HealthStatus.UNHEALTHY in statuses:
            overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN
        
        # Get circuit breaker stats
        circuit_stats = await get_all_circuit_breaker_stats()
        
        # Get monitoring metrics
        monitoring_stats = metrics_collector.get_health_status()
        
        return {
            'overall_status': overall_status.value,
            'timestamp': datetime.now().isoformat(),
            'last_check_time': self.last_check_time.isoformat() if self.last_check_time else None,
            'summary': self._get_summary_stats(results),
            'checks': [
                {
                    'name': r.name,
                    'status': r.status.value,
                    'message': r.message,
                    'duration_ms': r.duration_ms,
                    'details': r.details,
                    'error': r.error
                }
                for r in results
            ],
            'circuit_breakers': {
                name: {
                    'state': stats.state.value,
                    'failure_rate': stats.failure_rate,
                    'total_requests': stats.total_requests
                }
                for name, stats in circuit_stats.items()
            },
            'monitoring': monitoring_stats
        }
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get a lightweight health summary."""
        overall_health = await self.get_overall_health()
        
        return {
            'status': overall_health['overall_status'],
            'timestamp': overall_health['timestamp'],
            'summary': overall_health['summary'],
            'checks_count': len(overall_health['checks']),
            'circuit_breakers_count': len(overall_health['circuit_breakers'])
        }


# Global health check manager
health_manager = HealthCheckManager()