"""Monitoring and metrics utilities."""

import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict, deque
import asyncio

from app.utils.logger import get_logger
from app.utils.correlation import get_correlation_id

logger = get_logger(__name__)


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    correlation_id: Optional[str] = None


@dataclass
class RequestMetrics:
    """Request-level metrics."""
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    status: str = "in_progress"
    error: Optional[str] = None
    correlation_id: Optional[str] = None
    
    def finish(self, status: str = "success", error: Optional[str] = None):
        """Mark request as finished."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = status
        self.error = error


class MetricsCollector:
    """Collects and stores application metrics."""
    
    def __init__(self, max_points: int = 1000):
        self.max_points = max_points
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self.counters: Dict[str, int] = defaultdict(int)
        self.active_requests: Dict[str, RequestMetrics] = {}
        self.start_time = datetime.now()
    
    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            labels=labels or {},
            correlation_id=get_correlation_id()
        )
        
        self.metrics[name].append(point)
        logger.debug(f"Recorded metric {name}: {value}")
    
    def increment_counter(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        key = f"{name}:{':'.join(f'{k}={v}' for k, v in (labels or {}).items())}"
        self.counters[key] += 1
        logger.debug(f"Incremented counter {name}")
    
    def start_request(self, request_id: str) -> RequestMetrics:
        """Start tracking a request."""
        metrics = RequestMetrics(
            start_time=time.time(),
            correlation_id=get_correlation_id()
        )
        self.active_requests[request_id] = metrics
        return metrics
    
    def finish_request(self, request_id: str, status: str = "success", error: Optional[str] = None):
        """Finish tracking a request."""
        if request_id in self.active_requests:
            metrics = self.active_requests[request_id]
            metrics.finish(status, error)
            
            # Record duration metric
            self.record_metric(
                "request_duration_seconds",
                metrics.duration,
                {"status": status}
            )
            
            # Increment request counter
            self.increment_counter("requests_total", {"status": status})
            
            # Clean up
            del self.active_requests[request_id]
            
            logger.debug(f"Finished request {request_id}: {status} in {metrics.duration:.3f}s")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        now = datetime.now()
        uptime = now - self.start_time
        
        # Calculate recent averages (last 5 minutes)
        recent_cutoff = now - timedelta(minutes=5)
        
        summary = {
            "timestamp": now.isoformat(),
            "uptime_seconds": uptime.total_seconds(),
            "active_requests": len(self.active_requests),
            "counters": dict(self.counters),
            "metrics": {}
        }
        
        # Process each metric
        for name, points in self.metrics.items():
            if not points:
                continue
                
            recent_points = [p for p in points if p.timestamp >= recent_cutoff]
            all_values = [p.value for p in points]
            recent_values = [p.value for p in recent_points]
            
            summary["metrics"][name] = {
                "total_points": len(points),
                "recent_points": len(recent_points),
                "all_time": {
                    "min": min(all_values) if all_values else 0,
                    "max": max(all_values) if all_values else 0,
                    "avg": sum(all_values) / len(all_values) if all_values else 0,
                    "latest": all_values[-1] if all_values else 0
                },
                "recent_5min": {
                    "min": min(recent_values) if recent_values else 0,
                    "max": max(recent_values) if recent_values else 0,
                    "avg": sum(recent_values) / len(recent_values) if recent_values else 0
                }
            }
        
        return summary
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status based on metrics."""
        now = datetime.now()
        
        # Check for recent errors
        error_requests = sum(1 for m in self.active_requests.values() if m.error)
        total_requests = len(self.active_requests)
        
        # Check recent request durations
        recent_durations = []
        if "request_duration_seconds" in self.metrics:
            recent_cutoff = now - timedelta(minutes=5)
            recent_durations = [
                p.value for p in self.metrics["request_duration_seconds"]
                if p.timestamp >= recent_cutoff
            ]
        
        # Determine health status
        status = "healthy"
        issues = []
        
        if error_requests > 0:
            issues.append(f"{error_requests} requests with errors")
            status = "degraded"
        
        if recent_durations and max(recent_durations) > 30:  # 30 second threshold
            issues.append("Slow requests detected")
            status = "degraded"
        
        if total_requests > 100:  # Too many concurrent requests
            issues.append("High concurrent request load")
            status = "degraded"
        
        return {
            "status": status,
            "timestamp": now.isoformat(),
            "issues": issues,
            "metrics": {
                "active_requests": total_requests,
                "error_requests": error_requests,
                "avg_recent_duration": sum(recent_durations) / len(recent_durations) if recent_durations else 0,
                "max_recent_duration": max(recent_durations) if recent_durations else 0
            }
        }


# Global metrics collector instance
metrics_collector = MetricsCollector()


# Convenience functions
def record_prediction_time(duration: float, prediction_type: str):
    """Record prediction generation time."""
    metrics_collector.record_metric(
        "prediction_duration_seconds",
        duration,
        {"type": prediction_type}
    )


def record_api_call(duration: float, endpoint: str, status: str):
    """Record API call metrics."""
    metrics_collector.record_metric(
        "api_call_duration_seconds",
        duration,
        {"endpoint": endpoint, "status": status}
    )
    
    metrics_collector.increment_counter(
        "api_calls_total",
        {"endpoint": endpoint, "status": status}
    )


def record_cache_operation(operation: str, hit: bool):
    """Record cache operation metrics."""
    metrics_collector.increment_counter(
        "cache_operations_total",
        {"operation": operation, "result": "hit" if hit else "miss"}
    )


class RequestTracker:
    """Context manager for tracking request metrics."""
    
    def __init__(self, request_name: str):
        self.request_name = request_name
        self.request_id = f"{request_name}_{get_correlation_id() or 'unknown'}"
        self.metrics = None
    
    def __enter__(self):
        self.metrics = metrics_collector.start_request(self.request_id)
        return self.metrics
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            metrics_collector.finish_request(self.request_id, "success")
        else:
            metrics_collector.finish_request(
                self.request_id, 
                "error", 
                str(exc_val) if exc_val else "unknown_error"
            )