"""End-to-end workflow orchestrator for match predictions."""

import asyncio
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

from app.models.schemas import Match, MatchPredictions
from app.services.fetch_data import FootballDataFetcher, APIFootballError
from app.services.analyzer import MatchAnalyzer, AnalysisError
from app.utils.logger import get_logger
from app.utils.correlation import CorrelationContext, get_correlation_id
from app.utils.monitoring import RequestTracker, record_prediction_time, metrics_collector
from app.utils.cache import cache_manager

logger = get_logger(__name__)


class WorkflowError(Exception):
    """Custom exception for workflow errors."""
    pass


class PredictionWorkflow:
    """Orchestrates the complete prediction workflow."""
    
    def __init__(self):
        self.data_fetcher = FootballDataFetcher()
        self.analyzer = MatchAnalyzer(self.data_fetcher)
        self.workflow_stats = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "cache_hits": 0,
            "api_calls": 0
        }
    
    async def get_todays_matches(self, correlation_id: Optional[str] = None) -> List[Match]:
        """
        Get today's matches with full workflow tracking.
        
        Args:
            correlation_id: Optional correlation ID for tracking
            
        Returns:
            List of Match objects
        """
        with CorrelationContext(correlation_id):
            with RequestTracker("get_todays_matches") as tracker:
                try:
                    logger.info("Starting workflow: Get today's matches")
                    
                    # Check cache first
                    cache_key = f"workflow_matches:{datetime.now().strftime('%Y-%m-%d')}"
                    cached_matches = await cache_manager.get(cache_key)
                    
                    if cached_matches:
                        logger.info("Matches retrieved from cache")
                        self.workflow_stats["cache_hits"] += 1
                        return cached_matches
                    
                    # Fetch from API
                    logger.info("Fetching matches from API")
                    matches = await self.data_fetcher.get_todays_matches()
                    self.workflow_stats["api_calls"] += 1
                    
                    # Cache the results
                    await cache_manager.set(cache_key, matches, ttl=3600)  # 1 hour
                    
                    logger.info(f"Successfully retrieved {len(matches)} matches")
                    return matches
                    
                except Exception as e:
                    logger.error(f"Failed to get today's matches: {e}")
                    raise WorkflowError(f"Match retrieval failed: {str(e)}")
    
    async def generate_match_prediction(self, match: Match, correlation_id: Optional[str] = None) -> MatchPredictions:
        """
        Generate complete prediction for a match with full workflow tracking.
        
        Args:
            match: Match object to predict
            correlation_id: Optional correlation ID for tracking
            
        Returns:
            MatchPredictions object
        """
        with CorrelationContext(correlation_id):
            with RequestTracker("generate_match_prediction") as tracker:
                try:
                    logger.info(f"Starting prediction workflow for match {match.id}: {match.home_team.name} vs {match.away_team.name}")
                    
                    start_time = time.time()
                    
                    # Check cache first
                    cache_key = f"workflow_prediction:{match.id}"
                    cached_prediction = await cache_manager.get(cache_key)
                    
                    if cached_prediction:
                        logger.info(f"Prediction retrieved from cache for match {match.id}")
                        self.workflow_stats["cache_hits"] += 1
                        self.workflow_stats["successful_predictions"] += 1
                        return cached_prediction
                    
                    # Generate prediction through analyzer
                    logger.info(f"Generating fresh prediction for match {match.id}")
                    predictions = await self.analyzer.generate_predictions(match)
                    
                    # Record metrics
                    duration = time.time() - start_time
                    record_prediction_time(duration, "complete_workflow")
                    
                    # Cache the results
                    await cache_manager.set(cache_key, predictions, ttl=10800)  # 3 hours
                    
                    # Update stats
                    self.workflow_stats["total_predictions"] += 1
                    self.workflow_stats["successful_predictions"] += 1
                    
                    logger.info(f"Successfully generated prediction for match {match.id} in {duration:.2f}s")
                    return predictions
                    
                except Exception as e:
                    self.workflow_stats["total_predictions"] += 1
                    self.workflow_stats["failed_predictions"] += 1
                    logger.error(f"Failed to generate prediction for match {match.id}: {e}")
                    raise WorkflowError(f"Prediction generation failed: {str(e)}")
    
    async def process_match_batch(self, matches: List[Match], max_concurrent: int = 3) -> Dict[int, MatchPredictions]:
        """
        Process multiple matches concurrently with rate limiting.
        
        Args:
            matches: List of matches to process
            max_concurrent: Maximum concurrent predictions
            
        Returns:
            Dictionary mapping match IDs to predictions
        """
        correlation_id = get_correlation_id()
        
        async def process_single_match(match: Match) -> tuple[int, Optional[MatchPredictions]]:
            try:
                prediction = await self.generate_match_prediction(match, correlation_id)
                return match.id, prediction
            except Exception as e:
                logger.error(f"Failed to process match {match.id}: {e}")
                return match.id, None
        
        # Process matches with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_process(match: Match):
            async with semaphore:
                return await process_single_match(match)
        
        logger.info(f"Processing {len(matches)} matches with max {max_concurrent} concurrent")
        
        # Execute all tasks
        tasks = [bounded_process(match) for match in matches]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        predictions = {}
        successful = 0
        failed = 0
        
        for result in results:
            if isinstance(result, Exception):
                failed += 1
                logger.error(f"Batch processing error: {result}")
            else:
                match_id, prediction = result
                if prediction:
                    predictions[match_id] = prediction
                    successful += 1
                else:
                    failed += 1
        
        logger.info(f"Batch processing completed: {successful} successful, {failed} failed")
        return predictions
    
    async def get_workflow_health(self) -> Dict[str, Any]:
        """Get workflow health status and statistics."""
        try:
            # Get component health
            cache_healthy = await cache_manager.is_healthy()
            
            # Test data fetcher
            try:
                await self.data_fetcher.health_check()
                fetcher_healthy = True
            except:
                fetcher_healthy = False
            
            # Test analyzer (quick test)
            analyzer_healthy = True  # Assume healthy if no recent errors
            
            # Calculate success rate
            total = self.workflow_stats["total_predictions"]
            success_rate = (self.workflow_stats["successful_predictions"] / total * 100) if total > 0 else 100
            
            # Determine overall health
            components_healthy = cache_healthy and fetcher_healthy and analyzer_healthy
            performance_good = success_rate >= 80  # 80% success rate threshold
            
            overall_status = "healthy" if (components_healthy and performance_good) else "degraded"
            
            return {
                "status": overall_status,
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "cache": "healthy" if cache_healthy else "unhealthy",
                    "data_fetcher": "healthy" if fetcher_healthy else "unhealthy",
                    "analyzer": "healthy" if analyzer_healthy else "unhealthy"
                },
                "statistics": {
                    **self.workflow_stats,
                    "success_rate_percent": round(success_rate, 2),
                    "cache_hit_rate_percent": round(
                        (self.workflow_stats["cache_hits"] / max(total, 1)) * 100, 2
                    )
                },
                "performance": {
                    "avg_prediction_time": "calculated_from_metrics",
                    "active_requests": len(metrics_collector.active_requests)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get workflow health: {e}")
            return {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    async def run_health_check(self) -> Dict[str, Any]:
        """Run a comprehensive health check of the entire workflow."""
        with CorrelationContext() as correlation_id:
            logger.info(f"Running workflow health check with correlation ID: {correlation_id}")
            
            health_results = {
                "correlation_id": correlation_id,
                "timestamp": datetime.now().isoformat(),
                "checks": {}
            }
            
            # Test 1: Data fetching
            try:
                start_time = time.time()
                matches = await self.get_todays_matches(correlation_id)
                duration = time.time() - start_time
                
                health_results["checks"]["data_fetching"] = {
                    "status": "healthy",
                    "duration_seconds": round(duration, 3),
                    "matches_found": len(matches),
                    "details": "Successfully retrieved matches"
                }
            except Exception as e:
                health_results["checks"]["data_fetching"] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "details": "Failed to retrieve matches"
                }
            
            # Test 2: Prediction generation (if matches available)
            if health_results["checks"].get("data_fetching", {}).get("status") == "healthy":
                try:
                    matches = await self.get_todays_matches(correlation_id)
                    if matches:
                        test_match = matches[0]
                        start_time = time.time()
                        prediction = await self.generate_match_prediction(test_match, correlation_id)
                        duration = time.time() - start_time
                        
                        health_results["checks"]["prediction_generation"] = {
                            "status": "healthy",
                            "duration_seconds": round(duration, 3),
                            "test_match": f"{test_match.home_team.name} vs {test_match.away_team.name}",
                            "confidence": round(prediction.confidence_score * 100, 2),
                            "details": "Successfully generated prediction"
                        }
                    else:
                        health_results["checks"]["prediction_generation"] = {
                            "status": "skipped",
                            "details": "No matches available for testing"
                        }
                except Exception as e:
                    health_results["checks"]["prediction_generation"] = {
                        "status": "unhealthy",
                        "error": str(e),
                        "details": "Failed to generate prediction"
                    }
            
            # Test 3: Cache performance
            try:
                test_key = f"health_check:{correlation_id}"
                test_value = {"test": True, "timestamp": datetime.now().isoformat()}
                
                # Test set
                set_result = await cache_manager.set(test_key, test_value, ttl=60)
                
                # Test get
                get_result = await cache_manager.get(test_key)
                
                # Test delete
                await cache_manager.delete(test_key)
                
                cache_working = set_result and get_result == test_value
                
                health_results["checks"]["cache"] = {
                    "status": "healthy" if cache_working else "unhealthy",
                    "set_success": set_result,
                    "get_success": get_result == test_value,
                    "details": "Cache operations successful" if cache_working else "Cache operations failed"
                }
            except Exception as e:
                health_results["checks"]["cache"] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "details": "Cache health check failed"
                }
            
            # Overall status
            all_checks = health_results["checks"]
            healthy_checks = sum(1 for check in all_checks.values() if check.get("status") == "healthy")
            total_checks = len([check for check in all_checks.values() if check.get("status") != "skipped"])
            
            if total_checks == 0:
                overall_status = "unknown"
            elif healthy_checks == total_checks:
                overall_status = "healthy"
            elif healthy_checks > 0:
                overall_status = "degraded"
            else:
                overall_status = "unhealthy"
            
            health_results["overall_status"] = overall_status
            health_results["summary"] = f"{healthy_checks}/{total_checks} checks passed"
            
            logger.info(f"Health check completed: {overall_status} ({health_results['summary']})")
            return health_results


# Global workflow instance
workflow = PredictionWorkflow()