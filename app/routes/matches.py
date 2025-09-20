"""FastAPI routes for match data and predictions."""

from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
import asyncio

from app.models.schemas import (
    Match, MatchPredictions, AnalysisResult, HealthCheck,
    TeamStats, MatchFeatures
)
from app.services.fetch_data import FootballDataFetcher, APIFootballError
from app.services.analyzer import MatchAnalyzer, AnalysisError
from app.utils.logger import get_logger, set_correlation_id, get_correlation_id
from app.utils.cache import cache_manager, get_cached_match_predictions, cache_match_predictions
from app.utils.monitoring import RequestTracker, metrics_collector
from app.services.workflow import workflow
import uuid

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/matches", tags=["matches"])

# Global service instances
data_fetcher = FootballDataFetcher()
analyzer = MatchAnalyzer(data_fetcher)


async def get_data_fetcher() -> FootballDataFetcher:
    """Dependency to get data fetcher instance."""
    return data_fetcher


async def get_analyzer() -> MatchAnalyzer:
    """Dependency to get analyzer instance."""
    return analyzer


def set_request_correlation_id():
    """Set correlation ID for request tracking."""
    correlation_id = str(uuid.uuid4())[:8]
    set_correlation_id(correlation_id)
    return correlation_id


@router.get("/today", response_model=List[Match])
async def get_todays_matches(
    league_ids: Optional[List[int]] = Query(None, description="Filter by league IDs"),
    correlation_id: str = Depends(set_request_correlation_id)
) -> List[Match]:
    """
    Get today's football matches.
    
    Args:
        league_ids: Optional list of league IDs to filter matches
        
    Returns:
        List of today's matches
        
    Raises:
        HTTPException: If data fetching fails
    """
    with RequestTracker("get_todays_matches_api"):
        try:
            logger.info(f"Fetching today's matches via workflow (correlation_id: {correlation_id})")
            
            # Use workflow instead of direct fetcher
            matches = await workflow.get_todays_matches(correlation_id)
            
            logger.info(f"Retrieved {len(matches)} matches for today via workflow")
            return matches
            
        except APIFootballError as e:
            logger.error(f"API Football error: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"External API error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error fetching today's matches: {e}")
            raise HTTPException(
                status_code=500,
                detail="Internal server error while fetching matches"
            )


@router.get("/{match_id}/predictions", response_model=MatchPredictions)
async def get_match_predictions(
    match_id: int,
    force_refresh: bool = Query(False, description="Force refresh predictions"),
    analyzer_service: MatchAnalyzer = Depends(get_analyzer),
    fetcher: FootballDataFetcher = Depends(get_data_fetcher),
    correlation_id: str = Depends(set_request_correlation_id)
) -> MatchPredictions:
    """
    Get predictions for a specific match.
    
    Args:
        match_id: Match ID
        force_refresh: Force refresh predictions (skip cache)
        
    Returns:
        Match predictions
        
    Raises:
        HTTPException: If match not found or prediction fails
    """
    try:
        logger.info(f"Getting predictions for match {match_id} (correlation_id: {correlation_id})")
        
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached_predictions = await get_cached_match_predictions(match_id)
            if cached_predictions:
                logger.info(f"Returning cached predictions for match {match_id}")
                return cached_predictions
        
        # Get match data
        matches = await fetcher.get_todays_matches()
        match = next((m for m in matches if m.id == match_id), None)
        
        if not match:
            logger.warning(f"Match {match_id} not found")
            raise HTTPException(
                status_code=404,
                detail=f"Match with ID {match_id} not found"
            )
        
        # Generate predictions
        predictions = await analyzer_service.generate_predictions(match)
        
        # Cache predictions
        await cache_match_predictions(match_id, predictions)
        
        logger.info(f"Generated predictions for match {match_id}")
        return predictions
        
    except HTTPException:
        raise
    except AnalysisError as e:
        logger.error(f"Analysis error for match {match_id}: {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Analysis failed: {str(e)}"
        )
    except APIFootballError as e:
        logger.error(f"API Football error for match {match_id}: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"External API error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error getting predictions for match {match_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while generating predictions"
        )


@router.post("/{match_id}/analyze", response_model=AnalysisResult)
async def analyze_match(
    match_id: int,
    background_tasks: BackgroundTasks,
    analyzer_service: MatchAnalyzer = Depends(get_analyzer),
    fetcher: FootballDataFetcher = Depends(get_data_fetcher),
    correlation_id: str = Depends(set_request_correlation_id)
) -> AnalysisResult:
    """
    Perform comprehensive match analysis.
    
    Args:
        match_id: Match ID
        background_tasks: Background tasks for async processing
        
    Returns:
        Complete analysis result
        
    Raises:
        HTTPException: If match not found or analysis fails
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"Starting comprehensive analysis for match {match_id} (correlation_id: {correlation_id})")
        
        # Get match data
        matches = await fetcher.get_todays_matches()
        match = next((m for m in matches if m.id == match_id), None)
        
        if not match:
            logger.warning(f"Match {match_id} not found for analysis")
            raise HTTPException(
                status_code=404,
                detail=f"Match with ID {match_id} not found"
            )
        
        # Get team statistics
        home_stats = await fetcher.get_team_stats(match.home_team.id)
        away_stats = await fetcher.get_team_stats(match.away_team.id)
        
        # Extract features
        features = await analyzer_service.extract_features(match, home_stats, away_stats)
        
        # Generate predictions
        predictions = await analyzer_service.generate_predictions(match)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Cache results in background
        background_tasks.add_task(cache_match_predictions, match_id, predictions)
        
        result = AnalysisResult(
            match=match,
            predictions=predictions,
            features=features,
            processing_time_ms=int(processing_time),
            success=True
        )
        
        logger.info(f"Completed analysis for match {match_id} in {processing_time:.0f}ms")
        return result
        
    except HTTPException:
        raise
    except (AnalysisError, APIFootballError) as e:
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.error(f"Analysis failed for match {match_id}: {e}")
        
        # Return partial result with error
        return AnalysisResult(
            match=Match(
                id=match_id,
                home_team={"id": 0, "name": "Unknown"},
                away_team={"id": 0, "name": "Unknown"},
                kickoff_time=datetime.now(),
                league="Unknown",
                status="NS"
            ),
            predictions=None,
            features=None,
            processing_time_ms=int(processing_time),
            success=False,
            error_message=str(e)
        )
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.error(f"Unexpected error analyzing match {match_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during analysis"
        )


@router.get("/{match_id}/features", response_model=MatchFeatures)
async def get_match_features(
    match_id: int,
    analyzer_service: MatchAnalyzer = Depends(get_analyzer),
    fetcher: FootballDataFetcher = Depends(get_data_fetcher),
    correlation_id: str = Depends(set_request_correlation_id)
) -> MatchFeatures:
    """
    Get extracted features for a match.
    
    Args:
        match_id: Match ID
        
    Returns:
        Match features for ML models
        
    Raises:
        HTTPException: If match not found or feature extraction fails
    """
    try:
        logger.info(f"Extracting features for match {match_id} (correlation_id: {correlation_id})")
        
        # Get match data
        matches = await fetcher.get_todays_matches()
        match = next((m for m in matches if m.id == match_id), None)
        
        if not match:
            raise HTTPException(
                status_code=404,
                detail=f"Match with ID {match_id} not found"
            )
        
        # Get team statistics
        home_stats = await fetcher.get_team_stats(match.home_team.id)
        away_stats = await fetcher.get_team_stats(match.away_team.id)
        
        # Extract features
        features = await analyzer_service.extract_features(match, home_stats, away_stats)
        
        logger.info(f"Features extracted for match {match_id}")
        return features
        
    except HTTPException:
        raise
    except (AnalysisError, APIFootballError) as e:
        logger.error(f"Error extracting features for match {match_id}: {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Feature extraction failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error extracting features for match {match_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during feature extraction"
        )


@router.get("/{match_id}/teams/{team_id}/stats", response_model=TeamStats)
async def get_team_stats(
    match_id: int,
    team_id: int,
    season: int = Query(2024, description="Season year"),
    fetcher: FootballDataFetcher = Depends(get_data_fetcher),
    correlation_id: str = Depends(set_request_correlation_id)
) -> TeamStats:
    """
    Get team statistics for a specific team in a match.
    
    Args:
        match_id: Match ID (for context)
        team_id: Team ID
        season: Season year
        
    Returns:
        Team statistics
        
    Raises:
        HTTPException: If team not found or stats unavailable
    """
    try:
        logger.info(f"Getting stats for team {team_id} in match {match_id} (correlation_id: {correlation_id})")
        
        stats = await fetcher.get_team_stats(team_id, season)
        
        logger.info(f"Retrieved stats for team {team_id}")
        return stats
        
    except APIFootballError as e:
        logger.error(f"API error getting stats for team {team_id}: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"External API error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error getting stats for team {team_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while fetching team stats"
        )


@router.get("/health", response_model=HealthCheck)
async def health_check(
    fetcher: FootballDataFetcher = Depends(get_data_fetcher),
    correlation_id: str = Depends(set_request_correlation_id)
) -> HealthCheck:
    """
    Health check endpoint for monitoring.
    
    Returns:
        Health status of the service and dependencies
    """
    try:
        logger.info(f"Health check requested (correlation_id: {correlation_id})")
        
        services = {}
        
        # Check API-Football
        try:
            api_health = await fetcher.health_check()
            services["api_football"] = api_health.get("status", "unknown")
        except Exception as e:
            services["api_football"] = f"error: {str(e)}"
        
        # Check cache
        try:
            cache_healthy = await cache_manager.is_healthy()
            services["cache"] = "healthy" if cache_healthy else "unhealthy"
        except Exception as e:
            services["cache"] = f"error: {str(e)}"
        
        # Check AI service
        try:
            ai_health = await analyzer.ai_summary.test_connection()
            services["ai_service"] = ai_health.get("status", "unknown")
        except Exception as e:
            services["ai_service"] = f"error: {str(e)}"
        
        # Overall status
        overall_status = "healthy" if all(
            status in ["healthy", "success"] for status in services.values()
            if not status.startswith("error")
        ) else "degraded"
        
        health = HealthCheck(
            status=overall_status,
            services=services
        )
        
        logger.info(f"Health check completed: {overall_status}")
        return health
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheck(
            status="unhealthy",
            services={"error": str(e)}
        )


@router.get("/stats", response_model=dict)
async def get_service_stats(
    correlation_id: str = Depends(set_request_correlation_id)
) -> dict:
    """
    Get service statistics and metrics.
    
    Returns:
        Service statistics including cache metrics, model info, etc.
    """
    try:
        logger.info(f"Service stats requested (correlation_id: {correlation_id})")
        
        stats = {
            "timestamp": datetime.now().isoformat(),
            "correlation_id": correlation_id
        }
        
        # Cache statistics
        try:
            cache_stats = await cache_manager.get_stats()
            stats["cache"] = cache_stats
        except Exception as e:
            stats["cache"] = {"error": str(e)}
        
        # ML model information
        try:
            model_info = analyzer.ml_models.get_model_info()
            stats["ml_models"] = model_info
        except Exception as e:
            stats["ml_models"] = {"error": str(e)}
        
        # Service uptime (simplified)
        stats["service"] = {
            "status": "running",
            "version": "1.0.0"
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting service stats: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.delete("/cache", response_model=dict)
async def clear_cache(
    pattern: Optional[str] = Query(None, description="Cache key pattern to clear"),
    correlation_id: str = Depends(set_request_correlation_id)
) -> dict:
    """
    Clear cache entries.
    
    Args:
        pattern: Optional pattern to match cache keys
        
    Returns:
        Number of cache entries cleared
    """
    try:
        logger.info(f"Cache clear requested with pattern: {pattern} (correlation_id: {correlation_id})")
        
        if pattern:
            cleared = await cache_manager.clear_pattern(pattern)
        else:
            cleared = await cache_manager.clear_pattern("*")
        
        result = {
            "cleared_entries": cleared,
            "pattern": pattern or "*",
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Cleared {cleared} cache entries")
        return result
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return {
            "error": str(e),
            "cleared_entries": 0,
            "timestamp": datetime.now().isoformat()
        }


# Note: Exception handlers are implemented at endpoint level with try-catch blocks
# Global exception handlers would be added to the main FastAPI app, not the router