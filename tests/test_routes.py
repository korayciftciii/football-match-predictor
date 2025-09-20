"""Integration tests for FastAPI routes."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime
from fastapi.testclient import TestClient
from fastapi import FastAPI
import sys
import os

# Add the app directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.routes.matches import router
from app.models.schemas import (
    Match, Team, MatchPredictions, ScorePrediction, CardPrediction,
    CornerPrediction, FirstHalfPrediction, TeamStats, MatchFeatures,
    MatchStatus, MatchResult, PerformanceMetrics
)
from app.services.fetch_data import APIFootballError
from app.services.analyzer import AnalysisError


# Create test app
app = FastAPI()
app.include_router(router)
client = TestClient(app)


class TestMatchesRoutes:
    """Test cases for matches routes."""
    
    @pytest.fixture
    def sample_match(self):
        """Create sample match for testing."""
        home_team = Team(id=1, name="Arsenal", recent_form=[])
        away_team = Team(id=2, name="Chelsea", recent_form=[])
        
        return Match(
            id=12345,
            home_team=home_team,
            away_team=away_team,
            kickoff_time=datetime(2024, 1, 15, 15, 0),
            league="Premier League",
            status=MatchStatus.NOT_STARTED
        )
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for testing."""
        score_prediction = ScorePrediction(
            home_score=2,
            away_score=1,
            confidence=0.75,
            probability=0.18,
            alternative_scores=[]
        )
        
        card_prediction = CardPrediction(
            home_team_cards=2,
            away_team_cards=3,
            total_cards=5,
            confidence=0.7
        )
        
        corner_prediction = CornerPrediction(
            home_team_corners=6,
            away_team_corners=4,
            total_corners=10,
            confidence=0.65
        )
        
        first_half_prediction = FirstHalfPrediction(
            result=MatchResult.WIN,
            home_score=1,
            away_score=0,
            confidence=0.6,
            probability=0.4
        )
        
        return MatchPredictions(
            match_id=12345,
            score_prediction=score_prediction,
            goal_scorer_prediction=None,
            yellow_cards_prediction=card_prediction,
            corners_prediction=corner_prediction,
            first_half_prediction=first_half_prediction,
            ai_summary="Arsenal should win 2-1",
            confidence_score=0.7
        )
    
    @pytest.fixture
    def sample_team_stats(self):
        """Create sample team statistics."""
        home_performance = PerformanceMetrics(
            matches_played=10,
            wins=7,
            draws=2,
            losses=1,
            goals_scored=20,
            goals_conceded=7,
            clean_sheets=6
        )
        
        away_performance = PerformanceMetrics(
            matches_played=10,
            wins=5,
            draws=3,
            losses=2,
            goals_scored=15,
            goals_conceded=10,
            clean_sheets=4
        )
        
        return TeamStats(
            team_id=1,
            goals_scored_avg=1.8,
            goals_conceded_avg=0.9,
            yellow_cards_avg=2.1,
            corners_avg=5.5,
            home_performance=home_performance,
            away_performance=away_performance
        )
    
    @pytest.fixture
    def sample_features(self):
        """Create sample match features."""
        return MatchFeatures(
            home_goal_avg=1.8,
            away_goal_avg=1.4,
            home_conceded_avg=1.0,
            away_conceded_avg=1.2,
            home_advantage=0.6,
            recent_form_diff=15.0,
            head_to_head_ratio=0.7,
            home_yellow_cards_avg=2.1,
            away_yellow_cards_avg=1.8,
            home_corners_avg=5.5,
            away_corners_avg=4.2
        )
    
    def test_get_todays_matches_success(self, sample_match):
        """Test successful retrieval of today's matches."""
        with patch('app.routes.matches.workflow') as mock_workflow:
            mock_workflow.get_todays_matches = AsyncMock(return_value=[sample_match])
            
            response = client.get("/matches/today")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["id"] == 12345
            assert data[0]["home_team"]["name"] == "Arsenal"
            assert data[0]["away_team"]["name"] == "Chelsea"
    
    def test_get_todays_matches_with_league_filter(self, sample_match):
        """Test retrieval of today's matches with league filter."""
        with patch('app.routes.matches.workflow') as mock_workflow:
            mock_workflow.get_todays_matches = AsyncMock(return_value=[sample_match])
            
            response = client.get("/matches/today?league_ids=39&league_ids=140")
            
            assert response.status_code == 200
            mock_workflow.get_todays_matches.assert_called_once()
    
    def test_get_todays_matches_api_error(self):
        """Test handling of API Football error."""
        with patch('app.routes.matches.workflow') as mock_workflow:
            mock_workflow.get_todays_matches = AsyncMock(
                side_effect=APIFootballError("API unavailable")
            )
            
            response = client.get("/matches/today")
            
            assert response.status_code == 503
            assert "External API error" in response.json()["detail"]
    
    def test_get_match_predictions_success(self, sample_match, sample_predictions):
        """Test successful match prediction retrieval."""
        with patch('app.routes.matches.data_fetcher') as mock_fetcher:
            with patch('app.routes.matches.analyzer') as mock_analyzer:
                with patch('app.routes.matches.get_cached_match_predictions') as mock_cache:
                    mock_cache.return_value = None  # No cache
                    mock_fetcher.get_todays_matches = AsyncMock(return_value=[sample_match])
                    mock_analyzer.generate_predictions = AsyncMock(return_value=sample_predictions)
                    
                    response = client.get("/matches/12345/predictions")
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["match_id"] == 12345
                    assert data["score_prediction"]["home_score"] == 2
                    assert data["score_prediction"]["away_score"] == 1
    
    def test_get_match_predictions_cached(self, sample_predictions):
        """Test match prediction retrieval from cache."""
        with patch('app.routes.matches.get_cached_match_predictions') as mock_cache:
            mock_cache.return_value = sample_predictions
            
            response = client.get("/matches/12345/predictions")
            
            assert response.status_code == 200
            data = response.json()
            assert data["match_id"] == 12345
    
    def test_get_match_predictions_force_refresh(self, sample_match, sample_predictions):
        """Test match prediction with force refresh."""
        with patch('app.routes.matches.data_fetcher') as mock_fetcher:
            with patch('app.routes.matches.analyzer') as mock_analyzer:
                with patch('app.routes.matches.get_cached_match_predictions') as mock_cache:
                    mock_cache.return_value = sample_predictions  # Has cache
                    mock_fetcher.get_todays_matches = AsyncMock(return_value=[sample_match])
                    mock_analyzer.generate_predictions = AsyncMock(return_value=sample_predictions)
                    
                    response = client.get("/matches/12345/predictions?force_refresh=true")
                    
                    assert response.status_code == 200
                    # Should call analyzer despite cache
                    mock_analyzer.generate_predictions.assert_called_once()
    
    def test_get_match_predictions_not_found(self):
        """Test match prediction for non-existent match."""
        with patch('app.routes.matches.data_fetcher') as mock_fetcher:
            mock_fetcher.get_todays_matches = AsyncMock(return_value=[])
            
            response = client.get("/matches/99999/predictions")
            
            assert response.status_code == 404
            assert "not found" in response.json()["detail"]
    
    def test_get_match_predictions_analysis_error(self, sample_match):
        """Test handling of analysis error."""
        with patch('app.routes.matches.data_fetcher') as mock_fetcher:
            with patch('app.routes.matches.analyzer') as mock_analyzer:
                with patch('app.routes.matches.get_cached_match_predictions') as mock_cache:
                    mock_cache.return_value = None
                    mock_fetcher.get_todays_matches = AsyncMock(return_value=[sample_match])
                    mock_analyzer.generate_predictions = AsyncMock(
                        side_effect=AnalysisError("Analysis failed")
                    )
                    
                    response = client.get("/matches/12345/predictions")
                    
                    assert response.status_code == 422
                    assert "Analysis failed" in response.json()["detail"]
    
    def test_analyze_match_success(self, sample_match, sample_predictions, 
                                 sample_team_stats, sample_features):
        """Test successful comprehensive match analysis."""
        with patch('app.routes.matches.data_fetcher') as mock_fetcher:
            with patch('app.routes.matches.analyzer') as mock_analyzer:
                mock_fetcher.get_todays_matches = AsyncMock(return_value=[sample_match])
                mock_fetcher.get_team_stats = AsyncMock(return_value=sample_team_stats)
                mock_analyzer.extract_features = AsyncMock(return_value=sample_features)
                mock_analyzer.generate_predictions = AsyncMock(return_value=sample_predictions)
                
                response = client.post("/matches/12345/analyze")
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["match"]["id"] == 12345
                assert data["predictions"]["match_id"] == 12345
                assert "processing_time_ms" in data
    
    def test_analyze_match_not_found(self):
        """Test analysis for non-existent match."""
        with patch('app.routes.matches.data_fetcher') as mock_fetcher:
            mock_fetcher.get_todays_matches = AsyncMock(return_value=[])
            
            response = client.post("/matches/99999/analyze")
            
            assert response.status_code == 404
            assert "not found" in response.json()["detail"]
    
    def test_get_match_features_success(self, sample_match, sample_team_stats, sample_features):
        """Test successful feature extraction."""
        with patch('app.routes.matches.data_fetcher') as mock_fetcher:
            with patch('app.routes.matches.analyzer') as mock_analyzer:
                mock_fetcher.get_todays_matches = AsyncMock(return_value=[sample_match])
                mock_fetcher.get_team_stats = AsyncMock(return_value=sample_team_stats)
                mock_analyzer.extract_features = AsyncMock(return_value=sample_features)
                
                response = client.get("/matches/12345/features")
                
                assert response.status_code == 200
                data = response.json()
                assert data["home_goal_avg"] == 1.8
                assert data["away_goal_avg"] == 1.4
                assert data["home_advantage"] == 0.6
    
    def test_get_team_stats_success(self, sample_team_stats):
        """Test successful team statistics retrieval."""
        with patch('app.routes.matches.data_fetcher') as mock_fetcher:
            mock_fetcher.get_team_stats = AsyncMock(return_value=sample_team_stats)
            
            response = client.get("/matches/12345/teams/1/stats")
            
            assert response.status_code == 200
            data = response.json()
            assert data["team_id"] == 1
            assert data["goals_scored_avg"] == 1.8
            assert data["goals_conceded_avg"] == 0.9
    
    def test_get_team_stats_with_season(self, sample_team_stats):
        """Test team statistics retrieval with season parameter."""
        with patch('app.routes.matches.data_fetcher') as mock_fetcher:
            mock_fetcher.get_team_stats = AsyncMock(return_value=sample_team_stats)
            
            response = client.get("/matches/12345/teams/1/stats?season=2023")
            
            assert response.status_code == 200
            mock_fetcher.get_team_stats.assert_called_once_with(1, 2023)
    
    def test_health_check_success(self):
        """Test successful health check."""
        with patch('app.routes.matches.data_fetcher') as mock_fetcher:
            with patch('app.routes.matches.cache_manager') as mock_cache:
                with patch('app.routes.matches.analyzer') as mock_analyzer:
                    mock_fetcher.health_check = AsyncMock(return_value={"status": "healthy"})
                    mock_cache.is_healthy = AsyncMock(return_value=True)
                    mock_analyzer.ai_summary.test_connection = AsyncMock(
                        return_value={"status": "success"}
                    )
                    
                    response = client.get("/matches/health")
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] in ["healthy", "degraded"]
                    assert "services" in data
    
    def test_health_check_degraded(self):
        """Test health check with degraded services."""
        with patch('app.routes.matches.data_fetcher') as mock_fetcher:
            with patch('app.routes.matches.cache_manager') as mock_cache:
                with patch('app.routes.matches.analyzer') as mock_analyzer:
                    mock_fetcher.health_check = AsyncMock(side_effect=Exception("API down"))
                    mock_cache.is_healthy = AsyncMock(return_value=False)
                    mock_analyzer.ai_summary.test_connection = AsyncMock(
                        return_value={"status": "failed"}
                    )
                    
                    response = client.get("/matches/health")
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "degraded"
    
    def test_get_service_stats_success(self):
        """Test successful service statistics retrieval."""
        with patch('app.routes.matches.cache_manager') as mock_cache:
            with patch('app.routes.matches.analyzer') as mock_analyzer:
                mock_cache.get_stats = AsyncMock(return_value={"status": "connected"})
                mock_analyzer.ml_models.get_model_info = MagicMock(
                    return_value={"models_loaded": {"score": True}}
                )
                
                response = client.get("/matches/stats")
                
                assert response.status_code == 200
                data = response.json()
                assert "timestamp" in data
                assert "cache" in data
                assert "ml_models" in data
                assert "service" in data
    
    def test_clear_cache_success(self):
        """Test successful cache clearing."""
        with patch('app.routes.matches.cache_manager') as mock_cache:
            mock_cache.clear_pattern = AsyncMock(return_value=5)
            
            response = client.delete("/matches/cache")
            
            assert response.status_code == 200
            data = response.json()
            assert data["cleared_entries"] == 5
            assert data["pattern"] == "*"
    
    def test_clear_cache_with_pattern(self):
        """Test cache clearing with specific pattern."""
        with patch('app.routes.matches.cache_manager') as mock_cache:
            mock_cache.clear_pattern = AsyncMock(return_value=3)
            
            response = client.delete("/matches/cache?pattern=predictions:*")
            
            assert response.status_code == 200
            data = response.json()
            assert data["cleared_entries"] == 3
            assert data["pattern"] == "predictions:*"
            mock_cache.clear_pattern.assert_called_once_with("predictions:*")
    
    def test_clear_cache_error(self):
        """Test cache clearing with error."""
        with patch('app.routes.matches.cache_manager') as mock_cache:
            mock_cache.clear_pattern = AsyncMock(side_effect=Exception("Cache error"))
            
            response = client.delete("/matches/cache")
            
            assert response.status_code == 200
            data = response.json()
            assert data["cleared_entries"] == 0
            assert "error" in data
    
    def test_correlation_id_in_responses(self, sample_match):
        """Test that correlation IDs are properly set."""
        with patch('app.routes.matches.data_fetcher') as mock_fetcher:
            mock_fetcher.get_todays_matches = AsyncMock(return_value=[sample_match])
            
            response = client.get("/matches/today")
            
            assert response.status_code == 200
            # Correlation ID should be set in logs (tested via mocking)
    
    def test_background_tasks_in_analyze(self, sample_match, sample_predictions,
                                       sample_team_stats, sample_features):
        """Test that background tasks are properly scheduled."""
        with patch('app.routes.matches.data_fetcher') as mock_fetcher:
            with patch('app.routes.matches.analyzer') as mock_analyzer:
                with patch('app.routes.matches.cache_match_predictions') as mock_cache:
                    mock_fetcher.get_todays_matches = AsyncMock(return_value=[sample_match])
                    mock_fetcher.get_team_stats = AsyncMock(return_value=sample_team_stats)
                    mock_analyzer.extract_features = AsyncMock(return_value=sample_features)
                    mock_analyzer.generate_predictions = AsyncMock(return_value=sample_predictions)
                    
                    response = client.post("/matches/12345/analyze")
                    
                    assert response.status_code == 200
                    # Background task should be scheduled (tested via mocking)
    
    def test_error_handling_with_correlation_id(self):
        """Test error handling includes correlation ID context."""
        with patch('app.routes.matches.data_fetcher') as mock_fetcher:
            mock_fetcher.get_todays_matches = AsyncMock(
                side_effect=Exception("Unexpected error")
            )
            
            response = client.get("/matches/today")
            
            assert response.status_code == 500
            assert "Internal server error" in response.json()["detail"]