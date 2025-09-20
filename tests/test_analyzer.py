"""Unit tests for analysis service."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime
import sys
import os

# Add the app directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.services.analyzer import MatchAnalyzer, AnalysisError
from app.services.fetch_data import FootballDataFetcher
from app.models.schemas import (
    Match, Team, TeamStats, PlayerStats, MatchFeatures, TeamMetrics,
    MatchStatus, PlayerPosition, PerformanceMetrics, MatchResult
)


class TestMatchAnalyzer:
    """Test cases for MatchAnalyzer."""
    
    @pytest.fixture
    def mock_data_fetcher(self):
        """Create a mock FootballDataFetcher."""
        return AsyncMock(spec=FootballDataFetcher)
    
    @pytest.fixture
    def analyzer(self, mock_data_fetcher):
        """Create a MatchAnalyzer instance for testing."""
        return MatchAnalyzer(mock_data_fetcher)
    
    @pytest.fixture
    def sample_match(self):
        """Create a sample match for testing."""
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
    def sample_home_stats(self):
        """Create sample home team statistics."""
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
            goals_scored_avg=1.75,
            goals_conceded_avg=0.85,
            yellow_cards_avg=2.1,
            corners_avg=5.5,
            home_performance=home_performance,
            away_performance=away_performance
        )
    
    @pytest.fixture
    def sample_away_stats(self):
        """Create sample away team statistics."""
        home_performance = PerformanceMetrics(
            matches_played=10,
            wins=6,
            draws=2,
            losses=2,
            goals_scored=18,
            goals_conceded=9,
            clean_sheets=5
        )
        
        away_performance = PerformanceMetrics(
            matches_played=10,
            wins=4,
            draws=4,
            losses=2,
            goals_scored=12,
            goals_conceded=8,
            clean_sheets=3
        )
        
        return TeamStats(
            team_id=2,
            goals_scored_avg=1.5,
            goals_conceded_avg=0.85,
            yellow_cards_avg=1.8,
            corners_avg=4.8,
            home_performance=home_performance,
            away_performance=away_performance
        )
    
    @pytest.fixture
    def sample_players(self):
        """Create sample player statistics."""
        return [
            PlayerStats(
                player_id=101,
                name="Harry Kane",
                position=PlayerPosition.ATTACKER,
                goals_recent=12,
                assists_recent=3,
                minutes_played=1350,
                goal_probability=0.8
            ),
            PlayerStats(
                player_id=102,
                name="Kevin De Bruyne",
                position=PlayerPosition.MIDFIELDER,
                goals_recent=5,
                assists_recent=8,
                minutes_played=1200,
                goal_probability=0.4
            ),
            PlayerStats(
                player_id=103,
                name="Virgil van Dijk",
                position=PlayerPosition.DEFENDER,
                goals_recent=2,
                assists_recent=1,
                minutes_played=1400,
                goal_probability=0.1
            )
        ]
    
    @pytest.mark.asyncio
    async def test_extract_features_success(self, analyzer, sample_match, sample_home_stats, sample_away_stats):
        """Test successful feature extraction."""
        analyzer.data_fetcher.get_head_to_head.return_value = []
        
        features = await analyzer.extract_features(sample_match, sample_home_stats, sample_away_stats)
        
        assert isinstance(features, MatchFeatures)
        assert features.home_goal_avg == 1.75
        assert features.away_goal_avg == 1.5
        assert features.home_conceded_avg == 0.85
        assert features.away_conceded_avg == 0.85
        assert 0.0 <= features.home_advantage <= 1.0
        assert -100.0 <= features.recent_form_diff <= 100.0
        assert 0.0 <= features.head_to_head_ratio <= 1.0
    
    def test_calculate_team_metrics_success(self, analyzer, sample_home_stats):
        """Test successful team metrics calculation."""
        recent_matches = []  # Empty for simplicity
        
        metrics = analyzer.calculate_team_metrics(sample_home_stats, recent_matches)
        
        assert isinstance(metrics, TeamMetrics)
        assert metrics.team_id == 1
        assert metrics.attack_strength > 0
        assert metrics.defense_strength > 0
        assert 0.0 <= metrics.form_rating <= 100.0
        assert 0.0 <= metrics.home_advantage_factor <= 2.0
        assert 0.0 <= metrics.discipline_rating <= 10.0
    
    @pytest.mark.asyncio
    async def test_generate_predictions_success(self, analyzer, sample_match, sample_home_stats, sample_away_stats):
        """Test successful prediction generation."""
        analyzer.data_fetcher.get_team_stats.side_effect = [sample_home_stats, sample_away_stats]
        analyzer.data_fetcher.get_head_to_head.return_value = []
        
        predictions = await analyzer.generate_predictions(sample_match)
        
        assert predictions.match_id == sample_match.id
        assert predictions.score_prediction is not None
        assert predictions.yellow_cards_prediction is not None
        assert predictions.corners_prediction is not None
        assert predictions.first_half_prediction is not None
        assert 0.0 <= predictions.confidence_score <= 1.0
        assert len(predictions.ai_summary) > 0
    
    def test_calculate_home_advantage(self, analyzer, sample_home_stats, sample_away_stats):
        """Test home advantage calculation."""
        advantage = analyzer._calculate_home_advantage(sample_home_stats, sample_away_stats)
        
        assert 0.0 <= advantage <= 1.0
        assert isinstance(advantage, float)
    
    def test_calculate_form_difference(self, analyzer, sample_home_stats, sample_away_stats):
        """Test form difference calculation."""
        form_diff = analyzer._calculate_form_difference(sample_home_stats, sample_away_stats)
        
        assert -100.0 <= form_diff <= 100.0
        assert isinstance(form_diff, float)
    
    @pytest.mark.asyncio
    async def test_calculate_h2h_ratio_with_matches(self, analyzer):
        """Test H2H ratio calculation with matches."""
        mock_matches = [
            MagicMock(status=MagicMock(value="FT")),
            MagicMock(status=MagicMock(value="FT")),
            MagicMock(status=MagicMock(value="FT"))
        ]
        analyzer.data_fetcher.get_head_to_head.return_value = mock_matches
        
        ratio = await analyzer._calculate_h2h_ratio(1, 2)
        
        assert 0.0 <= ratio <= 1.0
        assert isinstance(ratio, float)
    
    @pytest.mark.asyncio
    async def test_calculate_h2h_ratio_no_matches(self, analyzer):
        """Test H2H ratio calculation with no matches."""
        analyzer.data_fetcher.get_head_to_head.return_value = []
        
        ratio = await analyzer._calculate_h2h_ratio(1, 2)
        
        assert ratio == 0.5  # Default neutral ratio
    
    def test_calculate_form_rating_with_matches(self, analyzer):
        """Test form rating calculation with matches."""
        mock_matches = [
            MagicMock(status=MagicMock(value="FT"), home_team=MagicMock(id=1)),
            MagicMock(status=MagicMock(value="FT"), home_team=MagicMock(id=2)),
            MagicMock(status=MagicMock(value="NS"), home_team=MagicMock(id=1))  # Not finished
        ]
        
        rating = analyzer._calculate_form_rating(mock_matches, 1)
        
        assert 0.0 <= rating <= 100.0
        assert isinstance(rating, float)
    
    def test_calculate_form_rating_no_matches(self, analyzer):
        """Test form rating calculation with no matches."""
        rating = analyzer._calculate_form_rating([], 1)
        
        assert rating == 50.0  # Default average form
    
    def test_calculate_home_advantage_factor(self, analyzer, sample_home_stats):
        """Test home advantage factor calculation."""
        factor = analyzer._calculate_home_advantage_factor(sample_home_stats)
        
        assert 0.0 <= factor <= 2.0
        assert isinstance(factor, float)
    
    def test_analyze_player_form(self, analyzer, sample_players):
        """Test player form analysis."""
        analyzed_players = analyzer.analyze_player_form(sample_players.copy())
        
        assert len(analyzed_players) == len(sample_players)
        
        # Check that attackers have higher probability than defenders
        attacker = next(p for p in analyzed_players if p.position == PlayerPosition.ATTACKER)
        defender = next(p for p in analyzed_players if p.position == PlayerPosition.DEFENDER)
        
        assert attacker.goal_probability >= defender.goal_probability
        
        # Check that all probabilities are within bounds
        for player in analyzed_players:
            assert 0.0 <= player.goal_probability <= 1.0
    
    def test_calculate_match_importance(self, analyzer, sample_match, sample_home_stats, sample_away_stats):
        """Test match importance calculation."""
        importance = analyzer.calculate_match_importance(sample_match, sample_home_stats, sample_away_stats)
        
        assert 0.0 <= importance <= 1.0
        assert isinstance(importance, float)
    
    def test_calculate_match_importance_premier_league(self, analyzer, sample_home_stats, sample_away_stats):
        """Test match importance for Premier League."""
        home_team = Team(id=1, name="Arsenal", recent_form=[])
        away_team = Team(id=2, name="Chelsea", recent_form=[])
        
        match = Match(
            id=12345,
            home_team=home_team,
            away_team=away_team,
            kickoff_time=datetime(2024, 1, 13, 15, 0),  # Saturday
            league="Premier League",
            status=MatchStatus.NOT_STARTED
        )
        
        importance = analyzer.calculate_match_importance(match, sample_home_stats, sample_away_stats)
        
        # Premier League weekend match should have high importance
        assert importance >= 0.8
    
    @pytest.mark.asyncio
    async def test_extract_features_error_handling(self, analyzer, sample_match):
        """Test error handling in feature extraction."""
        analyzer.data_fetcher.get_head_to_head.side_effect = Exception("API error")
        
        # Should still work with fallback values
        home_stats = MagicMock()
        home_stats.goals_scored_avg = 1.5
        home_stats.goals_conceded_avg = 1.0
        home_stats.yellow_cards_avg = 2.0
        home_stats.corners_avg = 5.0
        
        away_stats = MagicMock()
        away_stats.goals_scored_avg = 1.2
        away_stats.goals_conceded_avg = 1.1
        away_stats.yellow_cards_avg = 1.8
        away_stats.corners_avg = 4.5
        
        # Mock the methods that might be called
        analyzer._calculate_home_advantage = MagicMock(return_value=0.6)
        analyzer._calculate_form_difference = MagicMock(return_value=10.0)
        
        features = await analyzer.extract_features(sample_match, home_stats, away_stats)
        
        assert isinstance(features, MatchFeatures)
        assert features.head_to_head_ratio == 0.5  # Default fallback value
    
    @pytest.mark.asyncio
    async def test_generate_predictions_error_handling(self, analyzer, sample_match):
        """Test error handling in prediction generation."""
        analyzer.data_fetcher.get_team_stats.side_effect = Exception("API error")
        
        with pytest.raises(AnalysisError):
            await analyzer.generate_predictions(sample_match)
    
    def test_calculate_team_metrics_error_handling(self, analyzer):
        """Test error handling in team metrics calculation."""
        # Invalid team stats that might cause errors
        invalid_stats = MagicMock()
        invalid_stats.team_id = 1
        invalid_stats.goals_scored_avg = None  # Invalid value
        
        with pytest.raises(AnalysisError):
            analyzer.calculate_team_metrics(invalid_stats, [])