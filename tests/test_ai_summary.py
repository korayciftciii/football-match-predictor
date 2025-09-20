"""Unit tests for AI summary service."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime
import sys
import os

# Add the app directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.services.ai_summary import AISummaryGenerator, AISummaryError
from app.models.schemas import (
    Match, Team, MatchPredictions, ScorePrediction, PlayerPrediction,
    CardPrediction, CornerPrediction, FirstHalfPrediction,
    TeamStats, MatchFeatures, MatchStatus, MatchResult,
    PlayerPosition, PerformanceMetrics
)


class TestAISummaryGenerator:
    """Test cases for AISummaryGenerator."""
    
    @pytest.fixture
    def ai_generator(self):
        """Create AISummaryGenerator instance for testing."""
        return AISummaryGenerator()
    
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
        
        goal_scorer_prediction = PlayerPrediction(
            player_id=101,
            player_name="Harry Kane",
            team_id=1,
            probability=0.65,
            confidence=0.8,
            reasoning="Recent form and position"
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
            goal_scorer_prediction=goal_scorer_prediction,
            yellow_cards_prediction=card_prediction,
            corners_prediction=corner_prediction,
            first_half_prediction=first_half_prediction,
            ai_summary="Test summary",
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
    
    def test_init(self, ai_generator):
        """Test AISummaryGenerator initialization."""
        assert ai_generator.client is not None
        assert len(ai_generator.available_models) > 0
        assert ai_generator.default_model in ai_generator.available_models
        assert ai_generator.max_retries > 0
        assert ai_generator.timeout > 0
    
    def test_prepare_context(self, ai_generator, sample_match, sample_predictions, 
                           sample_team_stats, sample_features):
        """Test context preparation for AI model."""
        away_stats = sample_team_stats  # Use same stats for simplicity
        
        context = ai_generator._prepare_context(
            sample_match, sample_predictions, sample_team_stats, away_stats, sample_features
        )
        
        assert "match_info" in context
        assert "predictions" in context
        assert "team_stats" in context
        assert "analysis" in context
        
        # Check match info
        assert context["match_info"]["home_team"] == "Arsenal"
        assert context["match_info"]["away_team"] == "Chelsea"
        assert context["match_info"]["league"] == "Premier League"
        
        # Check predictions
        assert context["predictions"]["score"] == "2-1"
        assert context["predictions"]["goal_scorer"] == "Harry Kane"
        assert context["predictions"]["yellow_cards"] == 5
        assert context["predictions"]["corners"] == 10
        
        # Check team stats
        assert context["team_stats"]["home"]["name"] == "Arsenal"
        assert context["team_stats"]["away"]["name"] == "Chelsea"
        
        # Check analysis
        assert "home_advantage" in context["analysis"]
        assert "form_difference" in context["analysis"]
        assert "h2h_ratio" in context["analysis"]
    
    def test_create_turkish_prompt(self, ai_generator, sample_match, sample_predictions,
                                 sample_team_stats, sample_features):
        """Test Turkish prompt creation."""
        away_stats = sample_team_stats
        context = ai_generator._prepare_context(
            sample_match, sample_predictions, sample_team_stats, away_stats, sample_features
        )
        
        prompt = ai_generator._create_turkish_prompt(context)
        
        assert isinstance(prompt, str)
        assert len(prompt) > 100
        assert "Arsenal" in prompt
        assert "Chelsea" in prompt
        assert "2-1" in prompt
        assert "Harry Kane" in prompt
        assert "Premier League" in prompt
        
        # Check for Turkish words
        turkish_words = ["maç", "tahmin", "skor", "gol", "takım"]
        assert any(word in prompt.lower() for word in turkish_words)
    
    def test_validate_summary_valid(self, ai_generator):
        """Test summary validation with valid summary."""
        valid_summary = "Arsenal - Chelsea maçında 2-1 skoruyla Arsenal'ın galip geleceği tahmin ediliyor. Harry Kane'ın gol atma ihtimali yüksek."
        
        result = ai_generator._validate_summary(valid_summary)
        
        assert result is True
    
    def test_validate_summary_invalid_too_short(self, ai_generator):
        """Test summary validation with too short summary."""
        short_summary = "Kısa özet"
        
        result = ai_generator._validate_summary(short_summary)
        
        assert result is False
    
    def test_validate_summary_invalid_no_turkish(self, ai_generator):
        """Test summary validation with no Turkish content."""
        english_summary = "Arsenal will win against Chelsea with a score of 2-1. Harry Kane is likely to score."
        
        result = ai_generator._validate_summary(english_summary)
        
        assert result is False
    
    def test_validate_summary_invalid_too_long(self, ai_generator):
        """Test summary validation with too long summary."""
        long_summary = "A" * 1001  # Too long
        
        result = ai_generator._validate_summary(long_summary)
        
        assert result is False
    
    def test_generate_fallback_summary_home_win(self, ai_generator, sample_match, 
                                              sample_predictions, sample_team_stats):
        """Test fallback summary generation for home win."""
        away_stats = sample_team_stats
        
        summary = ai_generator._generate_fallback_summary(
            sample_match, sample_predictions, sample_team_stats, away_stats
        )
        
        assert isinstance(summary, str)
        assert len(summary) > 50
        assert "Arsenal" in summary
        assert "Chelsea" in summary
        assert "2-1" in summary
        assert "Harry Kane" in summary
        assert "galip" in summary.lower()
    
    def test_generate_fallback_summary_draw(self, ai_generator, sample_match, 
                                          sample_team_stats):
        """Test fallback summary generation for draw."""
        # Create draw prediction
        draw_score = ScorePrediction(
            home_score=1, away_score=1, confidence=0.6, probability=0.25, alternative_scores=[]
        )
        
        draw_predictions = MatchPredictions(
            match_id=12345,
            score_prediction=draw_score,
            goal_scorer_prediction=None,
            yellow_cards_prediction=CardPrediction(
                home_team_cards=2, away_team_cards=2, total_cards=4, confidence=0.7
            ),
            corners_prediction=CornerPrediction(
                home_team_corners=5, away_team_corners=5, total_corners=10, confidence=0.6
            ),
            first_half_prediction=FirstHalfPrediction(
                result=MatchResult.DRAW, home_score=0, away_score=0, confidence=0.5, probability=0.33
            ),
            ai_summary="Test summary with enough characters",
            confidence_score=0.6
        )
        
        away_stats = sample_team_stats
        
        summary = ai_generator._generate_fallback_summary(
            sample_match, draw_predictions, sample_team_stats, away_stats
        )
        
        assert "beraberlik" in summary.lower()
        assert "1-1" in summary
    
    @pytest.mark.asyncio
    async def test_generate_match_summary_with_ai_success(self, ai_generator, sample_match,
                                                        sample_predictions, sample_team_stats,
                                                        sample_features):
        """Test successful AI summary generation."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Arsenal - Chelsea maçında 2-1 skoruyla Arsenal'ın galip geleceği tahmin ediliyor. Harry Kane'ın gol atma ihtimali yüksek."
        
        with patch.object(ai_generator.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            away_stats = sample_team_stats
            summary = await ai_generator.generate_match_summary(
                sample_match, sample_predictions, sample_team_stats, away_stats, sample_features
            )
            
            assert isinstance(summary, str)
            assert len(summary) > 50
            assert "Arsenal" in summary
            assert "Chelsea" in summary
            mock_create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_match_summary_ai_failure_fallback(self, ai_generator, sample_match,
                                                            sample_predictions, sample_team_stats,
                                                            sample_features):
        """Test AI summary generation with failure and fallback."""
        with patch.object(ai_generator.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("API Error")
            
            away_stats = sample_team_stats
            summary = await ai_generator.generate_match_summary(
                sample_match, sample_predictions, sample_team_stats, away_stats, sample_features
            )
            
            # Should return fallback summary
            assert isinstance(summary, str)
            assert len(summary) > 50
            assert "Arsenal" in summary
            assert "Chelsea" in summary
    
    @pytest.mark.asyncio
    async def test_generate_quick_summary_score(self, ai_generator):
        """Test quick summary generation for score."""
        data = {
            "score": "2-1",
            "confidence": 0.75,
            "home_goals_avg": 1.8,
            "away_goals_avg": 1.4
        }
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Skor tahmini 2-1 (%75 güvenle)."
        
        with patch.object(ai_generator.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            summary = await ai_generator.generate_quick_summary("score", data)
            
            assert isinstance(summary, str)
            assert "2-1" in summary
    
    @pytest.mark.asyncio
    async def test_generate_quick_summary_goal_scorer(self, ai_generator):
        """Test quick summary generation for goal scorer."""
        data = {
            "player_name": "Harry Kane",
            "probability": 0.65
        }
        
        summary = await ai_generator.generate_quick_summary("goal_scorer", data)
        
        assert isinstance(summary, str)
        assert "Harry Kane" in summary
        assert "65" in summary
    
    @pytest.mark.asyncio
    async def test_generate_quick_summary_unknown_type(self, ai_generator):
        """Test quick summary generation for unknown type."""
        summary = await ai_generator.generate_quick_summary("unknown_type", {})
        
        assert summary == "Tahmin analizi tamamlandı."
    
    @pytest.mark.asyncio
    async def test_test_connection_success(self, ai_generator):
        """Test successful connection test."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Bağlantı başarılı"
        
        with patch.object(ai_generator.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            result = await ai_generator.test_connection()
            
            assert result["status"] == "success"
            assert result["model"] == ai_generator.default_model
            assert "Bağlantı başarılı" in result["response"]
    
    @pytest.mark.asyncio
    async def test_test_connection_failure(self, ai_generator):
        """Test connection test failure."""
        with patch.object(ai_generator.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("Connection failed")
            
            result = await ai_generator.test_connection()
            
            assert result["status"] == "failed"
            assert "error" in result
    
    def test_format_prediction_explanation_score(self, ai_generator):
        """Test prediction explanation formatting for score."""
        data = {"score": "2-1", "confidence": 0.75}
        
        explanation = ai_generator.format_prediction_explanation("score", data)
        
        assert "Skor Tahmini: 2-1" in explanation
        assert "75" in explanation
    
    def test_format_prediction_explanation_goal_scorer(self, ai_generator):
        """Test prediction explanation formatting for goal scorer."""
        data = {"player_name": "Harry Kane", "probability": 0.65}
        
        explanation = ai_generator.format_prediction_explanation("goal_scorer", data)
        
        assert "Gol Atacak Oyuncu: Harry Kane" in explanation
        assert "65" in explanation
    
    def test_format_prediction_explanation_cards(self, ai_generator):
        """Test prediction explanation formatting for cards."""
        data = {"total_cards": 5, "home_cards": 2, "away_cards": 3}
        
        explanation = ai_generator.format_prediction_explanation("cards", data)
        
        assert "Sarı Kart: 5 toplam" in explanation
        assert "2 ev sahibi" in explanation
        assert "3 deplasman" in explanation
    
    def test_format_prediction_explanation_corners(self, ai_generator):
        """Test prediction explanation formatting for corners."""
        data = {"total_corners": 10, "home_corners": 6, "away_corners": 4}
        
        explanation = ai_generator.format_prediction_explanation("corners", data)
        
        assert "Korner: 10 toplam" in explanation
        assert "6 ev sahibi" in explanation
        assert "4 deplasman" in explanation
    
    def test_format_prediction_explanation_first_half(self, ai_generator):
        """Test prediction explanation formatting for first half."""
        data = {"result": "WIN", "score": "1-0"}
        
        explanation = ai_generator.format_prediction_explanation("first_half", data)
        
        assert "İlk Yarı: 1-0" in explanation
        assert "WIN" in explanation
    
    def test_format_prediction_explanation_unknown(self, ai_generator):
        """Test prediction explanation formatting for unknown type."""
        explanation = ai_generator.format_prediction_explanation("unknown", {})
        
        assert "unknown tahmini" in explanation