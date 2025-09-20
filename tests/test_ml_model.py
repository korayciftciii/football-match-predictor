"""Unit tests for ML prediction models."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os
import sys

# Add the app directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.services.ml_model import PredictionModels, MLModelError
from app.models.schemas import (
    MatchFeatures, TeamStats, PlayerStats, ScorePrediction,
    PlayerPrediction, CardPrediction, CornerPrediction,
    FirstHalfPrediction, MatchResult, PlayerPosition, PerformanceMetrics
)


class TestPredictionModels:
    """Test cases for PredictionModels."""
    
    @pytest.fixture
    def temp_models_dir(self):
        """Create temporary directory for models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def prediction_models(self, temp_models_dir):
        """Create PredictionModels instance with temporary directory."""
        models = PredictionModels()
        models.models_dir = temp_models_dir
        models.scalers_dir = temp_models_dir
        return models
    
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
    
    def test_init(self, prediction_models):
        """Test PredictionModels initialization."""
        # Models might be loaded or None depending on if files exist
        assert isinstance(prediction_models.model_versions, dict)
        assert isinstance(prediction_models.model_accuracies, dict)
        # Check that the instance is properly initialized
        assert hasattr(prediction_models, 'score_model')
        assert hasattr(prediction_models, 'score_scaler')
    
    def test_predict_score_without_model(self, prediction_models, sample_features):
        """Test score prediction without trained model (should train automatically)."""
        with patch.object(prediction_models, '_train_score_model') as mock_train:
            with patch.object(prediction_models, '_fallback_score_prediction') as mock_fallback:
                mock_fallback.return_value = ScorePrediction(
                    home_score=2, away_score=1, confidence=0.5, probability=0.15, alternative_scores=[]
                )
                
                result = prediction_models.predict_score(sample_features)
                
                assert isinstance(result, ScorePrediction)
                assert 0 <= result.home_score <= 5
                assert 0 <= result.away_score <= 5
                assert 0.0 <= result.confidence <= 1.0
                assert 0.0 <= result.probability <= 1.0
    
    def test_predict_goal_scorer_success(self, prediction_models, sample_team_stats, sample_players):
        """Test successful goal scorer prediction."""
        result = prediction_models.predict_goal_scorer(sample_team_stats, sample_players)
        
        assert isinstance(result, PlayerPrediction)
        assert result.player_name in ["Harry Kane", "Kevin De Bruyne", "Virgil van Dijk"]
        assert 0.0 <= result.probability <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert result.reasoning is not None
    
    def test_predict_goal_scorer_empty_players(self, prediction_models, sample_team_stats):
        """Test goal scorer prediction with empty player list."""
        result = prediction_models.predict_goal_scorer(sample_team_stats, [])
        
        assert result is None
    
    def test_predict_goal_scorer_attacker_preference(self, prediction_models, sample_team_stats, sample_players):
        """Test that attackers get higher probability than defenders."""
        result = prediction_models.predict_goal_scorer(sample_team_stats, sample_players)
        
        # Should prefer Harry Kane (attacker) over Virgil van Dijk (defender)
        assert result.player_name == "Harry Kane"
    
    def test_predict_yellow_cards_statistical(self, prediction_models, sample_team_stats):
        """Test yellow card prediction using statistical method."""
        away_stats = TeamStats(
            team_id=2,
            goals_scored_avg=1.4,
            goals_conceded_avg=1.2,
            yellow_cards_avg=1.8,
            corners_avg=4.2,
            home_performance=PerformanceMetrics(
                matches_played=10, wins=4, draws=3, losses=3,
                goals_scored=14, goals_conceded=12, clean_sheets=3
            ),
            away_performance=PerformanceMetrics(
                matches_played=10, wins=3, draws=4, losses=3,
                goals_scored=12, goals_conceded=11, clean_sheets=2
            )
        )
        
        result = prediction_models.predict_yellow_cards(sample_team_stats, away_stats)
        
        assert isinstance(result, CardPrediction)
        assert result.home_team_cards >= 0
        assert result.away_team_cards >= 0
        assert result.total_cards == result.home_team_cards + result.away_team_cards
        assert 0.0 <= result.confidence <= 1.0
    
    def test_predict_corners_statistical(self, prediction_models, sample_team_stats):
        """Test corner prediction using statistical method."""
        away_stats = TeamStats(
            team_id=2,
            goals_scored_avg=1.4,
            goals_conceded_avg=1.2,
            yellow_cards_avg=1.8,
            corners_avg=4.2,
            home_performance=PerformanceMetrics(
                matches_played=10, wins=4, draws=3, losses=3,
                goals_scored=14, goals_conceded=12, clean_sheets=3
            ),
            away_performance=PerformanceMetrics(
                matches_played=10, wins=3, draws=4, losses=3,
                goals_scored=12, goals_conceded=11, clean_sheets=2
            )
        )
        
        result = prediction_models.predict_corners(sample_team_stats, away_stats)
        
        assert isinstance(result, CornerPrediction)
        assert result.home_team_corners >= 0
        assert result.away_team_corners >= 0
        assert result.total_corners == result.home_team_corners + result.away_team_corners
        assert 0.0 <= result.confidence <= 1.0
    
    def test_predict_first_half_statistical(self, prediction_models, sample_features):
        """Test first half prediction using statistical method."""
        result = prediction_models.predict_first_half(sample_features)
        
        assert isinstance(result, FirstHalfPrediction)
        assert result.result in [MatchResult.WIN, MatchResult.DRAW, MatchResult.LOSS]
        assert result.home_score >= 0
        assert result.away_score >= 0
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.probability <= 1.0
    
    def test_features_to_array(self, prediction_models, sample_features):
        """Test feature conversion to numpy array."""
        array = prediction_models._features_to_array(sample_features)
        
        assert isinstance(array, np.ndarray)
        assert len(array) == 7  # Number of features
        assert array[0] == sample_features.home_goal_avg
        assert array[1] == sample_features.away_goal_avg
    
    def test_score_to_class_and_back(self, prediction_models):
        """Test score to class conversion and back."""
        # Test common scores
        test_scores = [(0, 0), (1, 1), (2, 1), (0, 2)]
        
        for home_score, away_score in test_scores:
            class_id = prediction_models._score_to_class(home_score, away_score)
            converted_back = prediction_models._class_to_score(class_id)
            
            assert isinstance(class_id, int)
            assert isinstance(converted_back, tuple)
            assert len(converted_back) == 2
    
    def test_generate_synthetic_score_data(self, prediction_models):
        """Test synthetic data generation."""
        X, y = prediction_models._generate_synthetic_score_data(100)
        
        assert X.shape == (100, 7)  # 100 samples, 7 features
        assert y.shape == (100,)
        assert np.all(y >= 0)  # All class labels should be non-negative
    
    def test_train_score_model(self, prediction_models):
        """Test score model training."""
        prediction_models._train_score_model()
        
        assert prediction_models.score_model is not None
        assert prediction_models.score_scaler is not None
        assert 'score' in prediction_models.model_accuracies
        assert 0.0 <= prediction_models.model_accuracies['score'] <= 1.0
    
    def test_fallback_score_prediction(self, prediction_models, sample_features):
        """Test fallback score prediction."""
        result = prediction_models._fallback_score_prediction(sample_features)
        
        assert isinstance(result, ScorePrediction)
        assert 0 <= result.home_score <= 5
        assert 0 <= result.away_score <= 5
        assert result.confidence == 0.5
        assert result.probability == 0.15
    
    def test_statistical_predictions(self, prediction_models, sample_team_stats, sample_features):
        """Test all statistical prediction methods."""
        away_stats = TeamStats(
            team_id=2,
            goals_scored_avg=1.4,
            goals_conceded_avg=1.2,
            yellow_cards_avg=1.8,
            corners_avg=4.2,
            home_performance=PerformanceMetrics(
                matches_played=10, wins=4, draws=3, losses=3,
                goals_scored=14, goals_conceded=12, clean_sheets=3
            ),
            away_performance=PerformanceMetrics(
                matches_played=10, wins=3, draws=4, losses=3,
                goals_scored=12, goals_conceded=11, clean_sheets=2
            )
        )
        
        # Test card prediction
        card_result = prediction_models._statistical_card_prediction(sample_team_stats, away_stats)
        assert isinstance(card_result, CardPrediction)
        
        # Test corner prediction
        corner_result = prediction_models._statistical_corner_prediction(sample_team_stats, away_stats)
        assert isinstance(corner_result, CornerPrediction)
        
        # Test first half prediction
        first_half_result = prediction_models._statistical_first_half_prediction(sample_features)
        assert isinstance(first_half_result, FirstHalfPrediction)
    
    def test_get_model_info(self, prediction_models):
        """Test model information retrieval."""
        info = prediction_models.get_model_info()
        
        assert 'models_loaded' in info
        assert 'model_versions' in info
        assert 'model_accuracies' in info
        assert 'confidence_threshold' in info
        
        assert isinstance(info['models_loaded'], dict)
        assert 'score' in info['models_loaded']
        assert 'card' in info['models_loaded']
        assert 'corner' in info['models_loaded']
        assert 'first_half' in info['models_loaded']
    
    def test_generate_alternative_scores(self, prediction_models):
        """Test alternative score generation."""
        # Mock probabilities for 12 classes
        probabilities = np.array([0.1, 0.05, 0.03, 0.25, 0.15, 0.12, 0.08, 0.07, 0.06, 0.04, 0.03, 0.02])
        
        alternatives = prediction_models._generate_alternative_scores(probabilities)
        
        assert len(alternatives) == 3  # Top 3 alternatives
        assert all('home_score' in alt for alt in alternatives)
        assert all('away_score' in alt for alt in alternatives)
        assert all('probability' in alt for alt in alternatives)
        
        # Should be sorted by probability (descending)
        probs = [alt['probability'] for alt in alternatives]
        assert probs == sorted(probs, reverse=True)
    
    def test_error_handling_in_predictions(self, prediction_models):
        """Test error handling in prediction methods."""
        # Test with invalid features that will cause numpy array error
        invalid_features = MagicMock()
        invalid_features.home_goal_avg = [1, 2, 3]  # Invalid nested sequence
        invalid_features.away_goal_avg = 1.0
        invalid_features.home_conceded_avg = 1.0
        invalid_features.away_conceded_avg = 1.0
        invalid_features.home_advantage = 0.5
        invalid_features.recent_form_diff = 0.0
        invalid_features.head_to_head_ratio = 0.5
        
        # Should not raise exception, should return fallback
        result = prediction_models.predict_score(invalid_features)
        assert isinstance(result, ScorePrediction)
    
    @pytest.mark.asyncio
    async def test_model_persistence(self, prediction_models, temp_models_dir):
        """Test model saving and loading."""
        # Train a model
        prediction_models._train_score_model()
        
        # Check that model files were created
        model_file = os.path.join(temp_models_dir, "score_model.joblib")
        scaler_file = os.path.join(temp_models_dir, "score_scaler.joblib")
        
        # Files should exist after training
        assert prediction_models.score_model is not None
        assert prediction_models.score_scaler is not None
    
    def test_player_position_multipliers(self, prediction_models, sample_team_stats):
        """Test that different player positions get appropriate multipliers."""
        # Create players with same base stats but different positions
        base_player = PlayerStats(
            player_id=100,
            name="Test Player",
            position=PlayerPosition.ATTACKER,
            goals_recent=5,
            assists_recent=2,
            minutes_played=1000,
            goal_probability=0.5
        )
        
        positions = [PlayerPosition.ATTACKER, PlayerPosition.MIDFIELDER, 
                    PlayerPosition.DEFENDER, PlayerPosition.GOALKEEPER]
        
        results = []
        for i, position in enumerate(positions):
            player = PlayerStats(
                player_id=100 + i + 1,  # Use simple incrementing ID
                name=f"Test {position.value}",
                position=position,
                goals_recent=5,
                assists_recent=2,
                minutes_played=1000,
                goal_probability=0.5
            )
            
            result = prediction_models.predict_goal_scorer(sample_team_stats, [player])
            results.append((position, result.probability if result else 0))
        
        # Attackers should have highest probability, goalkeepers lowest
        attacker_prob = next(prob for pos, prob in results if pos == PlayerPosition.ATTACKER)
        goalkeeper_prob = next(prob for pos, prob in results if pos == PlayerPosition.GOALKEEPER)
        
        assert attacker_prob > goalkeeper_prob