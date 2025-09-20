"""Machine learning prediction models for football match analysis."""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from app.models.schemas import (
    MatchFeatures, TeamStats, PlayerStats, ScorePrediction, 
    PlayerPrediction, CardPrediction, CornerPrediction, 
    FirstHalfPrediction, MatchResult, PlayerPosition
)
from app.utils.logger import get_logger
from app.config import settings

logger = get_logger(__name__)


class MLModelError(Exception):
    """Custom exception for ML model errors."""
    pass


class PredictionModels:
    """Machine learning models for football match predictions."""
    
    def __init__(self):
        self.models_dir = "models"
        self.scalers_dir = "scalers"
        self._ensure_directories()
        
        # Model instances
        self.score_model = None
        self.score_scaler = None
        self.card_model = None
        self.card_scaler = None
        self.corner_model = None
        self.corner_scaler = None
        self.first_half_model = None
        self.first_half_scaler = None
        
        # Model metadata
        self.model_versions = {}
        self.model_accuracies = {}
        
        # Load existing models
        self._load_models()
    
    def _ensure_directories(self):
        """Ensure model and scaler directories exist."""
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.scalers_dir, exist_ok=True)
    
    def _load_models(self):
        """Load pre-trained models if they exist."""
        try:
            # Score prediction model
            score_model_path = os.path.join(self.models_dir, "score_model.joblib")
            score_scaler_path = os.path.join(self.scalers_dir, "score_scaler.joblib")
            
            if os.path.exists(score_model_path) and os.path.exists(score_scaler_path):
                self.score_model = joblib.load(score_model_path)
                self.score_scaler = joblib.load(score_scaler_path)
                logger.info("Loaded score prediction model")
            
            # Card prediction model
            card_model_path = os.path.join(self.models_dir, "card_model.joblib")
            card_scaler_path = os.path.join(self.scalers_dir, "card_scaler.joblib")
            
            if os.path.exists(card_model_path) and os.path.exists(card_scaler_path):
                self.card_model = joblib.load(card_model_path)
                self.card_scaler = joblib.load(card_scaler_path)
                logger.info("Loaded card prediction model")
            
            # Corner prediction model
            corner_model_path = os.path.join(self.models_dir, "corner_model.joblib")
            corner_scaler_path = os.path.join(self.scalers_dir, "corner_scaler.joblib")
            
            if os.path.exists(corner_model_path) and os.path.exists(corner_scaler_path):
                self.corner_model = joblib.load(corner_model_path)
                self.corner_scaler = joblib.load(corner_scaler_path)
                logger.info("Loaded corner prediction model")
            
            # First half prediction model
            first_half_model_path = os.path.join(self.models_dir, "first_half_model.joblib")
            first_half_scaler_path = os.path.join(self.scalers_dir, "first_half_scaler.joblib")
            
            if os.path.exists(first_half_model_path) and os.path.exists(first_half_scaler_path):
                self.first_half_model = joblib.load(first_half_model_path)
                self.first_half_scaler = joblib.load(first_half_scaler_path)
                logger.info("Loaded first half prediction model")
                
        except Exception as e:
            logger.warning(f"Error loading models: {e}")
    
    def _save_model(self, model: Any, scaler: Any, model_name: str):
        """Save a trained model and its scaler."""
        try:
            model_path = os.path.join(self.models_dir, f"{model_name}_model.joblib")
            scaler_path = os.path.join(self.scalers_dir, f"{model_name}_scaler.joblib")
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            self.model_versions[model_name] = datetime.now().isoformat()
            logger.info(f"Saved {model_name} model and scaler")
            
        except Exception as e:
            logger.error(f"Error saving {model_name} model: {e}")
    
    def predict_score(self, features: MatchFeatures) -> ScorePrediction:
        """
        Predict match score using Logistic Regression.
        
        Args:
            features: Match features for prediction
            
        Returns:
            ScorePrediction object
        """
        try:
            logger.debug("Predicting match score")
            
            if self.score_model is None or self.score_scaler is None:
                logger.info("Score model not available, training with synthetic data")
                self._train_score_model()
            
            # Prepare features
            feature_array = self._features_to_array(features)
            scaled_features = self.score_scaler.transform([feature_array])
            
            # Get prediction probabilities
            probabilities = self.score_model.predict_proba(scaled_features)[0]
            predicted_class = self.score_model.predict(scaled_features)[0]
            
            # Map class to score
            home_score, away_score = self._class_to_score(predicted_class)
            
            # Get confidence (max probability)
            confidence = float(np.max(probabilities))
            probability = float(probabilities[predicted_class])
            
            # Generate alternative scores
            alternative_scores = self._generate_alternative_scores(probabilities)
            
            return ScorePrediction(
                home_score=home_score,
                away_score=away_score,
                confidence=confidence,
                probability=probability,
                alternative_scores=alternative_scores
            )
            
        except Exception as e:
            logger.error(f"Error predicting score: {e}")
            return self._fallback_score_prediction(features)
    
    def predict_goal_scorer(self, team_stats: TeamStats, player_stats: List[PlayerStats]) -> Optional[PlayerPrediction]:
        """
        Predict most likely goal scorer using probability model.
        
        Args:
            team_stats: Team statistics
            player_stats: List of player statistics
            
        Returns:
            PlayerPrediction object or None
        """
        try:
            logger.debug("Predicting goal scorer")
            
            if not player_stats:
                return None
            
            # Calculate enhanced probabilities for each player
            enhanced_players = []
            
            for player in player_stats:
                # Base probability from recent performance
                base_prob = player.goal_probability
                
                # Position multiplier
                position_multipliers = {
                    PlayerPosition.ATTACKER: 1.5,
                    PlayerPosition.MIDFIELDER: 0.8,
                    PlayerPosition.DEFENDER: 0.3,
                    PlayerPosition.GOALKEEPER: 0.01
                }
                position_mult = position_multipliers.get(player.position, 0.5)
                
                # Minutes played factor (more minutes = higher chance)
                minutes_factor = min(player.minutes_played / 1000, 1.0)
                
                # Recent form factor (goals + assists)
                form_factor = (player.goals_recent + player.assists_recent * 0.5) / 10
                form_factor = min(form_factor, 1.0)
                
                # Team attack strength factor
                team_factor = min(team_stats.goals_scored_avg / 2.0, 1.0)
                
                # Calculate final probability
                final_prob = base_prob * position_mult * (0.5 + minutes_factor * 0.5) * (0.7 + form_factor * 0.3) * (0.8 + team_factor * 0.2)
                final_prob = min(final_prob, 0.95)  # Cap at 95%
                
                enhanced_players.append((player, final_prob))
            
            # Sort by probability and get the most likely scorer
            enhanced_players.sort(key=lambda x: x[1], reverse=True)
            best_player, best_prob = enhanced_players[0]
            
            # Calculate confidence based on probability gap
            if len(enhanced_players) > 1:
                second_best_prob = enhanced_players[1][1]
                confidence = min((best_prob - second_best_prob) + 0.5, 1.0)
            else:
                confidence = best_prob
            
            return PlayerPrediction(
                player_id=best_player.player_id,
                player_name=best_player.name,
                team_id=team_stats.team_id,
                probability=best_prob,
                confidence=confidence,
                reasoning=f"{best_player.name} has {best_player.goals_recent} goals in recent matches and plays as {best_player.position.value}"
            )
            
        except Exception as e:
            logger.error(f"Error predicting goal scorer: {e}")
            return None
    
    def predict_yellow_cards(self, home_stats: TeamStats, away_stats: TeamStats) -> CardPrediction:
        """
        Predict yellow cards using Poisson distribution.
        
        Args:
            home_stats: Home team statistics
            away_stats: Away team statistics
            
        Returns:
            CardPrediction object
        """
        try:
            logger.debug("Predicting yellow cards")
            
            if self.card_model is None or self.card_scaler is None:
                logger.info("Card model not available, using statistical approach")
                return self._statistical_card_prediction(home_stats, away_stats)
            
            # Prepare features
            features = [
                home_stats.yellow_cards_avg,
                away_stats.yellow_cards_avg,
                home_stats.goals_scored_avg,
                away_stats.goals_scored_avg,
                home_stats.goals_conceded_avg,
                away_stats.goals_conceded_avg
            ]
            
            scaled_features = self.card_scaler.transform([features])
            prediction = self.card_model.predict(scaled_features)[0]
            
            # Split prediction into home and away cards
            total_cards = max(0, round(prediction))
            home_cards = max(0, round(home_stats.yellow_cards_avg))
            away_cards = max(0, total_cards - home_cards)
            
            # Ensure total is consistent
            total_cards = home_cards + away_cards
            
            # Calculate confidence based on model certainty
            confidence = min(0.7 + (abs(prediction - total_cards) * 0.1), 1.0)
            
            return CardPrediction(
                home_team_cards=home_cards,
                away_team_cards=away_cards,
                total_cards=total_cards,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error predicting yellow cards: {e}")
            return self._statistical_card_prediction(home_stats, away_stats)
    
    def predict_corners(self, home_stats: TeamStats, away_stats: TeamStats) -> CornerPrediction:
        """
        Predict corners using linear regression.
        
        Args:
            home_stats: Home team statistics
            away_stats: Away team statistics
            
        Returns:
            CornerPrediction object
        """
        try:
            logger.debug("Predicting corners")
            
            if self.corner_model is None or self.corner_scaler is None:
                logger.info("Corner model not available, using statistical approach")
                return self._statistical_corner_prediction(home_stats, away_stats)
            
            # Prepare features
            features = [
                home_stats.corners_avg,
                away_stats.corners_avg,
                home_stats.goals_scored_avg,
                away_stats.goals_scored_avg,
                home_stats.home_performance.win_rate,
                away_stats.away_performance.win_rate
            ]
            
            scaled_features = self.corner_scaler.transform([features])
            prediction = self.corner_model.predict(scaled_features)[0]
            
            # Split prediction into home and away corners
            total_corners = max(0, round(prediction))
            home_corners = max(0, round(home_stats.corners_avg))
            away_corners = max(0, total_corners - home_corners)
            
            # Ensure total is consistent
            total_corners = home_corners + away_corners
            
            # Calculate confidence
            confidence = min(0.65 + (abs(prediction - total_corners) * 0.05), 1.0)
            
            return CornerPrediction(
                home_team_corners=home_corners,
                away_team_corners=away_corners,
                total_corners=total_corners,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error predicting corners: {e}")
            return self._statistical_corner_prediction(home_stats, away_stats)
    
    def predict_first_half(self, features: MatchFeatures) -> FirstHalfPrediction:
        """
        Predict first half result using classification.
        
        Args:
            features: Match features
            
        Returns:
            FirstHalfPrediction object
        """
        try:
            logger.debug("Predicting first half result")
            
            if self.first_half_model is None or self.first_half_scaler is None:
                logger.info("First half model not available, using statistical approach")
                return self._statistical_first_half_prediction(features)
            
            # Prepare features
            feature_array = self._features_to_array(features)
            scaled_features = self.first_half_scaler.transform([feature_array])
            
            # Get prediction
            probabilities = self.first_half_model.predict_proba(scaled_features)[0]
            predicted_class = self.first_half_model.predict(scaled_features)[0]
            
            # Map class to result
            class_to_result = {0: MatchResult.LOSS, 1: MatchResult.DRAW, 2: MatchResult.WIN}
            result = class_to_result[predicted_class]
            
            # Estimate first half scores (simplified)
            if result == MatchResult.WIN:
                home_score, away_score = 1, 0
            elif result == MatchResult.LOSS:
                home_score, away_score = 0, 1
            else:
                home_score, away_score = 0, 0
            
            confidence = float(np.max(probabilities))
            probability = float(probabilities[predicted_class])
            
            return FirstHalfPrediction(
                result=result,
                home_score=home_score,
                away_score=away_score,
                confidence=confidence,
                probability=probability
            )
            
        except Exception as e:
            logger.error(f"Error predicting first half: {e}")
            return self._statistical_first_half_prediction(features)
    
    def _train_score_model(self):
        """Train score prediction model with synthetic data."""
        try:
            logger.info("Training score prediction model with synthetic data")
            
            # Generate synthetic training data
            X, y = self._generate_synthetic_score_data(1000)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            self.score_scaler = StandardScaler()
            X_train_scaled = self.score_scaler.fit_transform(X_train)
            X_test_scaled = self.score_scaler.transform(X_test)
            
            # Train model
            self.score_model = LogisticRegression(random_state=42, max_iter=1000)
            self.score_model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.score_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            self.model_accuracies['score'] = accuracy
            
            # Save model
            self._save_model(self.score_model, self.score_scaler, "score")
            
            logger.info(f"Score model trained with accuracy: {accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Error training score model: {e}")
            raise MLModelError(f"Failed to train score model: {str(e)}")
    
    def _generate_synthetic_score_data(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic data for score prediction."""
        np.random.seed(42)
        
        # Generate features
        home_goal_avg = np.random.normal(1.5, 0.5, n_samples)
        away_goal_avg = np.random.normal(1.3, 0.5, n_samples)
        home_conceded_avg = np.random.normal(1.2, 0.4, n_samples)
        away_conceded_avg = np.random.normal(1.1, 0.4, n_samples)
        home_advantage = np.random.uniform(0.3, 0.8, n_samples)
        form_diff = np.random.normal(0, 20, n_samples)
        h2h_ratio = np.random.uniform(0.2, 0.8, n_samples)
        
        X = np.column_stack([
            home_goal_avg, away_goal_avg, home_conceded_avg, away_conceded_avg,
            home_advantage, form_diff, h2h_ratio
        ])
        
        # Generate labels (score classes)
        y = []
        for i in range(n_samples):
            # Simple logic to generate realistic score distributions
            home_expected = home_goal_avg[i] * (1 + home_advantage[i] * 0.2)
            away_expected = away_goal_avg[i]
            
            if home_expected > away_expected + 0.5:
                # Home win scenarios
                scores = [(2, 0), (2, 1), (3, 1), (1, 0)]
            elif away_expected > home_expected + 0.5:
                # Away win scenarios
                scores = [(0, 2), (1, 2), (1, 3), (0, 1)]
            else:
                # Draw scenarios
                scores = [(1, 1), (0, 0), (2, 2)]
            
            score = scores[np.random.randint(len(scores))]
            y.append(self._score_to_class(score[0], score[1]))
        
        return X, np.array(y)
    
    def _features_to_array(self, features: MatchFeatures) -> np.ndarray:
        """Convert MatchFeatures to numpy array."""
        return np.array([
            features.home_goal_avg,
            features.away_goal_avg,
            features.home_conceded_avg,
            features.away_conceded_avg,
            features.home_advantage,
            features.recent_form_diff,
            features.head_to_head_ratio
        ])
    
    def _score_to_class(self, home_score: int, away_score: int) -> int:
        """Convert score to class for classification."""
        # Map common scores to classes
        score_map = {
            (0, 0): 0, (1, 1): 1, (2, 2): 2,  # Draws
            (1, 0): 3, (2, 0): 4, (2, 1): 5, (3, 1): 6,  # Home wins
            (0, 1): 7, (0, 2): 8, (1, 2): 9, (1, 3): 10  # Away wins
        }
        return score_map.get((home_score, away_score), 11)  # Other scores
    
    def _class_to_score(self, class_id: int) -> Tuple[int, int]:
        """Convert class to score."""
        class_map = {
            0: (0, 0), 1: (1, 1), 2: (2, 2),
            3: (1, 0), 4: (2, 0), 5: (2, 1), 6: (3, 1),
            7: (0, 1), 8: (0, 2), 9: (1, 2), 10: (1, 3),
            11: (1, 1)  # Default for other scores
        }
        return class_map.get(class_id, (1, 1))
    
    def _generate_alternative_scores(self, probabilities: np.ndarray) -> List[Dict[str, Any]]:
        """Generate alternative score predictions."""
        alternatives = []
        
        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        
        for idx in top_indices:
            home_score, away_score = self._class_to_score(idx)
            alternatives.append({
                "home_score": home_score,
                "away_score": away_score,
                "probability": float(probabilities[idx])
            })
        
        return alternatives
    
    def _fallback_score_prediction(self, features: MatchFeatures) -> ScorePrediction:
        """Fallback score prediction using simple statistics."""
        try:
            home_expected = float(features.home_goal_avg) * (1 + float(features.home_advantage) * 0.2)
            away_expected = float(features.away_goal_avg)
            
            home_score = max(0, min(5, round(home_expected)))
            away_score = max(0, min(5, round(away_expected)))
        except (TypeError, ValueError):
            # Fallback to default values if features are invalid
            home_score = 1
            away_score = 1
        
        return ScorePrediction(
            home_score=home_score,
            away_score=away_score,
            confidence=0.5,
            probability=0.15,
            alternative_scores=[]
        )
    
    def _statistical_card_prediction(self, home_stats: TeamStats, away_stats: TeamStats) -> CardPrediction:
        """Statistical yellow card prediction."""
        home_cards = max(0, round(home_stats.yellow_cards_avg))
        away_cards = max(0, round(away_stats.yellow_cards_avg))
        total_cards = home_cards + away_cards
        
        return CardPrediction(
            home_team_cards=home_cards,
            away_team_cards=away_cards,
            total_cards=total_cards,
            confidence=0.6
        )
    
    def _statistical_corner_prediction(self, home_stats: TeamStats, away_stats: TeamStats) -> CornerPrediction:
        """Statistical corner prediction."""
        home_corners = max(0, round(home_stats.corners_avg))
        away_corners = max(0, round(away_stats.corners_avg))
        total_corners = home_corners + away_corners
        
        return CornerPrediction(
            home_team_corners=home_corners,
            away_team_corners=away_corners,
            total_corners=total_corners,
            confidence=0.55
        )
    
    def _statistical_first_half_prediction(self, features: MatchFeatures) -> FirstHalfPrediction:
        """Statistical first half prediction."""
        home_expected = features.home_goal_avg * 0.6  # 60% of goals in first half
        away_expected = features.away_goal_avg * 0.6
        
        home_expected *= (1 + features.home_advantage * 0.1)
        
        if home_expected > away_expected + 0.3:
            result = MatchResult.WIN
            home_score, away_score = 1, 0
        elif away_expected > home_expected + 0.3:
            result = MatchResult.LOSS
            home_score, away_score = 0, 1
        else:
            result = MatchResult.DRAW
            home_score, away_score = 0, 0
        
        return FirstHalfPrediction(
            result=result,
            home_score=home_score,
            away_score=away_score,
            confidence=0.5,
            probability=0.33
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "models_loaded": {
                "score": self.score_model is not None,
                "card": self.card_model is not None,
                "corner": self.corner_model is not None,
                "first_half": self.first_half_model is not None
            },
            "model_versions": self.model_versions,
            "model_accuracies": self.model_accuracies,
            "confidence_threshold": settings.model_confidence_threshold
        }