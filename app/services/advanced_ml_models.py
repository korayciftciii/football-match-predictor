"""
Advanced ML Models for Football Match Prediction
Integrates with Hugging Face models and implements sophisticated prediction algorithms
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import asyncio
import aiohttp
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime, timedelta

from app.models.schemas import MatchData, ScorePrediction, PlayerPrediction, MatchPredictions
from app.utils.logger import get_logger
from app.utils.cache import CacheManager

logger = get_logger(__name__)

class AdvancedMLPredictor:
    """Advanced ML predictor using multiple algorithms and external models."""
    
    def __init__(self):
        self.cache = CacheManager()
        self.models_loaded = False
        self.scaler = StandardScaler()
        
        # Model paths
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize models
        self.score_model = None
        self.cards_model = None
        self.corners_model = None
        self.goal_scorer_model = None
        
        # Hugging Face integration
        self.hf_api_url = "https://api-inference.huggingface.co/models"
        self.hf_headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY', '')}"}
    
    async def initialize_models(self):
        """Initialize and load all ML models."""
        try:
            await self._load_or_train_models()
            self.models_loaded = True
            logger.info("Advanced ML models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    async def _load_or_train_models(self):
        """Load existing models or train new ones."""
        # Score prediction model (Random Forest)
        score_model_path = os.path.join(self.model_dir, "score_model.joblib")
        if os.path.exists(score_model_path):
            self.score_model = joblib.load(score_model_path)
        else:
            await self._train_score_model()
        
        # Cards prediction model (Poisson Regression)
        cards_model_path = os.path.join(self.model_dir, "cards_model.joblib")
        if os.path.exists(cards_model_path):
            self.cards_model = joblib.load(cards_model_path)
        else:
            await self._train_cards_model()
        
        # Corners prediction model (Gradient Boosting)
        corners_model_path = os.path.join(self.model_dir, "corners_model.joblib")
        if os.path.exists(corners_model_path):
            self.corners_model = joblib.load(corners_model_path)
        else:
            await self._train_corners_model()
    
    async def _train_score_model(self):
        """Train score prediction model with synthetic data."""
        # Generate synthetic training data based on real football statistics
        X, y_home, y_away = self._generate_score_training_data()
        
        # Train separate models for home and away scores
        self.score_model = {
            'home': RandomForestRegressor(n_estimators=100, random_state=42),
            'away': RandomForestRegressor(n_estimators=100, random_state=42),
            'scaler': StandardScaler()
        }
        
        X_scaled = self.score_model['scaler'].fit_transform(X)
        self.score_model['home'].fit(X_scaled, y_home)
        self.score_model['away'].fit(X_scaled, y_away)
        
        # Save model
        joblib.dump(self.score_model, os.path.join(self.model_dir, "score_model.joblib"))
        logger.info("Score prediction model trained and saved")
    
    async def _train_cards_model(self):
        """Train cards prediction model."""
        X, y = self._generate_cards_training_data()
        
        self.cards_model = {
            'model': PoissonRegressor(alpha=0.1),
            'scaler': StandardScaler()
        }
        
        X_scaled = self.cards_model['scaler'].fit_transform(X)
        self.cards_model['model'].fit(X_scaled, y)
        
        joblib.dump(self.cards_model, os.path.join(self.model_dir, "cards_model.joblib"))
        logger.info("Cards prediction model trained and saved")
    
    async def _train_corners_model(self):
        """Train corners prediction model."""
        X, y = self._generate_corners_training_data()
        
        self.corners_model = {
            'model': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'scaler': StandardScaler()
        }
        
        X_scaled = self.corners_model['scaler'].fit_transform(X)
        self.corners_model['model'].fit(X_scaled, y)
        
        joblib.dump(self.corners_model, os.path.join(self.model_dir, "corners_model.joblib"))
        logger.info("Corners prediction model trained and saved")
    
    def _generate_score_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic training data for score prediction."""
        np.random.seed(42)
        n_samples = 5000
        
        # Features: [home_attack, home_defense, away_attack, away_defense, 
        #           home_form, away_form, head_to_head, league_position_diff]
        X = np.random.rand(n_samples, 8)
        
        # Realistic feature scaling
        X[:, 0] *= 3.0  # home_attack (0-3)
        X[:, 1] *= 2.0  # home_defense (0-2)
        X[:, 2] *= 3.0  # away_attack (0-3)
        X[:, 3] *= 2.0  # away_defense (0-2)
        X[:, 4] = X[:, 4] * 2 - 1  # home_form (-1 to 1)
        X[:, 5] = X[:, 5] * 2 - 1  # away_form (-1 to 1)
        X[:, 6] = X[:, 6] * 2 - 1  # head_to_head (-1 to 1)
        X[:, 7] = (X[:, 7] - 0.5) * 20  # position_diff (-10 to 10)
        
        # Generate realistic scores based on features
        home_expected = (X[:, 0] - X[:, 3] + X[:, 4] * 0.5 + 
                        np.where(X[:, 6] > 0, 0.3, -0.3) + 0.3)  # home advantage
        away_expected = (X[:, 2] - X[:, 1] + X[:, 5] * 0.5 + 
                        np.where(X[:, 6] < 0, 0.3, -0.3))
        
        # Add noise and ensure non-negative
        home_expected = np.maximum(0, home_expected + np.random.normal(0, 0.3, n_samples))
        away_expected = np.maximum(0, away_expected + np.random.normal(0, 0.3, n_samples))
        
        # Convert to discrete scores (Poisson-like distribution)
        y_home = np.random.poisson(home_expected)
        y_away = np.random.poisson(away_expected)
        
        # Cap at reasonable maximum
        y_home = np.minimum(y_home, 6)
        y_away = np.minimum(y_away, 6)
        
        return X, y_home, y_away
    
    def _generate_cards_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data for cards prediction."""
        np.random.seed(42)
        n_samples = 3000
        
        # Features: [team_aggression, referee_strictness, match_importance, rivalry]
        X = np.random.rand(n_samples, 4)
        X[:, 0] *= 5.0  # team_aggression (0-5)
        X[:, 1] *= 3.0  # referee_strictness (0-3)
        X[:, 2] *= 2.0  # match_importance (0-2)
        X[:, 3] *= 1.0  # rivalry (0-1)
        
        # Generate cards based on features
        expected_cards = (X[:, 0] * 0.8 + X[:, 1] * 1.2 + 
                         X[:, 2] * 0.5 + X[:, 3] * 2.0 + 1.0)
        
        y = np.random.poisson(expected_cards)
        y = np.minimum(y, 12)  # Cap at reasonable maximum
        
        return X, y
    
    def _generate_corners_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data for corners prediction."""
        np.random.seed(42)
        n_samples = 3000
        
        # Features: [attacking_style, possession_style, defensive_style, match_tempo]
        X = np.random.rand(n_samples, 4)
        X[:, 0] *= 3.0  # attacking_style (0-3)
        X[:, 1] *= 2.0  # possession_style (0-2)
        X[:, 2] *= 2.0  # defensive_style (0-2)
        X[:, 3] *= 2.0  # match_tempo (0-2)
        
        # Generate corners (classify into ranges: 0-6, 7-9, 10-12, 13+)
        expected_corners = (X[:, 0] * 2.5 + X[:, 1] * 1.5 + 
                           X[:, 2] * 0.5 + X[:, 3] * 1.0 + 4.0)
        
        # Convert to classes
        y = np.zeros(n_samples, dtype=int)
        y[expected_corners <= 6] = 0
        y[(expected_corners > 6) & (expected_corners <= 9)] = 1
        y[(expected_corners > 9) & (expected_corners <= 12)] = 2
        y[expected_corners > 12] = 3
        
        return X, y
    
    async def predict_score_advanced(self, match_data: MatchData) -> ScorePrediction:
        """Advanced score prediction using ML model."""
        if not self.models_loaded:
            await self.initialize_models()
        
        try:
            # Extract features
            features = await self._extract_score_features(match_data)
            features_scaled = self.score_model['scaler'].transform([features])
            
            # Predict scores
            home_score_pred = self.score_model['home'].predict(features_scaled)[0]
            away_score_pred = self.score_model['away'].predict(features_scaled)[0]
            
            # Round to integers and ensure non-negative
            home_score = max(0, round(home_score_pred))
            away_score = max(0, round(away_score_pred))
            
            # Calculate confidence based on feature certainty
            confidence = self._calculate_score_confidence(features)
            
            # Calculate win probability
            if home_score > away_score:
                probability = 0.6 + confidence * 0.2
            elif away_score > home_score:
                probability = 0.4 + confidence * 0.2
            else:
                probability = 0.5
            
            return ScorePrediction(
                home_score=home_score,
                away_score=away_score,
                confidence=confidence,
                probability=probability
            )
            
        except Exception as e:
            logger.error(f"Advanced score prediction failed: {e}")
            # Fallback to simple prediction
            return await self._fallback_score_prediction(match_data)
    
    async def predict_cards_advanced(self, match_data: MatchData) -> Dict:
        """Advanced cards prediction using ML model."""
        if not self.models_loaded:
            await self.initialize_models()
        
        try:
            features = await self._extract_cards_features(match_data)
            features_scaled = self.cards_model['scaler'].transform([features])
            
            total_cards = self.cards_model['model'].predict(features_scaled)[0]
            total_cards = max(1, round(total_cards))
            
            # Distribute between teams (slightly favor away team for more cards)
            home_cards = max(1, round(total_cards * 0.45))
            away_cards = max(1, total_cards - home_cards)
            
            confidence = min(0.9, 0.5 + abs(total_cards - 4) * 0.1)
            
            return {
                "home_team_cards": home_cards,
                "away_team_cards": away_cards,
                "total_cards": total_cards,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Advanced cards prediction failed: {e}")
            return await self._fallback_cards_prediction(match_data)
    
    async def predict_corners_advanced(self, match_data: MatchData) -> Dict:
        """Advanced corners prediction using ML model."""
        if not self.models_loaded:
            await self.initialize_models()
        
        try:
            features = await self._extract_corners_features(match_data)
            features_scaled = self.corners_model['scaler'].transform([features])
            
            corner_class = self.corners_model['model'].predict(features_scaled)[0]
            
            # Convert class to actual corner count
            corner_ranges = [(3, 6), (7, 9), (10, 12), (13, 16)]
            min_corners, max_corners = corner_ranges[corner_class]
            total_corners = np.random.randint(min_corners, max_corners + 1)
            
            # Distribute between teams (home team gets slight advantage)
            home_corners = max(1, round(total_corners * 0.55))
            away_corners = max(1, total_corners - home_corners)
            
            confidence = 0.6 + corner_class * 0.1
            
            return {
                "home_team_corners": home_corners,
                "away_team_corners": away_corners,
                "total_corners": total_corners,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Advanced corners prediction failed: {e}")
            return await self._fallback_corners_prediction(match_data)
    
    async def _extract_score_features(self, match_data: MatchData) -> List[float]:
        """Extract features for score prediction."""
        # This would normally use real team statistics
        # For now, we'll use synthetic features based on team IDs
        home_id = match_data.home_team.id if hasattr(match_data, 'home_team') else 1
        away_id = match_data.away_team.id if hasattr(match_data, 'away_team') else 2
        
        # Generate pseudo-random but consistent features based on team IDs
        np.random.seed(home_id + away_id)
        
        features = [
            np.random.uniform(1.0, 2.5),  # home_attack
            np.random.uniform(0.5, 1.5),  # home_defense
            np.random.uniform(1.0, 2.5),  # away_attack
            np.random.uniform(0.5, 1.5),  # away_defense
            np.random.uniform(-0.5, 0.5), # home_form
            np.random.uniform(-0.5, 0.5), # away_form
            np.random.uniform(-0.3, 0.3), # head_to_head
            np.random.uniform(-5, 5)      # position_diff
        ]
        
        return features
    
    async def _extract_cards_features(self, match_data: MatchData) -> List[float]:
        """Extract features for cards prediction."""
        home_id = match_data.home_team.id if hasattr(match_data, 'home_team') else 1
        away_id = match_data.away_team.id if hasattr(match_data, 'away_team') else 2
        
        np.random.seed(home_id * 2 + away_id)
        
        features = [
            np.random.uniform(2.0, 4.0),  # team_aggression
            np.random.uniform(1.0, 2.5),  # referee_strictness
            np.random.uniform(0.5, 1.5),  # match_importance
            np.random.uniform(0.0, 0.8)   # rivalry
        ]
        
        return features
    
    async def _extract_corners_features(self, match_data: MatchData) -> List[float]:
        """Extract features for corners prediction."""
        home_id = match_data.home_team.id if hasattr(match_data, 'home_team') else 1
        away_id = match_data.away_team.id if hasattr(match_data, 'away_team') else 2
        
        np.random.seed(home_id * 3 + away_id)
        
        features = [
            np.random.uniform(1.5, 2.8),  # attacking_style
            np.random.uniform(0.8, 1.8),  # possession_style
            np.random.uniform(0.5, 1.5),  # defensive_style
            np.random.uniform(1.0, 2.0)   # match_tempo
        ]
        
        return features
    
    def _calculate_score_confidence(self, features: List[float]) -> float:
        """Calculate confidence based on feature stability."""
        # Higher confidence when features are more extreme (clear favorites)
        feature_variance = np.var(features[:4])  # Attack/defense features
        confidence = min(0.9, 0.4 + feature_variance * 0.3)
        return confidence
    
    async def _fallback_score_prediction(self, match_data: MatchData) -> ScorePrediction:
        """Fallback score prediction when ML model fails."""
        return ScorePrediction(
            home_score=1,
            away_score=1,
            confidence=0.3,
            probability=0.5
        )
    
    async def _fallback_cards_prediction(self, match_data: MatchData) -> Dict:
        """Fallback cards prediction when ML model fails."""
        return {
            "home_team_cards": 2,
            "away_team_cards": 2,
            "total_cards": 4,
            "confidence": 0.3
        }
    
    async def _fallback_corners_prediction(self, match_data: MatchData) -> Dict:
        """Fallback corners prediction when ML model fails."""
        return {
            "home_team_corners": 5,
            "away_team_corners": 5,
            "total_corners": 10,
            "confidence": 0.3
        }
    
    async def query_huggingface_model(self, model_name: str, inputs: Dict) -> Optional[Dict]:
        """Query Hugging Face model for predictions."""
        if not self.hf_headers.get("Authorization", "").replace("Bearer ", ""):
            logger.warning("Hugging Face API key not provided")
            return None
        
        try:
            url = f"{self.hf_api_url}/{model_name}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, 
                    headers=self.hf_headers, 
                    json=inputs,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"HF API returned status {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Hugging Face API call failed: {e}")
            return None