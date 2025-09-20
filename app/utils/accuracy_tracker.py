"""Prediction accuracy tracking and validation system."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from app.models.schemas import MatchPredictions, Match, MatchStatus
from app.utils.logger import get_logger
from app.utils.cache import cache_manager

logger = get_logger(__name__)


class PredictionOutcome(Enum):
    """Prediction outcome types."""
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIALLY_CORRECT = "partially_correct"
    PENDING = "pending"
    CANCELLED = "cancelled"


@dataclass
class PredictionAccuracy:
    """Accuracy metrics for a specific prediction."""
    prediction_id: str
    match_id: int
    prediction_type: str
    predicted_value: Any
    actual_value: Optional[Any] = None
    outcome: PredictionOutcome = PredictionOutcome.PENDING
    confidence: float = 0.0
    accuracy_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    validated_at: Optional[datetime] = None


@dataclass
class AccuracyStats:
    """Overall accuracy statistics."""
    total_predictions: int
    correct_predictions: int
    incorrect_predictions: int
    partially_correct_predictions: int
    pending_predictions: int
    
    @property
    def accuracy_rate(self) -> float:
        """Calculate overall accuracy rate."""
        validated = self.correct_predictions + self.incorrect_predictions + self.partially_correct_predictions
        if validated == 0:
            return 0.0
        return (self.correct_predictions + (self.partially_correct_predictions * 0.5)) / validated * 100
    
    @property
    def validation_rate(self) -> float:
        """Calculate validation rate (how many predictions have been validated)."""
        if self.total_predictions == 0:
            return 0.0
        validated = self.total_predictions - self.pending_predictions
        return (validated / self.total_predictions) * 100


class AccuracyTracker:
    """Tracks and validates prediction accuracy over time."""
    
    def __init__(self):
        self.predictions: Dict[str, PredictionAccuracy] = {}
        self.stats_by_type: Dict[str, AccuracyStats] = {}
    
    async def record_prediction(self, match: Match, predictions: MatchPredictions) -> str:
        """
        Record a prediction for future accuracy validation.
        
        Args:
            match: Match object
            predictions: Predictions made
            
        Returns:
            Prediction ID for tracking
        """
        prediction_id = f"{match.id}_{int(datetime.now().timestamp())}"
        
        # Record individual prediction components
        prediction_records = []
        
        # Score prediction
        score_record = PredictionAccuracy(
            prediction_id=f"{prediction_id}_score",
            match_id=match.id,
            prediction_type="score",
            predicted_value=f"{predictions.score_prediction.home_score}-{predictions.score_prediction.away_score}",
            confidence=predictions.score_prediction.confidence
        )
        prediction_records.append(score_record)
        
        # Goal scorer prediction
        if predictions.goal_scorer_prediction:
            scorer_record = PredictionAccuracy(
                prediction_id=f"{prediction_id}_scorer",
                match_id=match.id,
                prediction_type="goal_scorer",
                predicted_value=predictions.goal_scorer_prediction.player_name,
                confidence=predictions.goal_scorer_prediction.probability
            )
            prediction_records.append(scorer_record)
        
        # Cards prediction
        cards_record = PredictionAccuracy(
            prediction_id=f"{prediction_id}_cards",
            match_id=match.id,
            prediction_type="yellow_cards",
            predicted_value=predictions.yellow_cards_prediction.total_cards,
            confidence=predictions.yellow_cards_prediction.confidence
        )
        prediction_records.append(cards_record)
        
        # Corners prediction
        corners_record = PredictionAccuracy(
            prediction_id=f"{prediction_id}_corners",
            match_id=match.id,
            prediction_type="corners",
            predicted_value=predictions.corners_prediction.total_corners,
            confidence=predictions.corners_prediction.confidence
        )
        prediction_records.append(corners_record)
        
        # Store all records
        for record in prediction_records:
            self.predictions[record.prediction_id] = record
        
        # Cache for persistence
        await self._cache_predictions(prediction_records)
        
        logger.info(f"Recorded {len(prediction_records)} predictions for match {match.id}")
        return prediction_id
    
    async def validate_prediction(self, match_id: int, actual_results: Dict[str, Any]) -> List[PredictionAccuracy]:
        """
        Validate predictions against actual match results.
        
        Args:
            match_id: Match ID
            actual_results: Actual match results
            
        Returns:
            List of validated prediction accuracies
        """
        # Find predictions for this match
        match_predictions = [
            pred for pred in self.predictions.values()
            if pred.match_id == match_id and pred.outcome == PredictionOutcome.PENDING
        ]
        
        if not match_predictions:
            logger.warning(f"No pending predictions found for match {match_id}")
            return []
        
        validated_predictions = []
        
        for prediction in match_predictions:
            try:
                # Validate based on prediction type
                if prediction.prediction_type == "score":
                    actual_score = actual_results.get("final_score")
                    if actual_score:
                        prediction.actual_value = actual_score
                        prediction.outcome = (
                            PredictionOutcome.CORRECT if prediction.predicted_value == actual_score
                            else PredictionOutcome.INCORRECT
                        )
                        prediction.accuracy_score = 1.0 if prediction.outcome == PredictionOutcome.CORRECT else 0.0
                
                elif prediction.prediction_type == "goal_scorer":
                    actual_scorers = actual_results.get("goal_scorers", [])
                    prediction.actual_value = actual_scorers
                    
                    if prediction.predicted_value in actual_scorers:
                        prediction.outcome = PredictionOutcome.CORRECT
                        prediction.accuracy_score = 1.0
                    else:
                        prediction.outcome = PredictionOutcome.INCORRECT
                        prediction.accuracy_score = 0.0
                
                elif prediction.prediction_type == "yellow_cards":
                    actual_cards = actual_results.get("yellow_cards")
                    if actual_cards is not None:
                        prediction.actual_value = actual_cards
                        
                        # Allow ±1 card tolerance for partial correctness
                        diff = abs(prediction.predicted_value - actual_cards)
                        if diff == 0:
                            prediction.outcome = PredictionOutcome.CORRECT
                            prediction.accuracy_score = 1.0
                        elif diff == 1:
                            prediction.outcome = PredictionOutcome.PARTIALLY_CORRECT
                            prediction.accuracy_score = 0.5
                        else:
                            prediction.outcome = PredictionOutcome.INCORRECT
                            prediction.accuracy_score = 0.0
                
                elif prediction.prediction_type == "corners":
                    actual_corners = actual_results.get("corners")
                    if actual_corners is not None:
                        prediction.actual_value = actual_corners
                        
                        # Allow ±2 corners tolerance for partial correctness
                        diff = abs(prediction.predicted_value - actual_corners)
                        if diff <= 1:
                            prediction.outcome = PredictionOutcome.CORRECT
                            prediction.accuracy_score = 1.0
                        elif diff <= 2:
                            prediction.outcome = PredictionOutcome.PARTIALLY_CORRECT
                            prediction.accuracy_score = 0.5
                        else:
                            prediction.outcome = PredictionOutcome.INCORRECT
                            prediction.accuracy_score = 0.0
                
                prediction.validated_at = datetime.now()
                validated_predictions.append(prediction)
                
                logger.debug(f"Validated prediction {prediction.prediction_id}: {prediction.outcome.value}")
                
            except Exception as e:
                logger.error(f"Error validating prediction {prediction.prediction_id}: {e}")
                continue
        
        # Update statistics
        await self._update_accuracy_stats()
        
        # Cache validated predictions
        await self._cache_predictions(validated_predictions)
        
        logger.info(f"Validated {len(validated_predictions)} predictions for match {match_id}")
        return validated_predictions
    
    async def get_accuracy_stats(self, prediction_type: Optional[str] = None) -> Dict[str, Any]:
        """Get accuracy statistics."""
        if prediction_type:
            # Get stats for specific prediction type
            type_predictions = [
                pred for pred in self.predictions.values()
                if pred.prediction_type == prediction_type
            ]
        else:
            # Get overall stats
            type_predictions = list(self.predictions.values())
        
        if not type_predictions:
            return {
                "prediction_type": prediction_type or "all",
                "total_predictions": 0,
                "accuracy_rate": 0.0,
                "validation_rate": 0.0
            }
        
        # Calculate stats
        total = len(type_predictions)
        correct = sum(1 for p in type_predictions if p.outcome == PredictionOutcome.CORRECT)
        incorrect = sum(1 for p in type_predictions if p.outcome == PredictionOutcome.INCORRECT)
        partially_correct = sum(1 for p in type_predictions if p.outcome == PredictionOutcome.PARTIALLY_CORRECT)
        pending = sum(1 for p in type_predictions if p.outcome == PredictionOutcome.PENDING)
        
        validated = total - pending
        accuracy_rate = 0.0
        if validated > 0:
            accuracy_rate = (correct + (partially_correct * 0.5)) / validated * 100
        
        validation_rate = (validated / total * 100) if total > 0 else 0.0
        
        return {
            "prediction_type": prediction_type or "all",
            "total_predictions": total,
            "correct_predictions": correct,
            "incorrect_predictions": incorrect,
            "partially_correct_predictions": partially_correct,
            "pending_predictions": pending,
            "accuracy_rate": round(accuracy_rate, 2),
            "validation_rate": round(validation_rate, 2),
            "average_confidence": round(
                sum(p.confidence for p in type_predictions) / total, 3
            ) if total > 0 else 0.0
        }
    
    async def _update_accuracy_stats(self):
        """Update internal accuracy statistics."""
        # Update stats by prediction type
        prediction_types = set(pred.prediction_type for pred in self.predictions.values())
        
        for pred_type in prediction_types:
            type_predictions = [
                pred for pred in self.predictions.values()
                if pred.prediction_type == pred_type
            ]
            
            total = len(type_predictions)
            correct = sum(1 for p in type_predictions if p.outcome == PredictionOutcome.CORRECT)
            incorrect = sum(1 for p in type_predictions if p.outcome == PredictionOutcome.INCORRECT)
            partially_correct = sum(1 for p in type_predictions if p.outcome == PredictionOutcome.PARTIALLY_CORRECT)
            pending = sum(1 for p in type_predictions if p.outcome == PredictionOutcome.PENDING)
            
            self.stats_by_type[pred_type] = AccuracyStats(
                total_predictions=total,
                correct_predictions=correct,
                incorrect_predictions=incorrect,
                partially_correct_predictions=partially_correct,
                pending_predictions=pending
            )
    
    async def _cache_predictions(self, predictions: List[PredictionAccuracy]):
        """Cache predictions for persistence."""
        try:
            for prediction in predictions:
                cache_key = f"accuracy_tracking:{prediction.prediction_id}"
                await cache_manager.set(cache_key, prediction, ttl=86400 * 7)  # 7 days
        except Exception as e:
            logger.warning(f"Failed to cache prediction accuracy data: {e}")
    
    async def cleanup_old_predictions(self, days_old: int = 30):
        """Clean up old prediction records."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        old_predictions = [
            pred_id for pred_id, pred in self.predictions.items()
            if pred.created_at < cutoff_date
        ]
        
        for pred_id in old_predictions:
            del self.predictions[pred_id]
        
        logger.info(f"Cleaned up {len(old_predictions)} old prediction records")


# Global accuracy tracker
accuracy_tracker = AccuracyTracker()