"""Analysis service for feature engineering and match analysis."""

import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np

from app.models.schemas import (
    Match, Team, TeamStats, PlayerStats, MatchFeatures, TeamMetrics,
    MatchResult, PerformanceMetrics, MatchPredictions, MatchData
)
from app.services.fetch_data import FootballDataFetcher
from app.services.ml_model import PredictionModels
from app.services.advanced_ml_models import AdvancedMLPredictor
from app.services.ai_summary import AISummaryGenerator
from app.utils.logger import get_logger
from app.utils.cache import cached

logger = get_logger(__name__)


class AnalysisError(Exception):
    """Custom exception for analysis-related errors."""
    pass


class MatchAnalyzer:
    """Service for match analysis and feature engineering."""
    
    def __init__(self, data_fetcher: FootballDataFetcher):
        self.data_fetcher = data_fetcher
        self.ml_models = PredictionModels()
        self.advanced_ml = AdvancedMLPredictor()
        self.ai_summary = AISummaryGenerator()
        
    async def extract_features(self, match: Match, home_stats: TeamStats, away_stats: TeamStats) -> MatchFeatures:
        """
        Extract ML features from match and team data.
        
        Args:
            match: Match object
            home_stats: Home team statistics
            away_stats: Away team statistics
            
        Returns:
            MatchFeatures object with extracted features
        """
        try:
            logger.info(f"Extracting features for match {match.id}: {match.home_team.name} vs {match.away_team.name}")
            
            # Basic goal statistics
            home_goal_avg = home_stats.goals_scored_avg
            away_goal_avg = away_stats.goals_scored_avg
            home_conceded_avg = home_stats.goals_conceded_avg
            away_conceded_avg = away_stats.goals_conceded_avg
            
            # Calculate home advantage factor
            home_advantage = self._calculate_home_advantage(home_stats, away_stats)
            
            # Calculate recent form difference
            recent_form_diff = self._calculate_form_difference(home_stats, away_stats)
            
            # Get head-to-head ratio
            h2h_ratio = await self._calculate_h2h_ratio(match.home_team.id, match.away_team.id)
            
            # Additional features
            home_yellow_cards_avg = home_stats.yellow_cards_avg
            away_yellow_cards_avg = away_stats.yellow_cards_avg
            home_corners_avg = home_stats.corners_avg
            away_corners_avg = away_stats.corners_avg
            
            features = MatchFeatures(
                home_goal_avg=home_goal_avg,
                away_goal_avg=away_goal_avg,
                home_conceded_avg=home_conceded_avg,
                away_conceded_avg=away_conceded_avg,
                home_advantage=home_advantage,
                recent_form_diff=recent_form_diff,
                head_to_head_ratio=h2h_ratio,
                home_yellow_cards_avg=home_yellow_cards_avg,
                away_yellow_cards_avg=away_yellow_cards_avg,
                home_corners_avg=home_corners_avg,
                away_corners_avg=away_corners_avg
            )
            
            logger.info(f"Features extracted successfully for match {match.id}")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features for match {match.id}: {e}")
            raise AnalysisError(f"Failed to extract features: {str(e)}")
    
    def calculate_team_metrics(self, team_stats: TeamStats, recent_matches: List[Match]) -> TeamMetrics:
        """
        Calculate advanced team metrics from statistics and recent matches.
        
        Args:
            team_stats: Team statistics
            recent_matches: List of recent matches
            
        Returns:
            TeamMetrics object with calculated metrics
        """
        try:
            logger.debug(f"Calculating team metrics for team {team_stats.team_id}")
            
            # Calculate attack strength (goals scored relative to league average)
            league_avg_goals = 1.5  # Approximate league average
            attack_strength = max(team_stats.goals_scored_avg / league_avg_goals, 0.1)
            
            # Calculate defense strength (inverse of goals conceded)
            defense_strength = max(2.0 - team_stats.goals_conceded_avg, 0.1)
            
            # Calculate form rating from recent matches
            form_rating = self._calculate_form_rating(recent_matches, team_stats.team_id)
            
            # Calculate home advantage factor
            home_advantage_factor = self._calculate_home_advantage_factor(team_stats)
            
            # Calculate discipline rating (lower is better)
            discipline_rating = min(team_stats.yellow_cards_avg, 10.0)
            
            metrics = TeamMetrics(
                team_id=team_stats.team_id,
                attack_strength=attack_strength,
                defense_strength=defense_strength,
                form_rating=form_rating,
                home_advantage_factor=home_advantage_factor,
                discipline_rating=discipline_rating
            )
            
            logger.debug(f"Team metrics calculated for team {team_stats.team_id}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating team metrics: {e}")
            raise AnalysisError(f"Failed to calculate team metrics: {str(e)}")
    
    @cached(ttl=10800, key_prefix="match_predictions")  # Cache for 3 hours
    async def generate_predictions(self, match: Match) -> MatchPredictions:
        """
        Generate comprehensive match predictions.
        
        Args:
            match: Match object
            
        Returns:
            MatchPredictions object (placeholder - will be completed in ML task)
        """
        try:
            logger.info(f"Generating predictions for match {match.id}")
            
            # Get team statistics with fallback
            try:
                home_stats = await self.data_fetcher.get_team_stats(match.home_team.id)
            except Exception as e:
                logger.warning(f"Failed to get home team stats, using fallback: {e}")
                home_stats = self._create_fallback_team_stats(match.home_team.id)
            
            try:
                away_stats = await self.data_fetcher.get_team_stats(match.away_team.id)
            except Exception as e:
                logger.warning(f"Failed to get away team stats, using fallback: {e}")
                away_stats = self._create_fallback_team_stats(match.away_team.id)
            
            # Extract features
            features = await self.extract_features(match, home_stats, away_stats)
            
            # This is a placeholder - actual ML predictions will be implemented in Task 6
            # For now, return basic statistical predictions
            basic_predictions = await self._generate_basic_predictions(match, home_stats, away_stats, features)
            
            logger.info(f"Predictions generated for match {match.id}")
            return basic_predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions for match {match.id}: {e}")
            raise AnalysisError(f"Failed to generate predictions: {str(e)}")
    
    def _calculate_home_advantage(self, home_stats: TeamStats, away_stats: TeamStats) -> float:
        """Calculate home advantage factor (0.0 to 1.0)."""
        try:
            # Home team's home performance vs away team's away performance
            home_home_goals = home_stats.home_performance.goals_per_match
            home_home_conceded = home_stats.home_performance.goals_conceded_per_match
            
            away_away_goals = away_stats.away_performance.goals_per_match
            away_away_conceded = away_stats.away_performance.goals_conceded_per_match
            
            # Calculate advantage based on goal difference
            home_goal_diff = home_home_goals - home_home_conceded
            away_goal_diff = away_away_goals - away_away_conceded
            
            # Normalize to 0-1 scale
            advantage = (home_goal_diff - away_goal_diff + 4) / 8  # Assuming max diff is 4
            return max(0.0, min(1.0, advantage))
            
        except Exception as e:
            logger.warning(f"Error calculating home advantage: {e}")
            return 0.5  # Default neutral advantage
    
    def _calculate_form_difference(self, home_stats: TeamStats, away_stats: TeamStats) -> float:
        """Calculate recent form difference (-100 to +100)."""
        try:
            home_form = home_stats.overall_form
            away_form = away_stats.overall_form
            
            return home_form - away_form
            
        except Exception as e:
            logger.warning(f"Error calculating form difference: {e}")
            return 0.0  # Default neutral form
    
    def _create_fallback_team_stats(self, team_id: int) -> TeamStats:
        """Create fallback team statistics when API data is unavailable."""
        try:
            # Create basic performance metrics
            home_performance = PerformanceMetrics(
                matches_played=10,
                wins=4,
                draws=3,
                losses=3,
                goals_scored=15,
                goals_conceded=12,
                clean_sheets=3
            )
            
            away_performance = PerformanceMetrics(
                matches_played=10,
                wins=3,
                draws=4,
                losses=3,
                goals_scored=12,
                goals_conceded=13,
                clean_sheets=2
            )
            
            # Create fallback team stats
            fallback_stats = TeamStats(
                team_id=team_id,
                goals_scored_avg=1.35,  # Average of home and away
                goals_conceded_avg=1.25,
                yellow_cards_avg=2.1,
                corners_avg=5.2,
                home_performance=home_performance,
                away_performance=away_performance,
                recent_matches_count=5
            )
            
            logger.info(f"Created fallback team stats for team {team_id}")
            return fallback_stats
            
        except Exception as e:
            logger.error(f"Error creating fallback team stats: {e}")
            raise AnalysisError(f"Failed to create fallback team stats: {str(e)}")
    
    @cached(ttl=86400, key_prefix="h2h_ratio")  # Cache for 24 hours
    async def _calculate_h2h_ratio(self, home_team_id: int, away_team_id: int) -> float:
        """Calculate head-to-head win ratio for home team (0.0 to 1.0)."""
        try:
            h2h_matches = await self.data_fetcher.get_head_to_head(home_team_id, away_team_id, 10)
            
            if not h2h_matches:
                return 0.5  # Default neutral ratio
            
            home_wins = 0
            total_matches = len(h2h_matches)
            
            for match in h2h_matches:
                # Determine winner based on match status and score
                # This is simplified - in real implementation, we'd need actual scores
                if match.status.value == "FT":  # Only count finished matches
                    # For now, assume random distribution - will be improved with actual score data
                    home_wins += 0.33  # Approximate
            
            return home_wins / total_matches if total_matches > 0 else 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating H2H ratio: {e}")
            return 0.5  # Default neutral ratio
    
    def _calculate_form_rating(self, recent_matches: List[Match], team_id: int) -> float:
        """Calculate form rating from recent matches (0-100)."""
        try:
            if not recent_matches:
                return 50.0  # Default average form
            
            points = 0
            matches_analyzed = 0
            
            for match in recent_matches:
                if match.status.value != "FT":  # Only count finished matches
                    continue
                
                # Determine if team was home or away
                is_home = match.home_team.id == team_id
                
                # For now, use simplified point calculation
                # In real implementation, we'd use actual match results
                # Assuming average performance for now
                points += 1.5  # Average points per match
                matches_analyzed += 1
            
            if matches_analyzed == 0:
                return 50.0
            
            # Convert to 0-100 scale (3 points = 100%, 0 points = 0%)
            avg_points = points / matches_analyzed
            return (avg_points / 3.0) * 100
            
        except Exception as e:
            logger.warning(f"Error calculating form rating: {e}")
            return 50.0  # Default average form
    
    def _calculate_home_advantage_factor(self, team_stats: TeamStats) -> float:
        """Calculate home advantage multiplier (0.0 to 2.0)."""
        try:
            home_perf = team_stats.home_performance
            away_perf = team_stats.away_performance
            
            if home_perf.matches_played == 0 or away_perf.matches_played == 0:
                return 1.0  # Default neutral
            
            home_points_per_match = (home_perf.wins * 3 + home_perf.draws) / home_perf.matches_played
            away_points_per_match = (away_perf.wins * 3 + away_perf.draws) / away_perf.matches_played
            
            if away_points_per_match == 0:
                return 2.0  # Maximum advantage
            
            advantage_factor = home_points_per_match / away_points_per_match
            return max(0.0, min(2.0, advantage_factor))
            
        except Exception as e:
            logger.warning(f"Error calculating home advantage factor: {e}")
            return 1.0  # Default neutral
    
    async def _generate_basic_predictions(self, match: Match, home_stats: TeamStats, 
                                        away_stats: TeamStats, features: MatchFeatures) -> MatchPredictions:
        """Generate advanced ML-powered predictions."""
        try:
            logger.info(f"Generating advanced ML predictions for match {match.id}")
            
            # Create match data for advanced ML models
            match_data = MatchData(
                home_team=match.home_team,
                away_team=match.away_team,
                league_id=203,  # Turkish Super League
                kickoff_time=match.kickoff_time
            )
            
            # Use advanced ML models for better predictions
            try:
                score_prediction = await self.advanced_ml.predict_score_advanced(match_data)
                logger.info(f"Advanced score prediction: {score_prediction.home_score}-{score_prediction.away_score}")
            except Exception as e:
                logger.warning(f"Advanced score prediction failed, using fallback: {e}")
                score_prediction = self.ml_models.predict_score(features)
            
            try:
                card_prediction_dict = await self.advanced_ml.predict_cards_advanced(match_data)
                card_prediction = type('CardPrediction', (), {
                    'home_team_cards': card_prediction_dict['home_team_cards'],
                    'away_team_cards': card_prediction_dict['away_team_cards'],
                    'total_cards': card_prediction_dict['total_cards'],
                    'confidence': card_prediction_dict['confidence']
                })()
                logger.info(f"Advanced cards prediction: {card_prediction.total_cards} total cards")
            except Exception as e:
                logger.warning(f"Advanced cards prediction failed, using fallback: {e}")
                card_prediction = self.ml_models.predict_yellow_cards(home_stats, away_stats)
            
            try:
                corner_prediction_dict = await self.advanced_ml.predict_corners_advanced(match_data)
                corner_prediction = type('CornerPrediction', (), {
                    'home_team_corners': corner_prediction_dict['home_team_corners'],
                    'away_team_corners': corner_prediction_dict['away_team_corners'],
                    'total_corners': corner_prediction_dict['total_corners'],
                    'confidence': corner_prediction_dict['confidence']
                })()
                logger.info(f"Advanced corners prediction: {corner_prediction.total_corners} total corners")
            except Exception as e:
                logger.warning(f"Advanced corners prediction failed, using fallback: {e}")
                corner_prediction = self.ml_models.predict_corners(home_stats, away_stats)
            
            # First half prediction using advanced features
            first_half_prediction = self.ml_models.predict_first_half(features)
            
            # Get player stats for goal scorer prediction
            goal_scorer_prediction = None
            try:
                home_players = await self.data_fetcher.get_player_stats(match.home_team.id)
                away_players = await self.data_fetcher.get_player_stats(match.away_team.id)
                
                # Analyze both teams and pick the most likely scorer
                home_scorer = self.ml_models.predict_goal_scorer(home_stats, home_players)
                away_scorer = self.ml_models.predict_goal_scorer(away_stats, away_players)
                
                # Choose the player with higher probability
                if home_scorer and away_scorer:
                    goal_scorer_prediction = home_scorer if home_scorer.probability > away_scorer.probability else away_scorer
                elif home_scorer:
                    goal_scorer_prediction = home_scorer
                elif away_scorer:
                    goal_scorer_prediction = away_scorer
                    
            except Exception as e:
                logger.warning(f"Error predicting goal scorer: {e}")
            
            # Calculate overall confidence
            confidence_scores = [
                score_prediction.confidence,
                card_prediction.confidence,
                corner_prediction.confidence,
                first_half_prediction.confidence
            ]
            
            if goal_scorer_prediction:
                confidence_scores.append(goal_scorer_prediction.confidence)
            
            overall_confidence = statistics.mean(confidence_scores)
            
            # Create initial predictions object
            predictions = MatchPredictions(
                match_id=match.id,
                score_prediction=score_prediction,
                goal_scorer_prediction=goal_scorer_prediction,
                yellow_cards_prediction=card_prediction,
                corners_prediction=corner_prediction,
                first_half_prediction=first_half_prediction,
                ai_summary="ML modelleri ile tahminler oluşturuldu.",
                confidence_score=overall_confidence
            )
            
            # Generate AI summary
            try:
                features = await self.extract_features(match, home_stats, away_stats)
                ai_summary = await self.ai_summary.generate_match_summary(
                    match, predictions, home_stats, away_stats, features
                )
                predictions.ai_summary = ai_summary
                logger.info(f"AI summary generated for match {match.id}")
            except Exception as e:
                logger.warning(f"AI summary generation failed for match {match.id}: {e}")
                # Keep the default summary
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating ML predictions: {e}")
            raise AnalysisError(f"Failed to generate ML predictions: {str(e)}")
    
    def analyze_player_form(self, players: List[PlayerStats]) -> List[PlayerStats]:
        """
        Analyze and rank players by current form.
        
        Args:
            players: List of player statistics
            
        Returns:
            List of players sorted by goal probability
        """
        try:
            # Sort players by goal probability (descending)
            sorted_players = sorted(players, key=lambda p: p.goal_probability, reverse=True)
            
            # Update goal probabilities based on recent form
            for player in sorted_players:
                # Adjust probability based on minutes played
                if player.minutes_played > 0:
                    form_factor = min(player.minutes_played / 1000, 1.0)  # Normalize by 1000 minutes
                    player.goal_probability *= form_factor
                
                # Boost probability for attackers
                if player.position.value == "Attacker":
                    player.goal_probability *= 1.2
                elif player.position.value == "Midfielder":
                    player.goal_probability *= 0.8
                elif player.position.value == "Defender":
                    player.goal_probability *= 0.3
                
                # Ensure probability stays within bounds
                player.goal_probability = max(0.0, min(1.0, player.goal_probability))
            
            return sorted_players
            
        except Exception as e:
            logger.error(f"Error analyzing player form: {e}")
            return players  # Return original list on error
    
    def calculate_match_importance(self, match: Match, home_stats: TeamStats, away_stats: TeamStats) -> float:
        """
        Calculate match importance factor (0.0 to 1.0).
        
        Args:
            match: Match object
            home_stats: Home team statistics
            away_stats: Away team statistics
            
        Returns:
            float: Match importance factor
        """
        try:
            importance_factors = []
            
            # League importance (simplified)
            league_importance = {
                "Premier League": 1.0,
                "La Liga": 1.0,
                "Bundesliga": 0.95,
                "Serie A": 0.95,
                "Ligue 1": 0.9
            }
            importance_factors.append(league_importance.get(match.league, 0.7))
            
            # Team quality difference (closer teams = more important)
            quality_diff = abs(home_stats.overall_form - away_stats.overall_form)
            quality_importance = 1.0 - (quality_diff / 100)  # Normalize
            importance_factors.append(max(0.5, quality_importance))
            
            # Time factor (weekend games might be more important)
            if match.kickoff_time.weekday() in [5, 6]:  # Saturday, Sunday
                importance_factors.append(1.0)
            else:
                importance_factors.append(0.8)
            
            return statistics.mean(importance_factors)
            
        except Exception as e:
            logger.warning(f"Error calculating match importance: {e}")
            return 0.7  # Default moderate importance