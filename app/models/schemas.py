"""Pydantic models and schemas for Football Match Predictor."""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class MatchStatus(str, Enum):
    """Match status enumeration."""
    NOT_STARTED = "NS"
    FIRST_HALF = "1H"
    HALFTIME = "HT"
    SECOND_HALF = "2H"
    EXTRA_TIME = "ET"
    PENALTY = "P"
    FINISHED = "FT"
    POSTPONED = "PST"
    CANCELLED = "CANC"
    ABANDONED = "ABD"


class MatchResult(str, Enum):
    """Match result enumeration."""
    WIN = "W"
    DRAW = "D"
    LOSS = "L"


class PlayerPosition(str, Enum):
    """Player position enumeration."""
    GOALKEEPER = "Goalkeeper"
    DEFENDER = "Defender"
    MIDFIELDER = "Midfielder"
    ATTACKER = "Attacker"


class PredictionType(str, Enum):
    """Prediction type enumeration."""
    SCORE = "score"
    GOAL_SCORER = "goal_scorer"
    YELLOW_CARDS = "yellow_cards"
    CORNERS = "corners"
    FIRST_HALF = "first_half"


# Core Data Models

class PerformanceMetrics(BaseModel):
    """Performance metrics for home/away analysis."""
    matches_played: int = Field(ge=0)
    wins: int = Field(ge=0)
    draws: int = Field(ge=0)
    losses: int = Field(ge=0)
    goals_scored: int = Field(ge=0)
    goals_conceded: int = Field(ge=0)
    clean_sheets: int = Field(ge=0)
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate percentage."""
        return (self.wins / self.matches_played * 100) if self.matches_played > 0 else 0.0
    
    @property
    def goals_per_match(self) -> float:
        """Calculate average goals scored per match."""
        return self.goals_scored / self.matches_played if self.matches_played > 0 else 0.0
    
    @property
    def goals_conceded_per_match(self) -> float:
        """Calculate average goals conceded per match."""
        return self.goals_conceded / self.matches_played if self.matches_played > 0 else 0.0


class Team(BaseModel):
    """Team model."""
    id: int = Field(gt=0)
    name: str = Field(min_length=1, max_length=100)
    logo_url: Optional[str] = None
    recent_form: List[MatchResult] = Field(default_factory=list, max_length=10)
    
    @field_validator('logo_url')
    @classmethod
    def validate_logo_url(cls, v):
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError('Logo URL must be a valid HTTP/HTTPS URL')
        return v


class MatchData(BaseModel):
    """Match data for ML predictions."""
    home_team: Team
    away_team: Team
    league_id: int = Field(gt=0)
    kickoff_time: datetime


class PlayerStats(BaseModel):
    """Player statistics model."""
    player_id: int = Field(gt=0)
    name: str = Field(min_length=1, max_length=100)
    position: PlayerPosition
    goals_recent: int = Field(ge=0, description="Goals in recent matches")
    assists_recent: int = Field(ge=0, description="Assists in recent matches")
    minutes_played: int = Field(ge=0, description="Total minutes played in recent matches")
    goal_probability: float = Field(ge=0.0, le=1.0, description="Calculated goal scoring probability")
    
    @property
    def goal_contributions(self) -> int:
        """Total goal contributions (goals + assists)."""
        return self.goals_recent + self.assists_recent
    
    @property
    def goals_per_90(self) -> float:
        """Goals per 90 minutes played."""
        return (self.goals_recent * 90) / self.minutes_played if self.minutes_played > 0 else 0.0


class TeamStats(BaseModel):
    """Team statistics model."""
    team_id: int = Field(gt=0)
    goals_scored_avg: float = Field(ge=0.0, description="Average goals scored per match")
    goals_conceded_avg: float = Field(ge=0.0, description="Average goals conceded per match")
    yellow_cards_avg: float = Field(ge=0.0, description="Average yellow cards per match")
    corners_avg: float = Field(ge=0.0, description="Average corners per match")
    home_performance: PerformanceMetrics
    away_performance: PerformanceMetrics
    recent_matches_count: int = Field(ge=1, le=10, default=5)
    
    @property
    def overall_form(self) -> float:
        """Calculate overall form score (0-100)."""
        total_matches = self.home_performance.matches_played + self.away_performance.matches_played
        total_wins = self.home_performance.wins + self.away_performance.wins
        total_draws = self.home_performance.draws + self.away_performance.draws
        
        if total_matches == 0:
            return 0.0
        
        # Win = 3 points, Draw = 1 point, Loss = 0 points
        points = (total_wins * 3) + (total_draws * 1)
        max_points = total_matches * 3
        return (points / max_points) * 100


class Match(BaseModel):
    """Match model."""
    id: int = Field(gt=0)
    home_team: Team
    away_team: Team
    kickoff_time: datetime
    league: str = Field(min_length=1, max_length=100)
    status: MatchStatus = MatchStatus.NOT_STARTED
    venue: Optional[str] = None
    referee: Optional[str] = None
    
    @field_validator('kickoff_time')
    @classmethod
    def validate_kickoff_time(cls, v):
        # Make timezone-aware comparison
        from datetime import timezone
        now = datetime.now(timezone.utc)
        if v.tzinfo is None:
            # If v is naive, make it UTC
            v = v.replace(tzinfo=timezone.utc)
        elif v < now:
            # Allow past times for testing, but log a warning
            pass
        return v


# Prediction Models

class ScorePrediction(BaseModel):
    """Score prediction model."""
    home_score: int = Field(ge=0, le=10)
    away_score: int = Field(ge=0, le=10)
    confidence: float = Field(ge=0.0, le=1.0)
    probability: float = Field(ge=0.0, le=1.0)
    alternative_scores: List[Dict[str, Any]] = Field(default_factory=list)
    
    @property
    def score_string(self) -> str:
        """Get score as string (e.g., '2-1')."""
        return f"{self.home_score}-{self.away_score}"


class PlayerPrediction(BaseModel):
    """Player prediction model."""
    player_id: int = Field(gt=0)
    player_name: str = Field(min_length=1)
    team_id: int = Field(gt=0)
    probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: Optional[str] = None


class CardPrediction(BaseModel):
    """Yellow card prediction model."""
    home_team_cards: int = Field(ge=0, le=10)
    away_team_cards: int = Field(ge=0, le=10)
    total_cards: int = Field(ge=0, le=20)
    confidence: float = Field(ge=0.0, le=1.0)
    
    @field_validator('total_cards')
    @classmethod
    def validate_total_cards(cls, v, info):
        if info.data and 'home_team_cards' in info.data and 'away_team_cards' in info.data:
            expected_total = info.data['home_team_cards'] + info.data['away_team_cards']
            if v != expected_total:
                raise ValueError('Total cards must equal sum of home and away cards')
        return v


class CornerPrediction(BaseModel):
    """Corner prediction model."""
    home_team_corners: int = Field(ge=0, le=20)
    away_team_corners: int = Field(ge=0, le=20)
    total_corners: int = Field(ge=0, le=40)
    confidence: float = Field(ge=0.0, le=1.0)
    
    @field_validator('total_corners')
    @classmethod
    def validate_total_corners(cls, v, info):
        if info.data and 'home_team_corners' in info.data and 'away_team_corners' in info.data:
            expected_total = info.data['home_team_corners'] + info.data['away_team_corners']
            if v != expected_total:
                raise ValueError('Total corners must equal sum of home and away corners')
        return v


class FirstHalfPrediction(BaseModel):
    """First half result prediction model."""
    result: MatchResult  # From home team perspective
    home_score: int = Field(ge=0, le=5)
    away_score: int = Field(ge=0, le=5)
    confidence: float = Field(ge=0.0, le=1.0)
    probability: float = Field(ge=0.0, le=1.0)


# Feature Engineering Models

class MatchFeatures(BaseModel):
    """Match features for ML models."""
    home_goal_avg: float = Field(ge=0.0)
    away_goal_avg: float = Field(ge=0.0)
    home_conceded_avg: float = Field(ge=0.0)
    away_conceded_avg: float = Field(ge=0.0)
    home_advantage: float = Field(ge=0.0, le=1.0, description="Home advantage factor")
    recent_form_diff: float = Field(ge=-100.0, le=100.0, description="Form difference (home - away)")
    head_to_head_ratio: float = Field(ge=0.0, le=1.0, description="Historical H2H win ratio for home team")
    home_yellow_cards_avg: float = Field(ge=0.0)
    away_yellow_cards_avg: float = Field(ge=0.0)
    home_corners_avg: float = Field(ge=0.0)
    away_corners_avg: float = Field(ge=0.0)


class TeamMetrics(BaseModel):
    """Calculated team metrics."""
    team_id: int = Field(gt=0)
    attack_strength: float = Field(ge=0.0, description="Attacking strength rating")
    defense_strength: float = Field(ge=0.0, description="Defensive strength rating")
    form_rating: float = Field(ge=0.0, le=100.0, description="Recent form rating")
    home_advantage_factor: float = Field(ge=0.0, le=2.0, description="Home performance multiplier")
    discipline_rating: float = Field(ge=0.0, le=10.0, description="Discipline rating (lower = better)")


# Comprehensive Prediction Model

class MatchPredictions(BaseModel):
    """Complete match predictions model."""
    match_id: int = Field(gt=0)
    score_prediction: ScorePrediction
    goal_scorer_prediction: Optional[PlayerPrediction] = None
    yellow_cards_prediction: CardPrediction
    corners_prediction: CornerPrediction
    first_half_prediction: FirstHalfPrediction
    ai_summary: str = Field(min_length=10, description="AI-generated summary in Turkish")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Overall prediction confidence")
    generated_at: datetime = Field(default_factory=datetime.now)
    
    @property
    def prediction_summary(self) -> Dict[str, str]:
        """Get a summary of all predictions."""
        return {
            "score": self.score_prediction.score_string,
            "goal_scorer": self.goal_scorer_prediction.player_name if self.goal_scorer_prediction else "N/A",
            "total_cards": str(self.yellow_cards_prediction.total_cards),
            "total_corners": str(self.corners_prediction.total_corners),
            "first_half": f"{self.first_half_prediction.home_score}-{self.first_half_prediction.away_score}",
            "confidence": f"{self.confidence_score:.2%}"
        }


# API Response Models

class AnalysisResult(BaseModel):
    """Analysis result model for API responses."""
    match: Match
    predictions: MatchPredictions
    features: MatchFeatures
    processing_time_ms: int = Field(ge=0)
    success: bool = True
    error_message: Optional[str] = None


class HealthCheck(BaseModel):
    """Health check response model."""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    services: Dict[str, str] = Field(default_factory=dict)
    version: str = "1.0.0"