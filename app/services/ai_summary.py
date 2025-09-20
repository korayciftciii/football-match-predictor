"""OpenRouter AI integration for generating prediction summaries."""

import asyncio
from typing import Dict, Any, Optional, List
import httpx
from openai import AsyncOpenAI
import json

from app.models.schemas import MatchPredictions, Match, TeamStats, MatchFeatures
from app.config import settings
from app.utils.logger import get_logger
from app.utils.cache import cached
from app.utils.circuit_breaker import circuit_breaker, OPENROUTER_CIRCUIT

logger = get_logger(__name__)


class AISummaryError(Exception):
    """Custom exception for AI summary generation errors."""
    pass


class AISummaryGenerator:
    """Service for generating AI-powered match analysis summaries."""
    
    def __init__(self):
        self.client = AsyncOpenAI(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/football-match-predictor",  # Your site URL
                "X-Title": "Football Match Predictor",  # Your site name
            }
        )
        
        # Available models on OpenRouter (free tier) - Updated with deepseek
        self.available_models = [
            "deepseek/deepseek-chat-v3.1:free",  # Primary model (user requested)
            "meta-llama/llama-3.2-3b-instruct:free",  # Backup free model
            "google/gemma-2-9b-it:free",  # Third option
            "mistralai/mistral-7b-instruct:free"  # Fourth option
        ]
        
        self.default_model = "deepseek/deepseek-chat-v3.1:free"  # Use deepseek as requested
        self.max_retries = 3
        self.timeout = 30
    
    async def generate_match_summary(self, match: Match, predictions: MatchPredictions, 
                                   home_stats: TeamStats, away_stats: TeamStats,
                                   features: MatchFeatures) -> str:
        """
        Generate comprehensive match analysis summary in Turkish.
        
        Args:
            match: Match information
            predictions: ML predictions
            home_stats: Home team statistics
            away_stats: Away team statistics
            features: Match features
            
        Returns:
            Turkish language summary string
        """
        try:
            logger.info(f"Generating AI summary for match {match.id}")
            
            # Prepare context data
            context = self._prepare_context(match, predictions, home_stats, away_stats, features)
            
            # Generate summary using OpenRouter
            summary = await self._generate_summary_with_ai(context)
            
            if not summary:
                logger.warning("AI summary generation failed, using fallback")
                summary = self._generate_fallback_summary(match, predictions, home_stats, away_stats)
            
            logger.info(f"AI summary generated successfully for match {match.id}")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating AI summary: {e}")
            return self._generate_fallback_summary(match, predictions, home_stats, away_stats)
    
    def _prepare_context(self, match: Match, predictions: MatchPredictions,
                        home_stats: TeamStats, away_stats: TeamStats,
                        features: MatchFeatures) -> Dict[str, Any]:
        """Prepare context data for AI model."""
        return {
            "match_info": {
                "home_team": match.home_team.name,
                "away_team": match.away_team.name,
                "league": match.league,
                "kickoff_time": match.kickoff_time.strftime("%d.%m.%Y %H:%M")
            },
            "predictions": {
                "score": f"{predictions.score_prediction.home_score}-{predictions.score_prediction.away_score}",
                "score_confidence": f"{predictions.score_prediction.confidence:.1%}",
                "goal_scorer": predictions.goal_scorer_prediction.player_name if predictions.goal_scorer_prediction else "Belirsiz",
                "goal_scorer_probability": f"{predictions.goal_scorer_prediction.probability:.1%}" if predictions.goal_scorer_prediction else "N/A",
                "yellow_cards": predictions.yellow_cards_prediction.total_cards,
                "corners": predictions.corners_prediction.total_corners,
                "first_half": f"{predictions.first_half_prediction.home_score}-{predictions.first_half_prediction.away_score}",
                "overall_confidence": f"{predictions.confidence_score:.1%}"
            },
            "team_stats": {
                "home": {
                    "name": match.home_team.name,
                    "goals_avg": f"{home_stats.goals_scored_avg:.1f}",
                    "conceded_avg": f"{home_stats.goals_conceded_avg:.1f}",
                    "form": f"{home_stats.overall_form:.0f}%",
                    "home_wins": home_stats.home_performance.wins,
                    "home_matches": home_stats.home_performance.matches_played
                },
                "away": {
                    "name": match.away_team.name,
                    "goals_avg": f"{away_stats.goals_scored_avg:.1f}",
                    "conceded_avg": f"{away_stats.goals_conceded_avg:.1f}",
                    "form": f"{away_stats.overall_form:.0f}%",
                    "away_wins": away_stats.away_performance.wins,
                    "away_matches": away_stats.away_performance.matches_played
                }
            },
            "analysis": {
                "home_advantage": f"{features.home_advantage:.1%}",
                "form_difference": f"{features.recent_form_diff:+.0f}",
                "h2h_ratio": f"{features.head_to_head_ratio:.1%}"
            }
        }
    
    @circuit_breaker("openrouter_ai", OPENROUTER_CIRCUIT)
    async def _generate_summary_with_ai(self, context: Dict[str, Any]) -> Optional[str]:
        """Generate summary using OpenRouter AI with circuit breaker protection."""
        for attempt in range(self.max_retries):
            try:
                prompt = self._create_turkish_prompt(context)
                
                response = await self.client.chat.completions.create(
                    model=self.default_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Sen bir futbol analisti ve tahmin uzmanısın. Verilen istatistikleri kullanarak Türkçe, anlaşılır ve ilgi çekici maç analizleri yazıyorsun. Analiz profesyonel ama sade bir dille yazılmalı."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=500,
                    temperature=0.7,
                    timeout=self.timeout
                )
                
                if response.choices and response.choices[0].message.content:
                    summary = response.choices[0].message.content.strip()
                    
                    # Validate summary quality
                    if self._validate_summary(summary):
                        return summary
                    else:
                        logger.warning(f"Generated summary failed validation on attempt {attempt + 1}")
                
            except Exception as e:
                logger.warning(f"AI generation attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    # Try different model on retry
                    if attempt == 1 and len(self.available_models) > 1:
                        self.default_model = self.available_models[1]  # meta-llama/llama-3.2-3b-instruct:free
                        logger.info(f"Switching to fallback model: {self.default_model}")
                    elif attempt == 2 and len(self.available_models) > 2:
                        self.default_model = self.available_models[2]  # google/gemma-2-9b-it:free
                        logger.info(f"Switching to third model: {self.default_model}")
                    
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
        return None
    
    def _create_turkish_prompt(self, context: Dict[str, Any]) -> str:
        """Create Turkish language prompt for AI model."""
        match_info = context["match_info"]
        predictions = context["predictions"]
        home_stats = context["team_stats"]["home"]
        away_stats = context["team_stats"]["away"]
        analysis = context["analysis"]
        
        prompt = f"""
{match_info['league']} maçı: {match_info['home_team']} vs {match_info['away_team']}
Maç Tarihi: {match_info['kickoff_time']}

TAHMIN SONUÇLARI:
- Skor Tahmini: {predictions['score']} (Güven: {predictions['score_confidence']})
- Gol Atacak Oyuncu: {predictions['goal_scorer']} (Olasılık: {predictions['goal_scorer_probability']})
- Sarı Kart: {predictions['yellow_cards']} adet
- Korner: {predictions['corners']} adet
- İlk Yarı: {predictions['first_half']}
- Genel Güven: {predictions['overall_confidence']}

TAKIM İSTATİSTİKLERİ:
{home_stats['name']}:
- Gol Ortalaması: {home_stats['goals_avg']} gol/maç
- Yediği Gol: {home_stats['conceded_avg']} gol/maç
- Form: {home_stats['form']}
- Evde: {home_stats['home_wins']}/{home_stats['home_matches']} galibiyet

{away_stats['name']}:
- Gol Ortalaması: {away_stats['goals_avg']} gol/maç
- Yediği Gol: {away_stats['conceded_avg']} gol/maç
- Form: {away_stats['form']}
- Deplasmanda: {away_stats['away_wins']}/{away_stats['away_matches']} galibiyet

ANALİZ FAKTÖRLERI:
- Ev Sahibi Avantajı: {analysis['home_advantage']}
- Form Farkı: {analysis['form_difference']}
- Geçmiş Karşılaşmalar: {analysis['h2h_ratio']} (ev sahibi lehine)

Bu verileri kullanarak 3-4 cümlelik, anlaşılır ve ilgi çekici bir Türkçe maç analizi yaz. 
Tahminlerin nedenlerini açıkla ve hangi faktörlerin önemli olduğunu belirt.
"""
        return prompt
    
    def _validate_summary(self, summary: str) -> bool:
        """Validate generated summary quality."""
        if not summary or len(summary) < 50:
            return False
        
        # Check for Turkish content
        turkish_words = ["maç", "takım", "gol", "skor", "tahmin", "analiz", "form", "avantaj"]
        if not any(word in summary.lower() for word in turkish_words):
            return False
        
        # Check for reasonable length
        if len(summary) > 1000:
            return False
        
        return True
    
    def _generate_fallback_summary(self, match: Match, predictions: MatchPredictions,
                                 home_stats: TeamStats, away_stats: TeamStats) -> str:
        """Generate fallback summary when AI fails."""
        try:
            home_team = match.home_team.name
            away_team = match.away_team.name
            score = f"{predictions.score_prediction.home_score}-{predictions.score_prediction.away_score}"
            
            # Determine likely winner
            if predictions.score_prediction.home_score > predictions.score_prediction.away_score:
                winner = home_team
                winner_reason = f"ev sahibi avantajı ve {home_stats.goals_scored_avg:.1f} gol ortalaması"
            elif predictions.score_prediction.away_score > predictions.score_prediction.home_score:
                winner = away_team
                winner_reason = f"{away_stats.goals_scored_avg:.1f} gol ortalaması ve iyi deplasman formu"
            else:
                winner = None
                winner_reason = "her iki takımın da benzer performansı"
            
            # Goal scorer info
            scorer_info = ""
            if predictions.goal_scorer_prediction:
                scorer_info = f" {predictions.goal_scorer_prediction.player_name}'ın gol atma ihtimali yüksek."
            
            # Build summary
            if winner:
                summary = f"{home_team} - {away_team} maçında {score} skoruyla {winner}'ın galip geleceği tahmin ediliyor. "
                summary += f"Bu tahminin nedeni {winner_reason}.{scorer_info} "
            else:
                summary = f"{home_team} - {away_team} maçında {score} beraberlik bekleniyor. "
                summary += f"Bu tahminin nedeni {winner_reason}.{scorer_info} "
            
            summary += f"Maçta toplam {predictions.yellow_cards_prediction.total_cards} sarı kart ve "
            summary += f"{predictions.corners_prediction.total_corners} korner bekleniyor."
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating fallback summary: {e}")
            return f"{match.home_team.name} - {match.away_team.name} maçı için analiz tamamlandı. Detaylı tahminler mevcut."
    
    @cached(ttl=3600, key_prefix="ai_summary")
    async def generate_quick_summary(self, prediction_type: str, data: Dict[str, Any]) -> str:
        """
        Generate quick summary for specific prediction type.
        
        Args:
            prediction_type: Type of prediction (score, goal_scorer, cards, etc.)
            data: Prediction data
            
        Returns:
            Quick summary string in Turkish
        """
        try:
            if prediction_type == "score":
                return await self._generate_score_summary(data)
            elif prediction_type == "goal_scorer":
                return await self._generate_goal_scorer_summary(data)
            elif prediction_type == "cards":
                return await self._generate_cards_summary(data)
            elif prediction_type == "corners":
                return await self._generate_corners_summary(data)
            else:
                return "Tahmin analizi tamamlandı."
                
        except Exception as e:
            logger.error(f"Error generating quick summary for {prediction_type}: {e}")
            return f"{prediction_type} tahmini için analiz tamamlandı."
    
    async def _generate_score_summary(self, data: Dict[str, Any]) -> str:
        """Generate score-specific summary."""
        try:
            prompt = f"""
Futbol maçı skor tahmini: {data.get('score', 'N/A')}
Güven oranı: {data.get('confidence', 0):.1%}
Ev sahibi gol ortalaması: {data.get('home_goals_avg', 0):.1f}
Deplasman gol ortalaması: {data.get('away_goals_avg', 0):.1f}

Bu skor tahminini 1-2 cümleyle Türkçe açıkla.
"""
            
            response = await self.client.chat.completions.create(
                model=self.default_model,
                messages=[
                    {"role": "system", "content": "Sen bir futbol analisti ve tahmin uzmanısın. Kısa ve öz açıklamalar yapıyorsun."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.6,
                timeout=15
            )
            
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.warning(f"AI score summary failed: {e}")
        
        # Fallback
        score = data.get('score', 'N/A')
        confidence = data.get('confidence', 0)
        return f"Skor tahmini {score} (%{confidence*100:.0f} güvenle)."
    
    async def _generate_goal_scorer_summary(self, data: Dict[str, Any]) -> str:
        """Generate goal scorer summary."""
        player_name = data.get('player_name', 'Belirsiz')
        probability = data.get('probability', 0)
        
        if player_name == 'Belirsiz':
            return "Gol atacak oyuncu belirsiz."
        
        return f"{player_name}'ın gol atma ihtimali %{probability*100:.0f}."
    
    async def _generate_cards_summary(self, data: Dict[str, Any]) -> str:
        """Generate cards summary."""
        total_cards = data.get('total_cards', 0)
        return f"Maçta toplam {total_cards} sarı kart bekleniyor."
    
    async def _generate_corners_summary(self, data: Dict[str, Any]) -> str:
        """Generate corners summary."""
        total_corners = data.get('total_corners', 0)
        return f"Maçta toplam {total_corners} korner bekleniyor."
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test OpenRouter connection."""
        try:
            logger.info(f"Testing OpenRouter connection with model: {self.default_model}")
            
            response = await self.client.chat.completions.create(
                model=self.default_model,
                messages=[
                    {"role": "user", "content": "Test mesajı. Sadece 'Bağlantı başarılı' diye cevap ver."}
                ],
                max_tokens=10,
                timeout=10
            )
            
            if response.choices and response.choices[0].message.content:
                return {
                    "status": "success",
                    "model": self.default_model,
                    "response": response.choices[0].message.content.strip(),
                    "usage": response.usage.dict() if response.usage else None
                }
            
        except Exception as e:
            logger.error(f"OpenRouter connection test failed: {e}")
            
            # Try fallback model
            try:
                logger.info("Trying fallback model...")
                fallback_model = "meta-llama/llama-3.2-3b-instruct:free"
                
                response = await self.client.chat.completions.create(
                    model=fallback_model,
                    messages=[
                        {"role": "user", "content": "Test"}
                    ],
                    max_tokens=5,
                    timeout=10
                )
                
                if response.choices and response.choices[0].message.content:
                    return {
                        "status": "success",
                        "model": fallback_model,
                        "response": response.choices[0].message.content.strip(),
                        "note": "Using fallback model"
                    }
                    
            except Exception as fallback_error:
                logger.error(f"Fallback model also failed: {fallback_error}")
            
            return {
                "status": "failed",
                "error": str(e),
                "model_attempted": self.default_model
            }
        
        return {"status": "failed", "error": "No response received"}
    
    def format_prediction_explanation(self, prediction_type: str, data: Dict[str, Any]) -> str:
        """
        Format prediction explanation for display.
        
        Args:
            prediction_type: Type of prediction
            data: Prediction data
            
        Returns:
            Formatted explanation string
        """
        try:
            if prediction_type == "score":
                score = data.get('score', 'N/A')
                confidence = data.get('confidence', 0)
                return f"Skor Tahmini: {score} (%{confidence*100:.0f} güven)"
            
            elif prediction_type == "goal_scorer":
                player = data.get('player_name', 'Belirsiz')
                prob = data.get('probability', 0)
                return f"Gol Atacak Oyuncu: {player} (%{prob*100:.0f} olasılık)"
            
            elif prediction_type == "cards":
                total = data.get('total_cards', 0)
                home = data.get('home_cards', 0)
                away = data.get('away_cards', 0)
                return f"Sarı Kart: {total} toplam ({home} ev sahibi, {away} deplasman)"
            
            elif prediction_type == "corners":
                total = data.get('total_corners', 0)
                home = data.get('home_corners', 0)
                away = data.get('away_corners', 0)
                return f"Korner: {total} toplam ({home} ev sahibi, {away} deplasman)"
            
            elif prediction_type == "first_half":
                result = data.get('result', 'Belirsiz')
                score = data.get('score', 'N/A')
                return f"İlk Yarı: {score} ({result})"
            
            else:
                return f"{prediction_type} tahmini mevcut"
                
        except Exception as e:
            logger.error(f"Error formatting prediction explanation: {e}")
            return f"{prediction_type} tahmini"