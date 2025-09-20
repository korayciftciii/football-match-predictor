"""
Hugging Face Model Integration for Football Predictions
Provides access to pre-trained models and datasets from Hugging Face Hub
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime

from app.utils.logger import get_logger
from app.utils.cache import CacheManager
from app.config import get_settings

logger = get_logger(__name__)

class HuggingFaceIntegration:
    """Integration with Hugging Face models and datasets."""
    
    def __init__(self):
        self.settings = get_settings()
        self.cache = CacheManager()
        self.api_url = "https://api-inference.huggingface.co"
        self.headers = {
            "Authorization": f"Bearer {self.settings.HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Recommended models for football prediction
        self.models = {
            "football_prediction": "microsoft/DialoGPT-medium",  # For text generation
            "sentiment_analysis": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "classification": "distilbert-base-uncased",
            "regression": "microsoft/DialoGPT-small"
        }
        
        # Football-specific datasets
        self.datasets = {
            "football_matches": "football-data/premier-league",
            "player_stats": "football-data/player-statistics",
            "team_performance": "football-data/team-metrics"
        }
    
    async def query_model(self, model_name: str, inputs: Dict[str, Any], 
                         timeout: int = 30) -> Optional[Dict]:
        """Query a Hugging Face model."""
        if not self.settings.HUGGINGFACE_API_KEY:
            logger.warning("Hugging Face API key not configured")
            return None
        
        try:
            cache_key = f"hf_model:{model_name}:{hash(str(inputs))}"
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                return cached_result
            
            url = f"{self.api_url}/models/{model_name}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=self.headers,
                    json=inputs,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        await self.cache.set(cache_key, result, ttl=3600)
                        return result
                    elif response.status == 503:
                        logger.warning(f"Model {model_name} is loading, please wait")
                        return {"status": "loading"}
                    else:
                        error_text = await response.text()
                        logger.error(f"HF API error {response.status}: {error_text}")
                        return None
                        
        except asyncio.TimeoutError:
            logger.error(f"Timeout querying model {model_name}")
            return None
        except Exception as e:
            logger.error(f"Error querying HF model {model_name}: {e}")
            return None
    
    async def get_football_insights(self, match_description: str) -> Optional[str]:
        """Get football insights using text generation model."""
        try:
            prompt = f"""
            Analyze this football match and provide insights:
            {match_description}
            
            Provide analysis on:
            1. Team strengths and weaknesses
            2. Key players to watch
            3. Tactical considerations
            4. Prediction factors
            
            Analysis:"""
            
            inputs = {
                "inputs": prompt,
                "parameters": {
                    "max_length": 200,
                    "temperature": 0.7,
                    "do_sample": True
                }
            }
            
            result = await self.query_model(self.models["football_prediction"], inputs)
            
            if result and isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
                # Extract only the analysis part
                if "Analysis:" in generated_text:
                    analysis = generated_text.split("Analysis:")[-1].strip()
                    return analysis
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting football insights: {e}")
            return None
    
    async def analyze_team_sentiment(self, team_news: List[str]) -> Dict[str, float]:
        """Analyze sentiment of team news and social media."""
        try:
            sentiments = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
            
            for news in team_news[:5]:  # Limit to 5 news items
                inputs = {"inputs": news}
                result = await self.query_model(self.models["sentiment_analysis"], inputs)
                
                if result and isinstance(result, list) and len(result) > 0:
                    for sentiment_data in result[0]:
                        label = sentiment_data.get("label", "").lower()
                        score = sentiment_data.get("score", 0.0)
                        
                        if "positive" in label:
                            sentiments["positive"] += score
                        elif "negative" in label:
                            sentiments["negative"] += score
                        else:
                            sentiments["neutral"] += score
            
            # Normalize scores
            total_score = sum(sentiments.values())
            if total_score > 0:
                for key in sentiments:
                    sentiments[key] /= total_score
            
            return sentiments
            
        except Exception as e:
            logger.error(f"Error analyzing team sentiment: {e}")
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
    
    async def get_similar_matches(self, match_features: Dict[str, float]) -> List[Dict]:
        """Find similar historical matches using embeddings."""
        try:
            # This would typically use a similarity search model
            # For now, we'll simulate with basic logic
            
            cache_key = f"similar_matches:{hash(str(match_features))}"
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                return cached_result
            
            # Simulate similar matches based on features
            similar_matches = []
            
            for i in range(3):  # Return 3 similar matches
                similar_match = {
                    "match_id": f"sim_{i+1}",
                    "teams": f"Team A vs Team B",
                    "score": f"{np.random.randint(0, 4)}-{np.random.randint(0, 4)}",
                    "similarity_score": np.random.uniform(0.7, 0.95),
                    "date": "2024-01-01",
                    "features": {
                        "home_strength": np.random.uniform(0.3, 0.8),
                        "away_strength": np.random.uniform(0.3, 0.8),
                        "form_difference": np.random.uniform(-0.5, 0.5)
                    }
                }
                similar_matches.append(similar_match)
            
            # Sort by similarity score
            similar_matches.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            await self.cache.set(cache_key, similar_matches, ttl=7200)
            return similar_matches
            
        except Exception as e:
            logger.error(f"Error finding similar matches: {e}")
            return []
    
    async def enhance_prediction_with_context(self, base_prediction: Dict, 
                                            context_data: Dict) -> Dict:
        """Enhance predictions using contextual information."""
        try:
            # Create context prompt
            context_prompt = f"""
            Base prediction: {json.dumps(base_prediction, indent=2)}
            Context: {json.dumps(context_data, indent=2)}
            
            Enhance this football prediction by considering:
            1. Recent team form and injuries
            2. Historical head-to-head results
            3. Weather and venue conditions
            4. Tactical matchups
            
            Provide enhanced prediction with confidence adjustments:
            """
            
            inputs = {
                "inputs": context_prompt,
                "parameters": {
                    "max_length": 150,
                    "temperature": 0.5
                }
            }
            
            result = await self.query_model(self.models["football_prediction"], inputs)
            
            enhanced_prediction = base_prediction.copy()
            
            if result and isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
                
                # Extract confidence adjustments (simplified)
                if "confidence" in generated_text.lower():
                    # Adjust confidence based on context
                    if "high confidence" in generated_text.lower():
                        enhanced_prediction["confidence_adjustment"] = 0.1
                    elif "low confidence" in generated_text.lower():
                        enhanced_prediction["confidence_adjustment"] = -0.1
                    else:
                        enhanced_prediction["confidence_adjustment"] = 0.0
                
                enhanced_prediction["context_analysis"] = generated_text
            
            return enhanced_prediction
            
        except Exception as e:
            logger.error(f"Error enhancing prediction with context: {e}")
            return base_prediction
    
    async def get_player_performance_prediction(self, player_data: Dict) -> Dict:
        """Predict player performance using ML models."""
        try:
            # Simulate player performance prediction
            # In real implementation, this would use specialized models
            
            performance_prediction = {
                "goals_probability": np.random.uniform(0.1, 0.8),
                "assists_probability": np.random.uniform(0.05, 0.6),
                "cards_probability": np.random.uniform(0.1, 0.4),
                "minutes_expected": np.random.randint(60, 90),
                "performance_rating": np.random.uniform(6.0, 9.0),
                "confidence": np.random.uniform(0.6, 0.9)
            }
            
            return performance_prediction
            
        except Exception as e:
            logger.error(f"Error predicting player performance: {e}")
            return {
                "goals_probability": 0.3,
                "assists_probability": 0.2,
                "cards_probability": 0.2,
                "minutes_expected": 75,
                "performance_rating": 7.0,
                "confidence": 0.5
            }
    
    async def get_tactical_analysis(self, team1_style: str, team2_style: str) -> Dict:
        """Analyze tactical matchup between teams."""
        try:
            prompt = f"""
            Tactical Analysis:
            Team 1 plays with: {team1_style}
            Team 2 plays with: {team2_style}
            
            Analyze the tactical matchup and predict:
            1. Which team has tactical advantage
            2. Key battles on the pitch
            3. Expected game flow
            4. Tactical adjustments needed
            
            Analysis:"""
            
            inputs = {
                "inputs": prompt,
                "parameters": {
                    "max_length": 180,
                    "temperature": 0.6
                }
            }
            
            result = await self.query_model(self.models["football_prediction"], inputs)
            
            tactical_analysis = {
                "advantage": "neutral",
                "key_battles": [],
                "game_flow": "balanced",
                "confidence": 0.7
            }
            
            if result and isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
                tactical_analysis["full_analysis"] = generated_text
                
                # Extract key insights (simplified)
                if "team 1" in generated_text.lower() and "advantage" in generated_text.lower():
                    tactical_analysis["advantage"] = "team1"
                elif "team 2" in generated_text.lower() and "advantage" in generated_text.lower():
                    tactical_analysis["advantage"] = "team2"
            
            return tactical_analysis
            
        except Exception as e:
            logger.error(f"Error getting tactical analysis: {e}")
            return {
                "advantage": "neutral",
                "key_battles": ["Midfield control", "Wing play"],
                "game_flow": "balanced",
                "confidence": 0.5,
                "full_analysis": "Tactical analysis unavailable"
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Hugging Face API health."""
        try:
            # Simple test query
            test_inputs = {"inputs": "Hello, this is a test."}
            result = await self.query_model("distilbert-base-uncased", test_inputs, timeout=10)
            
            return {
                "status": "healthy" if result is not None else "unhealthy",
                "api_accessible": result is not None,
                "models_available": len(self.models),
                "last_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"HF health check failed: {e}")
            return {
                "status": "unhealthy",
                "api_accessible": False,
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }

# Global instance
hf_integration = HuggingFaceIntegration()