"""Data fetching service for API-Football integration."""

import asyncio
import json
from datetime import datetime, date
from typing import List, Optional, Dict, Any
import httpx
from asyncio_throttle import Throttler

from app.config import settings
from app.models.schemas import (
    Match, Team, TeamStats, PlayerStats, MatchStatus, 
    MatchResult, PlayerPosition, PerformanceMetrics
)
from app.utils.logger import get_logger
from app.utils.cache import (
    cached, cache_team_stats, get_cached_team_stats,
    cache_todays_matches, get_cached_todays_matches
)
from app.utils.circuit_breaker import circuit_breaker, API_FOOTBALL_CIRCUIT

logger = get_logger(__name__)


class APIFootballError(Exception):
    """Custom exception for API-Football errors."""
    pass


class FootballDataFetcher:
    """Service for fetching football data from API-Football."""
    
    def __init__(self):
        self.base_url = "https://v3.football.api-sports.io"
        self.headers = {
            "X-RapidAPI-Key": settings.api_football_key,
            "X-RapidAPI-Host": "v3.football.api-sports.io"
        }
        # Rate limiting: API-Football allows 100 requests per day for free tier
        # We'll be conservative with 1 request per 2 seconds
        self.throttler = Throttler(rate_limit=30, period=60)  # 30 requests per minute
        self.timeout = httpx.Timeout(30.0)
        
    @circuit_breaker("api_football", API_FOOTBALL_CIRCUIT)
    async def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a rate-limited request to API-Football with circuit breaker protection."""
        async with self.throttler:
            url = f"{self.base_url}/{endpoint}"
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                try:
                    logger.info(f"Making API request to: {endpoint}")
                    response = await client.get(url, headers=self.headers, params=params or {})
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    # Check API response structure
                    if not data.get("response"):
                        logger.warning(f"Empty response from API: {endpoint} with params: {params}")
                        return {"response": []}
                    
                    logger.info(f"API request successful: {endpoint}, got {len(data['response'])} items")
                    # Debug: Log first few items to see what we're getting
                    if data['response'] and len(data['response']) > 0:
                        logger.debug(f"Sample response data: {json.dumps(data['response'][0], indent=2)}")
                    return data
                    
                except httpx.HTTPStatusError as e:
                    logger.error(f"HTTP error for {endpoint}: {e.response.status_code} - {e.response.text}")
                    if e.response.status_code == 429:
                        # Rate limit exceeded, wait longer
                        await asyncio.sleep(60)
                        raise APIFootballError(f"Rate limit exceeded for {endpoint}")
                    raise APIFootballError(f"HTTP {e.response.status_code}: {e.response.text}")
                    
                except httpx.RequestError as e:
                    logger.error(f"Request error for {endpoint}: {str(e)}")
                    raise APIFootballError(f"Request failed: {str(e)}")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error for {endpoint}: {str(e)}")
                    raise APIFootballError(f"Invalid JSON response: {str(e)}")

    @cached(ttl=21600, key_prefix="daily_matches")  # Cache for 6 hours (matches don't change during day)
    async def get_todays_matches(self, league_ids: List[int] = None, include_nearby_dates: bool = True) -> List[Match]:
        """
        Fetch today's matches from API-Football.
        
        NOTE: Free API plan only allows dates from 2025-09-18 to 2025-09-20
        and seasons from 2021 to 2023.
        
        Args:
            league_ids: Optional list of league IDs to filter matches
            include_nearby_dates: If True and no matches today, try nearby dates
            
        Returns:
            List of Match objects for today (or nearby dates if no matches today)
        """
        try:
            # Use the actual current date
            from datetime import date as date_class
            today = date_class.today().strftime("%Y-%m-%d")
            logger.info(f"Fetching matches for date: {today}")
            
            # Check if today is within the allowed date range for free plan
            allowed_dates = ["2025-09-18", "2025-09-19", "2025-09-20"]
            if today not in allowed_dates:
                logger.warning(f"Date {today} not in allowed range for free plan: {allowed_dates}")
                # Use the closest allowed date
                today = "2025-09-19"  # Use middle date as default
                logger.info(f"Using allowed date instead: {today}")
            
            params = {"date": today}
            
            # If specific leagues are requested, use them
            if league_ids:
                all_matches = []
                for league_id in league_ids:
                    params["league"] = league_id
                    data = await self._make_request("fixtures", params)
                    matches = self._parse_matches(data["response"])
                    all_matches.extend(matches)
                logger.info(f"Found {len(all_matches)} matches for requested leagues")
                return all_matches
            
            # Default: Try Turkish Süper Lig first (League ID: 203)
            logger.info(f"Fetching Turkish Süper Lig matches for date {today}")
            params["league"] = 203  # Turkish Süper Lig
            data = await self._make_request("fixtures", params)
            turkish_matches = self._parse_matches(data["response"])
            
            if turkish_matches:
                logger.info(f"Found {len(turkish_matches)} Turkish Süper Lig matches")
                return turkish_matches
            
            # If no Turkish matches, get all matches and filter
            logger.info("No Turkish Süper Lig matches found, fetching all matches")
            params.pop("league", None)  # Remove league filter
            data = await self._make_request("fixtures", params)
            all_matches = self._parse_matches(data["response"])
            
            logger.info(f"Found {len(all_matches)} total matches for {today}")
            
            # If no specific leagues requested, filter for Turkish Süper Lig only
            turkish_keywords = [
                'süper lig', 'super lig', 'trendyol süper lig', 'turkish super league'
            ]
            
            turkish_matches = []
            
            for match in all_matches:
                league_lower = match.league.lower()
                if any(keyword in league_lower for keyword in turkish_keywords):
                    turkish_matches.append(match)
            
            logger.info(f"Found {len(turkish_matches)} Turkish Süper Lig matches")
            
            # If no Turkish matches found, return a few popular matches as fallback
            if not turkish_matches:
                logger.warning("No Turkish Süper Lig matches found, using fallback matches")
                popular_keywords = [
                    'premier league', 'la liga', 'bundesliga', 'serie a', 'ligue 1',
                    'champions league', 'europa league', 'libertadores', 'sudamericana'
                ]
                
                fallback_matches = []
                for match in all_matches:
                    league_lower = match.league.lower()
                    if any(keyword in league_lower for keyword in popular_keywords):
                        fallback_matches.append(match)
                
                return fallback_matches[:5]  # Return only 5 fallback matches
            
            return turkish_matches
                
        except Exception as e:
            logger.error(f"Error fetching today's matches: {str(e)}")
            raise APIFootballError(f"Failed to fetch today's matches: {str(e)}")
    
    async def _get_nearby_matches(self) -> List[Match]:
        """Get matches from nearby dates if today has no matches."""
        from datetime import timedelta
        
        nearby_matches = []
        
        # For free plan, only these dates are allowed
        allowed_dates = ["2025-09-18", "2025-09-19", "2025-09-20"]
        current_date = date.today().strftime("%Y-%m-%d")
        
        # Try other allowed dates
        for target_date in allowed_dates:
            if target_date == current_date:
                continue  # Skip current date as we already tried it
                
            try:
                logger.info(f"Trying nearby date: {target_date}")
                params = {"date": target_date}
                data = await self._make_request("fixtures", params)
                matches = self._parse_matches(data["response"])
                
                if matches:
                    logger.info(f"Found {len(matches)} matches on {target_date}")
                    nearby_matches.extend(matches[:15])  # Limit per date
                    
                    # Stop after finding some matches
                    if len(nearby_matches) >= 20:
                        return nearby_matches[:20]
                        
            except Exception as e:
                logger.warning(f"Error checking date {target_date}: {e}")
                continue
        
        return nearby_matches

    @cached(ttl=86400, key_prefix="team_stats")  # Cache for 24 hours (team stats don't change daily)
    async def get_team_stats(self, team_id: int, season: int = 2023) -> TeamStats:
        """
        Fetch team statistics from API-Football.
        
        NOTE: Free API plan only allows seasons from 2021 to 2023.
        
        Args:
            team_id: Team ID
            season: Season year (default: 2023, max allowed for free plan)
            
        Returns:
            TeamStats object
        """
        try:
            # Ensure season is within allowed range for free plan
            if season > 2023:
                logger.warning(f"Season {season} not allowed for free plan, using 2023 instead")
                season = 2023
            elif season < 2021:
                logger.warning(f"Season {season} not allowed for free plan, using 2021 instead")
                season = 2021
                
            params = {"team": team_id, "season": season}
            data = await self._make_request("teams/statistics", params)
            
            if not data["response"]:
                raise APIFootballError(f"No statistics found for team {team_id}")
            
            stats_data = data["response"]
            return self._parse_team_stats(team_id, stats_data)
            
        except Exception as e:
            logger.error(f"Error fetching team stats for {team_id}: {str(e)}")
            raise APIFootballError(f"Failed to fetch team stats: {str(e)}")

    @cached(ttl=1800, key_prefix="recent_matches")  # Cache for 30 minutes
    async def get_recent_matches(self, team_id: int, limit: int = 5) -> List[Match]:
        """
        Fetch recent matches for a team.
        
        Args:
            team_id: Team ID
            limit: Number of recent matches to fetch
            
        Returns:
            List of recent Match objects
        """
        try:
            params = {"team": team_id, "last": limit}
            data = await self._make_request("fixtures", params)
            
            matches = self._parse_matches(data["response"])
            return matches[:limit]  # Ensure we don't exceed the limit
            
        except Exception as e:
            logger.error(f"Error fetching recent matches for team {team_id}: {str(e)}")
            raise APIFootballError(f"Failed to fetch recent matches: {str(e)}")

    @cached(ttl=86400, key_prefix="player_stats")  # Cache for 24 hours (player stats don't change daily)
    async def get_player_stats(self, team_id: int, season: int = 2023) -> List[PlayerStats]:
        """
        Fetch player statistics for a team.
        
        NOTE: Free API plan only allows seasons from 2021 to 2023.
        
        Args:
            team_id: Team ID
            season: Season year (default: 2023, max allowed for free plan)
            
        Returns:
            List of PlayerStats objects
        """
        try:
            # Ensure season is within allowed range for free plan
            if season > 2023:
                logger.warning(f"Season {season} not allowed for free plan, using 2023 instead")
                season = 2023
            elif season < 2021:
                logger.warning(f"Season {season} not allowed for free plan, using 2021 instead")
                season = 2021
                
            params = {"team": team_id, "season": season}
            data = await self._make_request("players", params)
            
            players = []
            for player_data in data["response"]:
                try:
                    player_stats = self._parse_player_stats(player_data)
                    if player_stats:
                        players.append(player_stats)
                except Exception as e:
                    logger.warning(f"Error parsing player data: {e}")
                    continue
            
            return players
            
        except Exception as e:
            logger.error(f"Error fetching player stats for team {team_id}: {str(e)}")
            raise APIFootballError(f"Failed to fetch player stats: {str(e)}")

    def _parse_matches(self, matches_data: List[Dict[str, Any]]) -> List[Match]:
        """Parse API response data into Match objects."""
        matches = []
        
        for match_data in matches_data:
            try:
                # Parse teams
                home_team = Team(
                    id=match_data["teams"]["home"]["id"],
                    name=match_data["teams"]["home"]["name"],
                    logo_url=match_data["teams"]["home"]["logo"]
                )
                
                away_team = Team(
                    id=match_data["teams"]["away"]["id"],
                    name=match_data["teams"]["away"]["name"],
                    logo_url=match_data["teams"]["away"]["logo"]
                )
                
                # Parse match status
                status_short = match_data["fixture"]["status"]["short"]
                match_status = self._map_match_status(status_short)
                
                # Parse kickoff time
                kickoff_str = match_data["fixture"]["date"]
                if kickoff_str.endswith('Z'):
                    kickoff_time = datetime.fromisoformat(kickoff_str.replace('Z', '+00:00'))
                else:
                    kickoff_time = datetime.fromisoformat(kickoff_str)
                
                match = Match(
                    id=match_data["fixture"]["id"],
                    home_team=home_team,
                    away_team=away_team,
                    kickoff_time=kickoff_time,
                    league=match_data["league"]["name"],
                    status=match_status,
                    venue=match_data["fixture"]["venue"]["name"] if match_data["fixture"]["venue"] else None,
                    referee=match_data["fixture"]["referee"]
                )
                
                matches.append(match)
                
            except Exception as e:
                logger.warning(f"Error parsing match data: {e}")
                continue
        
        return matches

    def _parse_team_stats(self, team_id: int, stats_data: Dict[str, Any]) -> TeamStats:
        """Parse API team statistics into TeamStats object."""
        try:
            fixtures = stats_data["fixtures"]
            goals = stats_data["goals"]
            cards = stats_data["cards"]
            
            # Parse home performance
            home_performance = PerformanceMetrics(
                matches_played=fixtures["played"]["home"],
                wins=fixtures["wins"]["home"],
                draws=fixtures["draws"]["home"],
                losses=fixtures["loses"]["home"],
                goals_scored=goals["for"]["total"]["home"],
                goals_conceded=goals["against"]["total"]["home"],
                clean_sheets=stats_data["clean_sheet"]["home"]
            )
            
            # Parse away performance
            away_performance = PerformanceMetrics(
                matches_played=fixtures["played"]["away"],
                wins=fixtures["wins"]["away"],
                draws=fixtures["draws"]["away"],
                losses=fixtures["loses"]["away"],
                goals_scored=goals["for"]["total"]["away"],
                goals_conceded=goals["against"]["total"]["away"],
                clean_sheets=stats_data["clean_sheet"]["away"]
            )
            
            # Calculate averages
            total_matches = fixtures["played"]["total"]
            goals_scored_avg = goals["for"]["average"]["total"] if goals["for"]["average"]["total"] else 0.0
            goals_conceded_avg = goals["against"]["average"]["total"] if goals["against"]["average"]["total"] else 0.0
            
            # Calculate yellow cards average (approximate from total)
            yellow_cards_total = cards["yellow"]["0-15"]["total"] + cards["yellow"]["16-30"]["total"] + \
                               cards["yellow"]["31-45"]["total"] + cards["yellow"]["46-60"]["total"] + \
                               cards["yellow"]["61-75"]["total"] + cards["yellow"]["76-90"]["total"] + \
                               cards["yellow"]["91-105"]["total"] + cards["yellow"]["106-120"]["total"]
            
            yellow_cards_avg = yellow_cards_total / total_matches if total_matches > 0 else 0.0
            
            # Estimate corners average (API-Football doesn't provide this in basic stats)
            corners_avg = 5.0  # Default estimate, will be improved with more data
            
            return TeamStats(
                team_id=team_id,
                goals_scored_avg=goals_scored_avg,
                goals_conceded_avg=goals_conceded_avg,
                yellow_cards_avg=yellow_cards_avg,
                corners_avg=corners_avg,
                home_performance=home_performance,
                away_performance=away_performance,
                recent_matches_count=5
            )
            
        except Exception as e:
            logger.error(f"Error parsing team stats: {e}")
            raise APIFootballError(f"Failed to parse team statistics: {str(e)}")

    def _parse_player_stats(self, player_data: Dict[str, Any]) -> Optional[PlayerStats]:
        """Parse API player data into PlayerStats object."""
        try:
            player_info = player_data["player"]
            statistics = player_data["statistics"]
            
            if not statistics:
                return None
            
            # Use the first statistics entry (current season)
            stats = statistics[0]
            
            # Map position
            position_map = {
                "Goalkeeper": PlayerPosition.GOALKEEPER,
                "Defender": PlayerPosition.DEFENDER,
                "Midfielder": PlayerPosition.MIDFIELDER,
                "Attacker": PlayerPosition.ATTACKER
            }
            
            position = position_map.get(stats["games"]["position"], PlayerPosition.MIDFIELDER)
            
            # Get goals and assists
            goals = stats["goals"]["total"] or 0
            assists = stats["goals"]["assists"] or 0
            minutes = stats["games"]["minutes"] or 0
            
            # Calculate goal probability based on recent performance
            appearances = stats["games"]["appearences"] or 1
            goal_probability = min(goals / appearances, 1.0) if appearances > 0 else 0.0
            
            return PlayerStats(
                player_id=player_info["id"],
                name=player_info["name"],
                position=position,
                goals_recent=goals,
                assists_recent=assists,
                minutes_played=minutes,
                goal_probability=goal_probability
            )
            
        except Exception as e:
            logger.warning(f"Error parsing player stats: {e}")
            return None

    def _map_match_status(self, status_short: str) -> MatchStatus:
        """Map API status to internal MatchStatus enum."""
        status_map = {
            "NS": MatchStatus.NOT_STARTED,
            "1H": MatchStatus.FIRST_HALF,
            "HT": MatchStatus.HALFTIME,
            "2H": MatchStatus.SECOND_HALF,
            "ET": MatchStatus.EXTRA_TIME,
            "P": MatchStatus.PENALTY,
            "FT": MatchStatus.FINISHED,
            "AET": MatchStatus.FINISHED,
            "PEN": MatchStatus.FINISHED,
            "PST": MatchStatus.POSTPONED,
            "CANC": MatchStatus.CANCELLED,
            "ABD": MatchStatus.ABANDONED
        }
        
        return status_map.get(status_short, MatchStatus.NOT_STARTED)

    @cached(ttl=604800, key_prefix="h2h")  # Cache for 7 days (historical data rarely changes)
    async def get_head_to_head(self, team1_id: int, team2_id: int, limit: int = 10) -> List[Match]:
        """
        Get head-to-head matches between two teams.
        
        Args:
            team1_id: First team ID
            team2_id: Second team ID
            limit: Number of recent H2H matches
            
        Returns:
            List of head-to-head matches
        """
        try:
            params = {"h2h": f"{team1_id}-{team2_id}", "last": limit}
            data = await self._make_request("fixtures/headtohead", params)
            
            return self._parse_matches(data["response"])
            
        except Exception as e:
            logger.error(f"Error fetching H2H for teams {team1_id} vs {team2_id}: {str(e)}")
            return []  # Return empty list on error, don't fail the whole process

    async def health_check(self) -> Dict[str, Any]:
        """Check API-Football service health."""
        try:
            # Make a simple request to check API status
            data = await self._make_request("status")
            
            # Extract response data properly
            response_data = data.get("response", {})
            if isinstance(response_data, dict):
                requests_info = response_data.get("requests", {})
                return {
                    "status": "healthy",
                    "api_requests_remaining": requests_info.get("current", 0),
                    "api_requests_limit": requests_info.get("limit_day", 100)
                }
            else:
                # If response format is different, still consider healthy if we got a response
                return {
                    "status": "healthy",
                    "api_requests_remaining": "unknown",
                    "api_requests_limit": 100
                }
            
        except Exception as e:
            logger.error(f"API-Football health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def get_cache_status(self) -> Dict[str, Any]:
        """Get cache status and statistics."""
        try:
            from app.utils.cache import cache_manager
            
            cache_info = {
                "cache_healthy": await cache_manager.is_healthy(),
                "cache_settings": {
                    "daily_matches_ttl": "6 hours",
                    "team_stats_ttl": "24 hours", 
                    "player_stats_ttl": "24 hours",
                    "h2h_ttl": "7 days",
                    "predictions_ttl": "3 hours"
                }
            }
            
            return cache_info
            
        except Exception as e:
            logger.error(f"Error getting cache status: {e}")
            return {"cache_healthy": False, "error": str(e)}