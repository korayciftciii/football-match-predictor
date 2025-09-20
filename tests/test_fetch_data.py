"""Unit tests for data fetching service."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, date
import httpx
import sys
import os

# Add the app directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.services.fetch_data import FootballDataFetcher, APIFootballError
from app.models.schemas import Match, Team, TeamStats, PlayerStats, MatchStatus


class TestFootballDataFetcher:
    """Test cases for FootballDataFetcher."""
    
    @pytest.fixture
    def fetcher(self):
        """Create a FootballDataFetcher instance for testing."""
        return FootballDataFetcher()
    
    @pytest.fixture
    def mock_match_response(self):
        """Mock API response for matches."""
        return {
            "response": [
                {
                    "fixture": {
                        "id": 12345,
                        "date": "2024-01-15T15:00:00+00:00",
                        "status": {"short": "NS"},
                        "venue": {"name": "Wembley Stadium"},
                        "referee": "John Doe"
                    },
                    "teams": {
                        "home": {
                            "id": 1,
                            "name": "Arsenal",
                            "logo": "https://example.com/arsenal.png"
                        },
                        "away": {
                            "id": 2,
                            "name": "Chelsea",
                            "logo": "https://example.com/chelsea.png"
                        }
                    },
                    "league": {"name": "Premier League"}
                }
            ]
        }
    
    @pytest.fixture
    def mock_team_stats_response(self):
        """Mock API response for team statistics."""
        return {
            "response": {
                "fixtures": {
                    "played": {"total": 20, "home": 10, "away": 10},
                    "wins": {"total": 12, "home": 7, "away": 5},
                    "draws": {"total": 5, "home": 2, "away": 3},
                    "loses": {"total": 3, "home": 1, "away": 2}
                },
                "goals": {
                    "for": {
                        "total": {"total": 35, "home": 20, "away": 15},
                        "average": {"total": 1.75, "home": 2.0, "away": 1.5}
                    },
                    "against": {
                        "total": {"total": 15, "home": 7, "away": 8},
                        "average": {"total": 0.75, "home": 0.7, "away": 0.8}
                    }
                },
                "cards": {
                    "yellow": {
                        "0-15": {"total": 2},
                        "16-30": {"total": 3},
                        "31-45": {"total": 4},
                        "46-60": {"total": 3},
                        "61-75": {"total": 2},
                        "76-90": {"total": 1},
                        "91-105": {"total": 0},
                        "106-120": {"total": 0}
                    }
                },
                "clean_sheet": {"home": 6, "away": 4, "total": 10}
            }
        }
    
    @pytest.fixture
    def mock_player_stats_response(self):
        """Mock API response for player statistics."""
        return {
            "response": [
                {
                    "player": {
                        "id": 101,
                        "name": "Harry Kane"
                    },
                    "statistics": [
                        {
                            "games": {
                                "appearences": 15,
                                "position": "Attacker",
                                "minutes": 1350
                            },
                            "goals": {
                                "total": 12,
                                "assists": 3
                            }
                        }
                    ]
                }
            ]
        }
    
    @pytest.mark.asyncio
    async def test_get_todays_matches_success(self, fetcher, mock_match_response):
        """Test successful fetching of today's matches."""
        with patch.object(fetcher, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_match_response
            
            matches = await fetcher.get_todays_matches([39])  # Premier League
            
            assert len(matches) == 1
            assert isinstance(matches[0], Match)
            assert matches[0].id == 12345
            assert matches[0].home_team.name == "Arsenal"
            assert matches[0].away_team.name == "Chelsea"
            assert matches[0].status == MatchStatus.NOT_STARTED
    
    @pytest.mark.asyncio
    async def test_get_team_stats_success(self, fetcher, mock_team_stats_response):
        """Test successful fetching of team statistics."""
        with patch.object(fetcher, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_team_stats_response
            
            team_stats = await fetcher.get_team_stats(1, 2024)
            
            assert isinstance(team_stats, TeamStats)
            assert team_stats.team_id == 1
            assert team_stats.goals_scored_avg == 1.75
            assert team_stats.goals_conceded_avg == 0.75
            assert team_stats.home_performance.wins == 7
            assert team_stats.away_performance.wins == 5
    
    @pytest.mark.asyncio
    async def test_get_player_stats_success(self, fetcher, mock_player_stats_response):
        """Test successful fetching of player statistics."""
        with patch.object(fetcher, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_player_stats_response
            
            players = await fetcher.get_player_stats(1, 2024)
            
            assert len(players) == 1
            assert isinstance(players[0], PlayerStats)
            assert players[0].name == "Harry Kane"
            assert players[0].goals_recent == 12
            assert players[0].assists_recent == 3
            assert players[0].goal_probability == 0.8  # 12/15 appearances
    
    @pytest.mark.asyncio
    async def test_make_request_http_error(self, fetcher):
        """Test handling of HTTP errors."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "404 Not Found", request=MagicMock(), response=MagicMock()
            )
            mock_response.status_code = 404
            mock_response.text = "Not Found"
            
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            with pytest.raises(APIFootballError):
                await fetcher._make_request("invalid-endpoint")
    
    @pytest.mark.asyncio
    async def test_make_request_rate_limit(self, fetcher):
        """Test handling of rate limit errors."""
        with patch('httpx.AsyncClient') as mock_client, \
             patch('asyncio.sleep') as mock_sleep:
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.text = "Rate limit exceeded"
            
            mock_http_error = httpx.HTTPStatusError(
                "429 Too Many Requests", 
                request=MagicMock(), 
                response=mock_response
            )
            mock_response.raise_for_status.side_effect = mock_http_error
            
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            with pytest.raises(APIFootballError, match="Rate limit exceeded"):
                await fetcher._make_request("fixtures")
            
            # Verify that sleep was called with 60 seconds
            mock_sleep.assert_called_once_with(60)
    
    @pytest.mark.asyncio
    async def test_get_recent_matches(self, fetcher, mock_match_response):
        """Test fetching recent matches for a team."""
        with patch.object(fetcher, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_match_response
            
            matches = await fetcher.get_recent_matches(1, 5)
            
            assert len(matches) == 1
            mock_request.assert_called_once_with("fixtures", {"team": 1, "last": 5})
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, fetcher):
        """Test successful health check."""
        mock_response = {
            "response": {
                "requests": {
                    "current": 50,
                    "limit_day": 100
                }
            }
        }
        
        with patch.object(fetcher, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            health = await fetcher.health_check()
            
            assert health["status"] == "healthy"
            assert health["api_requests_remaining"] == 50
            assert health["api_requests_limit"] == 100
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, fetcher):
        """Test health check failure."""
        with patch.object(fetcher, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = APIFootballError("API unavailable")
            
            health = await fetcher.health_check()
            
            assert health["status"] == "unhealthy"
            assert "API unavailable" in health["error"]
    
    def test_map_match_status(self, fetcher):
        """Test match status mapping."""
        assert fetcher._map_match_status("NS") == MatchStatus.NOT_STARTED
        assert fetcher._map_match_status("1H") == MatchStatus.FIRST_HALF
        assert fetcher._map_match_status("HT") == MatchStatus.HALFTIME
        assert fetcher._map_match_status("FT") == MatchStatus.FINISHED
        assert fetcher._map_match_status("UNKNOWN") == MatchStatus.NOT_STARTED  # Default
    
    @pytest.mark.asyncio
    async def test_get_head_to_head(self, fetcher, mock_match_response):
        """Test fetching head-to-head matches."""
        with patch.object(fetcher, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_match_response
            
            matches = await fetcher.get_head_to_head(1, 2, 5)
            
            assert len(matches) == 1
            mock_request.assert_called_once_with("fixtures/headtohead", {"h2h": "1-2", "last": 5})
    
    @pytest.mark.asyncio
    async def test_get_head_to_head_error_handling(self, fetcher):
        """Test head-to-head error handling returns empty list."""
        with patch.object(fetcher, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = APIFootballError("API error")
            
            matches = await fetcher.get_head_to_head(1, 2, 5)
            
            assert matches == []  # Should return empty list, not raise exception