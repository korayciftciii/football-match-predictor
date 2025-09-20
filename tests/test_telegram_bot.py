"""Unit tests for Telegram bot."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import sys
import os

# Add the app directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.bots.telegram_bot import FootballPredictionBot, TelegramBotError
from app.models.schemas import (
    Match, Team, MatchPredictions, ScorePrediction, CardPrediction,
    CornerPrediction, FirstHalfPrediction, MatchStatus, MatchResult
)


class TestFootballPredictionBot:
    """Test cases for FootballPredictionBot."""
    
    @pytest.fixture
    def bot(self):
        """Create bot instance for testing."""
        with patch('app.bots.telegram_bot.settings') as mock_settings:
            mock_settings.telegram_bot_token = "test_token"
            mock_settings.telegram_chat_id = "test_chat_id"
            return FootballPredictionBot()
    
    @pytest.fixture
    def mock_update(self):
        """Create mock Telegram update."""
        update = MagicMock()
        update.effective_user.id = 12345
        update.effective_user.first_name = "Test User"
        update.message.reply_text = AsyncMock()
        update.callback_query = None
        return update
    
    @pytest.fixture
    def mock_callback_query(self):
        """Create mock callback query."""
        query = MagicMock()
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        query.data = "test_data"
        
        update = MagicMock()
        update.callback_query = query
        update.effective_user.id = 12345
        update.effective_user.first_name = "Test User"
        
        return update, query
    
    @pytest.fixture
    def sample_match(self):
        """Create sample match for testing."""
        home_team = Team(id=1, name="Arsenal", recent_form=[])
        away_team = Team(id=2, name="Chelsea", recent_form=[])
        
        return Match(
            id=12345,
            home_team=home_team,
            away_team=away_team,
            kickoff_time=datetime(2024, 1, 15, 15, 0),
            league="Premier League",
            status=MatchStatus.NOT_STARTED
        )
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for testing."""
        score_prediction = ScorePrediction(
            home_score=2,
            away_score=1,
            confidence=0.75,
            probability=0.18,
            alternative_scores=[]
        )
        
        card_prediction = CardPrediction(
            home_team_cards=2,
            away_team_cards=3,
            total_cards=5,
            confidence=0.7
        )
        
        corner_prediction = CornerPrediction(
            home_team_corners=6,
            away_team_corners=4,
            total_corners=10,
            confidence=0.65
        )
        
        first_half_prediction = FirstHalfPrediction(
            result=MatchResult.WIN,
            home_score=1,
            away_score=0,
            confidence=0.6,
            probability=0.4
        )
        
        return MatchPredictions(
            match_id=12345,
            score_prediction=score_prediction,
            goal_scorer_prediction=None,
            yellow_cards_prediction=card_prediction,
            corners_prediction=corner_prediction,
            first_half_prediction=first_half_prediction,
            ai_summary="Arsenal should win 2-1 based on recent form",
            confidence_score=0.7
        )
    
    def test_init(self, bot):
        """Test bot initialization."""
        assert bot.token == "test_token"
        assert bot.chat_id == "test_chat_id"
        assert bot.application is None
        assert bot.is_running is False
        assert isinstance(bot.user_sessions, dict)
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, bot):
        """Test successful bot initialization."""
        with patch('telegram.ext.Application.builder') as mock_builder:
            mock_app = MagicMock()
            mock_builder.return_value.token.return_value.build.return_value = mock_app
            mock_app.bot.set_my_commands = AsyncMock()
            
            await bot.initialize()
            
            assert bot.application == mock_app
            mock_app.add_handler.assert_called()  # Should add multiple handlers
            mock_app.add_error_handler.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_failure(self, bot):
        """Test bot initialization failure."""
        with patch('telegram.ext.Application.builder') as mock_builder:
            mock_builder.side_effect = Exception("Initialization failed")
            
            with pytest.raises(TelegramBotError):
                await bot.initialize()
    
    @pytest.mark.asyncio
    async def test_start_command_success(self, bot, mock_update, sample_match):
        """Test successful /start command."""
        with patch.object(bot.data_fetcher, 'get_todays_matches') as mock_get_matches:
            mock_get_matches.return_value = [sample_match]
            
            await bot.start_command(mock_update, None)
            
            # Should create user session
            assert mock_update.effective_user.id in bot.user_sessions
            
            # Should call reply_text (through _send_matches_menu)
            mock_update.message.reply_text.assert_called()
    
    @pytest.mark.asyncio
    async def test_start_command_no_matches(self, bot, mock_update):
        """Test /start command with no matches."""
        with patch.object(bot.data_fetcher, 'get_todays_matches') as mock_get_matches:
            mock_get_matches.return_value = []
            
            await bot.start_command(mock_update, None)
            
            # Should still create user session
            assert mock_update.effective_user.id in bot.user_sessions
    
    @pytest.mark.asyncio
    async def test_help_command(self, bot, mock_update):
        """Test /help command."""
        await bot.help_command(mock_update, None)
        
        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args
        assert "Futbol Tahmin Botu Yardım" in call_args[0][0]
        assert call_args[1]["parse_mode"] == 'Markdown'
    
    @pytest.mark.asyncio
    async def test_matches_command(self, bot, mock_update, sample_match):
        """Test /matches command."""
        with patch.object(bot.data_fetcher, 'get_todays_matches') as mock_get_matches:
            mock_get_matches.return_value = [sample_match]
            
            await bot.matches_command(mock_update, None)
            
            mock_get_matches.assert_called_once()
            mock_update.message.reply_text.assert_called()
    
    @pytest.mark.asyncio
    async def test_stats_command(self, bot, mock_update):
        """Test /stats command."""
        # Create user session
        bot.user_sessions[mock_update.effective_user.id] = {
            "started_at": datetime.now(),
            "last_activity": datetime.now(),
            "matches_requested": 5
        }
        
        with patch.object(bot.analyzer.ml_models, 'get_model_info') as mock_model_info:
            mock_model_info.return_value = {"models_loaded": {"score": True}}
            
            await bot.stats_command(mock_update, None)
            
            mock_update.message.reply_text.assert_called_once()
            call_args = mock_update.message.reply_text.call_args
            assert "Bot İstatistikleri" in call_args[0][0]
            assert "5" in call_args[0][0]  # matches_requested
    
    @pytest.mark.asyncio
    async def test_health_command(self, bot, mock_update):
        """Test /health command."""
        with patch.object(bot.data_fetcher, 'health_check') as mock_api_health:
            with patch.object(bot.analyzer.ai_summary, 'test_connection') as mock_ai_health:
                with patch.object(bot.analyzer.ml_models, 'get_model_info') as mock_model_info:
                    mock_api_health.return_value = {"status": "healthy"}
                    mock_ai_health.return_value = {"status": "success"}
                    mock_model_info.return_value = {"models_loaded": {"score": True}}
                    
                    await bot.health_command(mock_update, None)
                    
                    mock_update.message.reply_text.assert_called_once()
                    call_args = mock_update.message.reply_text.call_args
                    assert "Sistem Sağlık Durumu" in call_args[0][0]
                    assert "✅" in call_args[0][0]
    
    @pytest.mark.asyncio
    async def test_handle_callback_query_match_selection(self, bot, sample_match, sample_predictions):
        """Test callback query for match selection."""
        update, query = self._create_callback_query("match_12345")
        
        with patch.object(bot, '_handle_match_selection') as mock_handle:
            await bot.handle_callback_query(update, None)
            
            query.answer.assert_called_once()
            mock_handle.assert_called_once_with(query, 12345)
    
    @pytest.mark.asyncio
    async def test_handle_callback_query_refresh(self, bot):
        """Test callback query for refresh matches."""
        update, query = self._create_callback_query("refresh_matches")
        
        with patch.object(bot, '_handle_refresh_matches') as mock_handle:
            await bot.handle_callback_query(update, None)
            
            query.answer.assert_called_once()
            mock_handle.assert_called_once_with(query)
    
    @pytest.mark.asyncio
    async def test_handle_callback_query_back(self, bot):
        """Test callback query for back action."""
        update, query = self._create_callback_query("back_matches")
        
        with patch.object(bot, '_handle_back_action') as mock_handle:
            await bot.handle_callback_query(update, None)
            
            query.answer.assert_called_once()
            mock_handle.assert_called_once_with(query, "back_matches")
    
    @pytest.mark.asyncio
    async def test_handle_message_greeting(self, bot, mock_update):
        """Test handling greeting messages."""
        mock_update.message.text = "merhaba"
        
        await bot.handle_message(mock_update, None)
        
        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args
        assert "Merhaba" in call_args[0][0]
    
    @pytest.mark.asyncio
    async def test_handle_message_match_request(self, bot, mock_update, sample_match):
        """Test handling match-related messages."""
        mock_update.message.text = "maç"
        
        with patch.object(bot, 'matches_command') as mock_matches:
            await bot.handle_message(mock_update, None)
            
            mock_matches.assert_called_once_with(mock_update, None)
    
    @pytest.mark.asyncio
    async def test_handle_message_unknown(self, bot, mock_update):
        """Test handling unknown messages."""
        mock_update.message.text = "unknown message"
        
        await bot.handle_message(mock_update, None)
        
        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args
        assert "Anlayamadım" in call_args[0][0]
    
    @pytest.mark.asyncio
    async def test_send_matches_menu_success(self, bot, mock_update, sample_match):
        """Test successful matches menu sending."""
        with patch.object(bot.data_fetcher, 'get_todays_matches') as mock_get_matches:
            mock_get_matches.return_value = [sample_match]
            
            await bot._send_matches_menu(mock_update, None, "Test Header")
            
            # Should call reply_text multiple times (loading + final message)
            assert mock_update.message.reply_text.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_send_matches_menu_no_matches(self, bot, mock_update):
        """Test matches menu with no matches."""
        with patch.object(bot.data_fetcher, 'get_todays_matches') as mock_get_matches:
            mock_get_matches.return_value = []
            
            await bot._send_matches_menu(mock_update, None, "Test Header")
            
            # Should show no matches message
            mock_update.message.reply_text.assert_called()
    
    @pytest.mark.asyncio
    async def test_handle_match_selection_success(self, bot, sample_match, sample_predictions):
        """Test successful match selection handling."""
        query = MagicMock()
        query.edit_message_text = AsyncMock()
        
        with patch.object(bot.data_fetcher, 'get_todays_matches') as mock_get_matches:
            with patch.object(bot.analyzer, 'generate_predictions') as mock_predictions:
                with patch('app.bots.telegram_bot.get_cached_match_predictions') as mock_cache:
                    mock_cache.return_value = None  # No cache
                    mock_get_matches.return_value = [sample_match]
                    mock_predictions.return_value = sample_predictions
                    
                    await bot._handle_match_selection(query, 12345)
                    
                    # Should call edit_message_text multiple times
                    assert query.edit_message_text.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_handle_match_selection_cached(self, bot, sample_predictions):
        """Test match selection with cached predictions."""
        query = MagicMock()
        query.edit_message_text = AsyncMock()
        
        with patch('app.bots.telegram_bot.get_cached_match_predictions') as mock_cache:
            with patch.object(bot, '_send_predictions') as mock_send:
                mock_cache.return_value = sample_predictions
                
                await bot._handle_match_selection(query, 12345)
                
                mock_send.assert_called_once_with(query, sample_predictions)
    
    @pytest.mark.asyncio
    async def test_handle_match_selection_not_found(self, bot):
        """Test match selection for non-existent match."""
        query = MagicMock()
        query.edit_message_text = AsyncMock()
        
        with patch.object(bot.data_fetcher, 'get_todays_matches') as mock_get_matches:
            with patch('app.bots.telegram_bot.get_cached_match_predictions') as mock_cache:
                mock_cache.return_value = None
                mock_get_matches.return_value = []  # No matches
                
                await bot._handle_match_selection(query, 99999)
                
                query.edit_message_text.assert_called_with("❌ Maç bulunamadı.")
    
    @pytest.mark.asyncio
    async def test_send_predictions_success(self, bot, sample_match, sample_predictions):
        """Test successful predictions sending."""
        query = MagicMock()
        query.edit_message_text = AsyncMock()
        
        await bot._send_predictions(query, sample_predictions, sample_match)
        
        query.edit_message_text.assert_called_once()
        call_args = query.edit_message_text.call_args
        
        # Check message content
        message = call_args[0][0]
        assert "Arsenal vs Chelsea" in message
        assert "2-1" in message
        assert "Arsenal should win" in message
        
        # Check reply markup
        assert "reply_markup" in call_args[1]
    
    @pytest.mark.asyncio
    async def test_send_predictions_to_chat(self, bot, sample_match, sample_predictions):
        """Test sending predictions to specific chat."""
        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock()
        
        bot.application = MagicMock()
        bot.application.bot = mock_bot
        
        with patch.object(bot.data_fetcher, 'get_todays_matches') as mock_get_matches:
            mock_get_matches.return_value = [sample_match]
            
            await bot.send_predictions(12345, sample_predictions)
            
            mock_bot.send_message.assert_called_once()
            call_args = mock_bot.send_message.call_args
            assert call_args[1]["chat_id"] == 12345
            assert "Arsenal vs Chelsea" in call_args[1]["text"]
    
    @pytest.mark.asyncio
    async def test_send_predictions_to_chat_not_initialized(self, bot, sample_predictions):
        """Test sending predictions when bot not initialized."""
        bot.application = None
        
        with pytest.raises(TelegramBotError):
            await bot.send_predictions(12345, sample_predictions)
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, bot):
        """Test bot health check when healthy."""
        mock_bot = MagicMock()
        mock_bot_info = MagicMock()
        mock_bot_info.username = "test_bot"
        mock_bot.get_me = AsyncMock(return_value=mock_bot_info)
        
        bot.application = MagicMock()
        bot.application.bot = mock_bot
        bot.is_running = True
        
        health = await bot.health_check()
        
        assert health["status"] == "healthy"
        assert health["bot_username"] == "test_bot"
        assert health["is_running"] is True
    
    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, bot):
        """Test bot health check when not initialized."""
        bot.application = None
        
        health = await bot.health_check()
        
        assert health["status"] == "not_initialized"
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, bot):
        """Test bot health check when unhealthy."""
        mock_bot = MagicMock()
        mock_bot.get_me = AsyncMock(side_effect=Exception("Bot error"))
        
        bot.application = MagicMock()
        bot.application.bot = mock_bot
        
        health = await bot.health_check()
        
        assert health["status"] == "unhealthy"
        assert "error" in health
    
    @pytest.mark.asyncio
    async def test_error_handler(self, bot, mock_update):
        """Test error handler."""
        context = MagicMock()
        context.error = Exception("Test error")
        
        await bot.error_handler(mock_update, context)
        
        mock_update.effective_message.reply_text.assert_called_once()
        call_args = mock_update.effective_message.reply_text.call_args
        assert "hata oluştu" in call_args[0][0]
    
    def _create_callback_query(self, data: str):
        """Helper to create callback query mock."""
        query = MagicMock()
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        query.data = data
        
        update = MagicMock()
        update.callback_query = query
        update.effective_user.id = 12345
        update.effective_user.first_name = "Test User"
        
        return update, query