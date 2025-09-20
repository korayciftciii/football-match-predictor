# Requirements Document

## Introduction

This document outlines the requirements for a football match prediction system that collects daily match data, analyzes player statistics, uses machine learning to predict scores and various statistics, and delivers results to users through a Telegram bot. The system serves as an MVP that combines data collection, ML predictions, AI-generated summaries, and user interaction through a messaging interface.

## Requirements

### Requirement 1

**User Story:** As a football fan, I want to receive match predictions for today's games through a Telegram bot, so that I can make informed decisions about matches I'm interested in.

#### Acceptance Criteria

1. WHEN a user sends /start to the Telegram bot THEN the system SHALL respond with "Bugün hangi maç için tahmin istiyorsunuz?" and display available matches
2. WHEN a user selects a specific match THEN the system SHALL provide predictions including score, goal scorer, yellow cards, corners, and first half result
3. WHEN predictions are generated THEN the system SHALL include an AI-generated summary explaining the reasoning behind predictions
4. IF no matches are available for the day THEN the system SHALL inform the user that no matches are scheduled

### Requirement 2

**User Story:** As a system administrator, I want the system to automatically collect match data from external APIs, so that predictions are based on current and accurate information.

#### Acceptance Criteria

1. WHEN the system starts THEN it SHALL fetch today's matches from API-Football
2. WHEN collecting match data THEN the system SHALL retrieve team information, player statistics, and recent match history
3. WHEN fetching team data THEN the system SHALL collect last 5 match performances for each team
4. IF API requests fail THEN the system SHALL log errors and retry with exponential backoff
5. WHEN data is successfully collected THEN the system SHALL cache results to minimize API calls

### Requirement 3

**User Story:** As a data analyst, I want the system to extract meaningful features from raw match data, so that ML models can make accurate predictions.

#### Acceptance Criteria

1. WHEN processing team data THEN the system SHALL calculate average goals scored per match
2. WHEN processing team data THEN the system SHALL calculate average goals conceded per match
3. WHEN processing player data THEN the system SHALL calculate goal contributions (goals + assists) from recent matches
4. WHEN processing team data THEN the system SHALL calculate average yellow cards per match
5. WHEN processing team data THEN the system SHALL differentiate between home and away performance statistics
6. WHEN processing match data THEN the system SHALL calculate average corner kicks per match

### Requirement 4

**User Story:** As a prediction system, I want to use machine learning models to generate accurate match predictions, so that users receive reliable forecasts.

#### Acceptance Criteria

1. WHEN generating score predictions THEN the system SHALL use Logistic Regression to predict likely scores (0-1, 1-1, 2-1, etc.)
2. WHEN predicting goal scorers THEN the system SHALL use probability models based on recent goal-scoring performance
3. WHEN predicting yellow cards THEN the system SHALL calculate probabilities based on team averages
4. WHEN predicting corners THEN the system SHALL use statistical averages from historical data
5. WHEN predicting first half results THEN the system SHALL use statistical ratios from team performance
6. IF insufficient data exists for predictions THEN the system SHALL use default probability distributions

### Requirement 5

**User Story:** As a user, I want to receive human-readable explanations of predictions, so that I can understand the reasoning behind the forecasts.

#### Acceptance Criteria

1. WHEN ML predictions are complete THEN the system SHALL generate AI summaries using OpenRouter integration
2. WHEN generating summaries THEN the system SHALL use statistical data to explain prediction reasoning
3. WHEN creating explanations THEN the system SHALL write in natural, user-friendly Turkish language
4. WHEN formatting responses THEN the system SHALL include specific statistics that support each prediction
5. IF AI summary generation fails THEN the system SHALL provide basic statistical summaries as fallback

### Requirement 6

**User Story:** As a system operator, I want the system to be built with FastAPI and proper configuration management, so that it's maintainable and scalable.

#### Acceptance Criteria

1. WHEN the system starts THEN it SHALL initialize FastAPI with proper routing structure
2. WHEN handling configuration THEN the system SHALL load API keys, tokens, and settings from environment variables
3. WHEN processing requests THEN the system SHALL implement proper error handling and logging
4. WHEN caching data THEN the system SHALL optionally use Redis for performance optimization
5. WHEN serving API endpoints THEN the system SHALL provide match data and analysis endpoints
6. IF environment variables are missing THEN the system SHALL fail gracefully with clear error messages

### Requirement 7

**User Story:** As a developer, I want the system to follow proper software architecture patterns, so that the codebase is maintainable and extensible.

#### Acceptance Criteria

1. WHEN organizing code THEN the system SHALL separate concerns into services, routes, models, and utilities
2. WHEN defining data structures THEN the system SHALL use Pydantic models for type safety
3. WHEN implementing services THEN the system SHALL create separate modules for data fetching, ML models, analysis, and AI integration
4. WHEN handling external integrations THEN the system SHALL implement proper abstraction layers
5. WHEN managing dependencies THEN the system SHALL define all requirements in requirements.txt with specific versions