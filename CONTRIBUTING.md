# 🤝 Contributing to Football Match Predictor

Thank you for your interest in contributing to the Football Match Predictor project! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)
- [Community](#community)

## Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment include:

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team. All complaints will be reviewed and investigated promptly and fairly.

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- Python 3.11 or higher
- Git
- Redis server
- Required API keys (API-Football, OpenRouter, Telegram Bot)
- Basic understanding of FastAPI, async/await, and machine learning concepts

### First Contribution

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/football-match-predictor.git
   cd football-match-predictor
   ```
3. **Set up development environment** (see [Development Setup](#development-setup))
4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
5. **Make your changes** and commit them
6. **Push to your fork** and create a pull request

## Development Setup

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

### Running the Application

```bash
# Start Redis
redis-server

# Run the application
python -m app.main

# Or with auto-reload for development
uvicorn app.main:app --reload
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_workflow.py -v
```

## Contributing Guidelines

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix issues in existing code
- **New features**: Add new functionality
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve test coverage
- **Performance**: Optimize existing code
- **Refactoring**: Improve code structure without changing functionality

### Contribution Workflow

1. **Check existing issues** to see if your contribution is already being worked on
2. **Create an issue** for new features or significant changes to discuss the approach
3. **Fork and clone** the repository
4. **Create a feature branch** from `develop`
5. **Make your changes** following our coding standards
6. **Write tests** for new functionality
7. **Update documentation** as needed
8. **Submit a pull request** with a clear description

### Branch Naming Convention

Use descriptive branch names with prefixes:

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `test/` - Test improvements
- `refactor/` - Code refactoring
- `perf/` - Performance improvements

Examples:
- `feature/corner-predictions`
- `fix/redis-connection-timeout`
- `docs/api-documentation-update`

## Pull Request Process

### Before Submitting

1. **Ensure all tests pass**:
   ```bash
   pytest
   ```

2. **Run code quality checks**:
   ```bash
   black app/ tests/
   isort app/ tests/
   flake8 app/ tests/
   mypy app/
   ```

3. **Update documentation** if needed

4. **Add tests** for new functionality

### Pull Request Template

When creating a pull request, please include:

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or breaking changes documented)
```

### Review Process

1. **Automated checks** must pass (CI/CD pipeline)
2. **Code review** by at least one maintainer
3. **Address feedback** if requested
4. **Approval** from maintainer
5. **Merge** to target branch

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **Imports**: Sorted with isort
- **Type hints**: Required for all functions
- **Docstrings**: Google style for all public functions

### Code Formatting

We use automated tools for consistent formatting:

```bash
# Format code
black app/ tests/

# Sort imports
isort app/ tests/

# Lint code
flake8 app/ tests/

# Type check
mypy app/
```

### Naming Conventions

- **Variables and functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`
- **Files and modules**: `snake_case`

### Example Code Style

```python
from typing import List, Optional
import asyncio
from app.utils.logger import get_logger

logger = get_logger(__name__)

class MatchPredictor:
    \"\"\"Predicts football match outcomes using ML models.
    
    This class provides methods for generating various types of predictions
    including score, goal scorers, and match statistics.
    \"\"\"
    
    def __init__(self, confidence_threshold: float = 0.6) -> None:
        \"\"\"Initialize the predictor with configuration.
        
        Args:
            confidence_threshold: Minimum confidence for predictions.
        \"\"\"
        self.confidence_threshold = confidence_threshold
        self._model_cache: dict = {}
    
    async def predict_score(
        self, 
        match_id: int, 
        use_cache: bool = True
    ) -> Optional[ScorePrediction]:
        \"\"\"Predict the final score of a match.
        
        Args:
            match_id: Unique identifier for the match.
            use_cache: Whether to use cached predictions.
            
        Returns:
            Score prediction or None if confidence is too low.
            
        Raises:
            ValueError: If match_id is invalid.
            PredictionError: If prediction generation fails.
        \"\"\"
        if match_id <= 0:
            raise ValueError(f\"Invalid match ID: {match_id}\")
        
        try:
            # Implementation here
            logger.info(f\"Generating score prediction for match {match_id}\")
            # ...
            return prediction
        except Exception as e:
            logger.error(f\"Failed to predict score for match {match_id}: {e}\")
            raise PredictionError(f\"Score prediction failed: {e}\") from e
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for API endpoints
├── fixtures/       # Test data and fixtures
└── conftest.py     # Shared test configuration
```

### Writing Tests

#### Unit Test Example

```python
import pytest
from unittest.mock import AsyncMock, patch
from app.services.analyzer import MLAnalyzer

class TestMLAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return MLAnalyzer()
    
    @pytest.mark.asyncio
    async def test_predict_score_success(self, analyzer):
        \"\"\"Test successful score prediction.\"\"\"
        # Arrange
        match_data = create_test_match_data()
        
        # Act
        result = await analyzer.predict_score(match_data)
        
        # Assert
        assert result is not None
        assert result.confidence > 0.5
        assert result.home_score >= 0
        assert result.away_score >= 0
    
    @pytest.mark.asyncio
    async def test_predict_score_low_confidence(self, analyzer):
        \"\"\"Test prediction with low confidence.\"\"\"
        # Test implementation
        pass
```

#### Integration Test Example

```python
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_get_match_predictions():
    \"\"\"Test getting match predictions via API.\"\"\"
    async with AsyncClient(app=app, base_url=\"http://test\") as client:
        response = await client.get(\"/matches/12345/predictions\")
    
    assert response.status_code == 200
    data = response.json()
    assert \"score_prediction\" in data
    assert \"confidence_score\" in data
```

### Test Coverage

- Maintain **minimum 80% test coverage**
- Focus on **critical business logic**
- Include **edge cases and error scenarios**
- Test **both success and failure paths**

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=app --cov-report=html

# Run specific test category
pytest tests/unit/ -v

# Run tests matching pattern
pytest -k \"test_predict\" -v
```

## Documentation

### Types of Documentation

1. **Code Documentation**: Docstrings and comments
2. **API Documentation**: OpenAPI/Swagger specs
3. **User Documentation**: README, guides, tutorials
4. **Developer Documentation**: Architecture, setup guides

### Docstring Standards

Use Google-style docstrings:

```python
def calculate_prediction_confidence(
    features: Dict[str, float],
    model_accuracy: float
) -> float:
    \"\"\"Calculate confidence score for a prediction.
    
    This function combines feature reliability scores with model accuracy
    to produce an overall confidence score for the prediction.
    
    Args:
        features: Dictionary of feature names to reliability scores.
        model_accuracy: Historical accuracy of the prediction model.
        
    Returns:
        Confidence score between 0.0 and 1.0.
        
    Raises:
        ValueError: If model_accuracy is not between 0 and 1.
        
    Example:
        >>> features = {\"home_form\": 0.8, \"away_form\": 0.6}
        >>> confidence = calculate_prediction_confidence(features, 0.75)
        >>> print(f\"Confidence: {confidence:.2f}\")
        Confidence: 0.72
    \"\"\"
    # Implementation here
```

### API Documentation

- Keep OpenAPI specs updated
- Include request/response examples
- Document error codes and responses
- Provide usage examples

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

1. **Clear title** describing the issue
2. **Steps to reproduce** the bug
3. **Expected behavior**
4. **Actual behavior**
5. **Environment information** (OS, Python version, etc.)
6. **Error messages** and stack traces
7. **Screenshots** if applicable

### Feature Requests

For feature requests, please include:

1. **Clear description** of the feature
2. **Use case** and motivation
3. **Proposed implementation** (if you have ideas)
4. **Alternatives considered**
5. **Additional context**

### Issue Templates

Use the provided issue templates when creating new issues:

- Bug Report Template
- Feature Request Template
- Documentation Issue Template
- Performance Issue Template

## Community

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas
- **Pull Requests**: Code contributions, reviews

### Getting Help

If you need help:

1. **Check existing documentation** first
2. **Search existing issues** for similar problems
3. **Create a new issue** with detailed information
4. **Be patient and respectful** when asking for help

### Recognition

Contributors are recognized in:

- **README.md**: Major contributors listed
- **CHANGELOG.md**: Contributors mentioned in releases
- **GitHub**: Contributor statistics and graphs

## Development Workflow

### Git Workflow

We use a modified Git Flow:

```
main (production)
├── develop (integration)
│   ├── feature/new-feature
│   ├── feature/another-feature
│   └── hotfix/critical-fix
└── release/v1.2.0
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(analyzer): add corner prediction model
fix(cache): resolve Redis connection timeout
docs(api): update endpoint documentation
test(workflow): add integration tests for prediction flow
```

### Release Process

1. **Create release branch** from develop
2. **Update version numbers** and changelog
3. **Final testing** and bug fixes
4. **Merge to main** and tag release
5. **Deploy to production**
6. **Merge back to develop**

## Thank You

Thank you for contributing to the Football Match Predictor project! Your contributions help make this project better for everyone. We appreciate your time and effort in improving the codebase, documentation, and community.

---

*For questions about contributing, please create an issue or reach out to the maintainers.*