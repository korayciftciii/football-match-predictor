# Multi-stage Dockerfile for Football Match Predictor
# This Dockerfile creates an optimized production image

# =============================================================================
# BUILD STAGE
# =============================================================================
FROM python:3.11-slim as builder

# Set build arguments
ARG PYTHON_VERSION=3.11
ARG APP_USER=appuser
ARG APP_UID=1000

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# =============================================================================
# PRODUCTION STAGE
# =============================================================================
FROM python:3.11-slim as production

# Set build arguments
ARG APP_USER=appuser
ARG APP_UID=1000

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create application user
RUN groupadd -r $APP_USER && \
    useradd -r -g $APP_USER -u $APP_UID -d /app -s /bin/bash $APP_USER

# Create application directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/config && \
    chown -R $APP_USER:$APP_USER /app

# Copy application code
COPY --chown=$APP_USER:$APP_USER . /app/

# Switch to non-root user
USER $APP_USER

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "app.main"]

# =============================================================================
# DEVELOPMENT STAGE (Optional)
# =============================================================================
FROM production as development

# Switch back to root for development setup
USER root

# Install development dependencies
COPY requirements-dev.txt /tmp/requirements-dev.txt
RUN pip install --no-cache-dir -r /tmp/requirements-dev.txt

# Install additional development tools
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Switch back to app user
USER $APP_USER

# Override command for development (with auto-reload)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# =============================================================================
# TESTING STAGE (Optional)
# =============================================================================
FROM development as testing

# Switch to root for test setup
USER root

# Install test-specific dependencies
RUN apt-get update && apt-get install -y \
    firefox-esr \
    && rm -rf /var/lib/apt/lists/*

# Switch back to app user
USER $APP_USER

# Set test environment
ENV ENVIRONMENT=test

# Override command for testing
CMD ["pytest", "--cov=app", "--cov-report=html", "--cov-report=xml", "-v"]

# =============================================================================
# BUILD INSTRUCTIONS
# =============================================================================

# Build production image:
# docker build -t football-predictor:latest .

# Build development image:
# docker build --target development -t football-predictor:dev .

# Build testing image:
# docker build --target testing -t football-predictor:test .

# Build with specific Python version:
# docker build --build-arg PYTHON_VERSION=3.11 -t football-predictor:py311 .

# Build with custom user:
# docker build --build-arg APP_USER=myuser --build-arg APP_UID=1001 -t football-predictor:custom .

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

# Run production container:
# docker run -d --name football-predictor -p 8000:8000 --env-file .env football-predictor:latest

# Run development container with volume mount:
# docker run -d --name football-predictor-dev -p 8000:8000 -v $(pwd):/app --env-file .env football-predictor:dev

# Run tests:
# docker run --rm -v $(pwd):/app --env-file .env.test football-predictor:test

# Interactive shell:
# docker run -it --rm -v $(pwd):/app football-predictor:dev bash

# =============================================================================
# OPTIMIZATION NOTES
# =============================================================================

# 1. Multi-stage build reduces final image size
# 2. Virtual environment isolation
# 3. Non-root user for security
# 4. Proper layer caching with requirements.txt first
# 5. Health check for container orchestration
# 6. Minimal runtime dependencies
# 7. Clean apt cache to reduce size

# =============================================================================
# SECURITY CONSIDERATIONS
# =============================================================================

# 1. Non-root user execution
# 2. Minimal base image (slim)
# 3. No unnecessary packages
# 4. Regular security updates
# 5. Proper file permissions
# 6. Environment variable handling

# =============================================================================
# TROUBLESHOOTING
# =============================================================================

# Common issues and solutions:

# 1. Permission denied errors:
#    - Ensure proper ownership with --chown flag
#    - Check user ID matches host system if needed

# 2. Module not found errors:
#    - Verify PYTHONPATH is set correctly
#    - Check virtual environment activation

# 3. Health check failures:
#    - Ensure curl is installed
#    - Verify application starts correctly
#    - Check port binding

# 4. Build failures:
#    - Clear Docker cache: docker builder prune
#    - Check requirements.txt syntax
#    - Verify base image availability

# 5. Runtime errors:
#    - Check environment variables
#    - Verify external service connectivity
#    - Review application logs