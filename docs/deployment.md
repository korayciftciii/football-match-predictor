# 🚀 Deployment Guide

This guide covers various deployment options for the Football Match Predictor application, from local development to production environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Configuration](#environment-configuration)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Production Considerations](#production-considerations)
- [Monitoring and Maintenance](#monitoring-and-maintenance)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **CPU**: 2+ cores recommended
- **RAM**: 2GB minimum, 4GB recommended
- **Storage**: 5GB free space
- **Network**: Stable internet connection for API access

### Software Dependencies

- **Python**: 3.11 or higher
- **Redis**: 7.0 or higher
- **Docker**: 20.10+ (for containerized deployment)
- **Git**: For source code management

### API Keys Required

1. **API-Football**: [Get free API key](https://www.api-football.com/)
2. **OpenRouter**: [Get API key](https://openrouter.ai/)
3. **Telegram Bot**: Create bot via [@BotFather](https://t.me/botfather)

## Environment Configuration

### Environment Variables

Create a `.env` file with the following configuration:

```env
# API Keys (Required)
API_FOOTBALL_KEY=your_api_football_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# Application Settings
LOG_LEVEL=INFO
ENVIRONMENT=production
HOST=0.0.0.0
PORT=8000

# Cache Settings
CACHE_TTL=3600
PREDICTION_CACHE_HOURS=3

# ML Model Settings
MODEL_CONFIDENCE_THRESHOLD=0.6
PREDICTION_TIMEOUT=30

# API Settings
API_RATE_LIMIT=100
API_TIMEOUT=30

# Security (Production)
SECRET_KEY=your_secret_key_here
ALLOWED_HOSTS=your-domain.com,localhost
CORS_ORIGINS=https://your-frontend.com
```

### Configuration Validation

Validate your configuration:

```bash
# Check environment variables
python -c "from app.config import settings; print('Config loaded successfully')"

# Test API connections
python -c "
from app.services.fetch_data import FootballDataFetcher
import asyncio
async def test(): 
    fetcher = FootballDataFetcher()
    health = await fetcher.health_check()
    print(f'API Health: {health}')
asyncio.run(test())
"
```

## Local Development

### Quick Setup

```bash
# Clone repository
git clone https://github.com/your-username/football-match-predictor.git
cd football-match-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your API keys

# Start Redis
redis-server

# Run application
python -m app.main
```

### Development Server

```bash
# Run with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run with debug logging
LOG_LEVEL=DEBUG uvicorn app.main:app --reload

# Run specific components
python -c "from app.bots.telegram_bot import FootballPredictionBot; import asyncio; bot = FootballPredictionBot(); asyncio.run(bot.start_polling())"
```

### Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Test specific components
pytest tests/test_workflow.py -v
```

## Docker Deployment

### Single Container

#### Dockerfile

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["python", "-m", "app.main"]
```

#### Build and Run

```bash
# Build image
docker build -t football-predictor .

# Run container
docker run -d \
  --name football-predictor \
  -p 8000:8000 \
  --env-file .env \
  football-predictor

# View logs
docker logs -f football-predictor

# Stop container
docker stop football-predictor
```

### Docker Compose (Recommended)

#### docker-compose.yml

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379/0
    env_file:
      - .env
    depends_on:
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    volumes:
      - ./logs:/app/logs
    networks:
      - football-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
      - ./redis.conf:/usr/local/etc/redis/redis.conf
    command: redis-server /usr/local/etc/redis/redis.conf
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - football-network

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: unless-stopped
    networks:
      - football-network

volumes:
  redis_data:

networks:
  football-network:
    driver: bridge
```

#### Redis Configuration (redis.conf)

```conf
# Redis configuration for production
maxmemory 256mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
appendonly yes
appendfsync everysec
```####
 Nginx Configuration (nginx.conf)

```nginx
events {
    worker_connections 1024;
}

http {
    upstream app {
        server app:8000;
    }

    server {
        listen 80;
        server_name your-domain.com;

        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        # SSL Configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";

        # Rate limiting
        limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

        location / {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }

        # Health check endpoint (no rate limiting)
        location /health {
            proxy_pass http://app;
            proxy_set_header Host $host;
        }
    }
}
```

#### Deploy with Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Scale application
docker-compose up -d --scale app=3

# Update application
docker-compose pull
docker-compose up -d

# Stop services
docker-compose down
```

## Cloud Deployment

### Heroku

#### Preparation

```bash
# Install Heroku CLI
# Create Heroku app
heroku create your-app-name

# Add Redis addon
heroku addons:create heroku-redis:mini

# Set environment variables
heroku config:set API_FOOTBALL_KEY=your_key
heroku config:set OPENROUTER_API_KEY=your_key
heroku config:set TELEGRAM_BOT_TOKEN=your_token
heroku config:set ENVIRONMENT=production
```

#### Procfile

```procfile
web: python -m app.main
worker: python -c "from app.bots.telegram_bot import FootballPredictionBot; import asyncio; bot = FootballPredictionBot(); asyncio.run(bot.start_polling())"
```

#### Deploy

```bash
# Deploy to Heroku
git push heroku main

# Scale dynos
heroku ps:scale web=1 worker=1

# View logs
heroku logs --tail

# Run commands
heroku run python -c "from app.config import settings; print(settings.dict())"
```

### AWS ECS (Fargate)

#### Task Definition

```json
{
  "family": "football-predictor",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "app",
      "image": "your-registry/football-predictor:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "ENVIRONMENT", "value": "production"},
        {"name": "REDIS_URL", "value": "redis://your-elasticache-cluster:6379/0"}
      ],
      "secrets": [
        {
          "name": "API_FOOTBALL_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:football-predictor/api-keys:API_FOOTBALL_KEY::"
        },
        {
          "name": "OPENROUTER_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:football-predictor/api-keys:OPENROUTER_API_KEY::"
        },
        {
          "name": "TELEGRAM_BOT_TOKEN",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:football-predictor/api-keys:TELEGRAM_BOT_TOKEN::"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/football-predictor",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

### Google Cloud Run

#### Deploy Script

```bash
#!/bin/bash
# deploy-gcp.sh

# Set project
gcloud config set project your-project-id

# Build and push image
gcloud builds submit --tag gcr.io/your-project-id/football-predictor

# Deploy to Cloud Run
gcloud run deploy football-predictor \
  --image gcr.io/your-project-id/football-predictor \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --concurrency 80 \
  --max-instances 10 \
  --set-env-vars ENVIRONMENT=production \
  --set-env-vars REDIS_URL=redis://your-memorystore-instance:6379/0

# Set up custom domain (optional)
gcloud run domain-mappings create --service football-predictor --domain your-domain.com
```

## Production Considerations

### Security

#### Environment Variables Security

```bash
# Use secrets management
# AWS Secrets Manager
aws secretsmanager create-secret --name football-predictor/api-keys --secret-string '{
  "API_FOOTBALL_KEY": "your_key",
  "OPENROUTER_API_KEY": "your_key",
  "TELEGRAM_BOT_TOKEN": "your_token"
}'

# Google Secret Manager
gcloud secrets create api-football-key --data-file=-
gcloud secrets create openrouter-api-key --data-file=-
gcloud secrets create telegram-bot-token --data-file=-
```

#### Network Security

```bash
# Firewall rules (example for GCP)
gcloud compute firewall-rules create allow-football-predictor \
  --allow tcp:8000 \
  --source-ranges 10.0.0.0/8 \
  --target-tags football-predictor

# Security groups (example for AWS)
aws ec2 create-security-group \
  --group-name football-predictor-sg \
  --description "Security group for Football Predictor"

aws ec2 authorize-security-group-ingress \
  --group-name football-predictor-sg \
  --protocol tcp \
  --port 8000 \
  --cidr 10.0.0.0/8
```

### Performance Optimization

#### Application Tuning

```python
# app/config.py - Production settings
class ProductionSettings(Settings):
    # Worker processes
    WORKERS: int = 4
    
    # Connection pools
    MAX_CONNECTIONS: int = 100
    POOL_SIZE: int = 20
    
    # Cache optimization
    CACHE_TTL: int = 3600
    PREDICTION_CACHE_HOURS: int = 6
    
    # Request timeouts
    API_TIMEOUT: int = 30
    PREDICTION_TIMEOUT: int = 60
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    BURST_LIMIT: int = 200
```

#### Redis Optimization

```conf
# redis-production.conf
maxmemory 1gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
appendonly yes
appendfsync everysec
tcp-keepalive 300
timeout 0
```

## Monitoring and Maintenance

### Health Monitoring

#### Uptime Monitoring

```bash
#!/bin/bash
# health-monitor.sh

ENDPOINT="https://your-domain.com/health"
WEBHOOK_URL="https://hooks.slack.com/your-webhook"

while true; do
    if ! curl -f $ENDPOINT > /dev/null 2>&1; then
        curl -X POST -H 'Content-type: application/json' \
            --data '{"text":"🚨 Football Predictor is DOWN!"}' \
            $WEBHOOK_URL
        echo "$(date): Service is DOWN" >> /var/log/health-monitor.log
    else
        echo "$(date): Service is UP" >> /var/log/health-monitor.log
    fi
    sleep 60
done
```

### Maintenance Tasks

#### Daily Maintenance

```bash
#!/bin/bash
# daily-maintenance.sh

# Clear old logs
find /app/logs -name "*.log" -mtime +30 -delete

# Clear old cache entries
redis-cli FLUSHDB

# Update application metrics
python -c "
from app.utils.monitoring import MetricsCollector
collector = MetricsCollector()
collector.generate_daily_report()
"

# Check disk space
df -h | awk '$5 > 80 {print "Disk usage warning: " $0}' | \
    xargs -I {} curl -X POST -H 'Content-type: application/json' \
    --data '{"text":"💾 {}"}' \
    https://hooks.slack.com/your-webhook

echo "Daily maintenance completed: $(date)"
```

## Troubleshooting

### Common Issues

#### Redis Connection Issues

```bash
# Check Redis connectivity
redis-cli ping

# Check Redis logs
docker logs redis-container

# Test from application
python -c "
import redis
r = redis.from_url('redis://localhost:6379/0')
print(r.ping())
"
```

#### API Rate Limiting

```bash
# Check API usage
curl -H "X-RapidAPI-Key: your_key" \
     "https://api-football-v1.p.rapidapi.com/v3/status"

# Monitor rate limits
grep "rate limit" /app/logs/app.log | tail -10
```

#### Memory Issues

```bash
# Check memory usage
free -h
docker stats

# Check application memory
python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with debug
python -m app.main --debug

# Check specific component
python -c "
import asyncio
from app.services.workflow import WorkflowOrchestrator
async def debug():
    orchestrator = WorkflowOrchestrator()
    result = await orchestrator.health_check()
    print(result)
asyncio.run(debug())
"
```

### Recovery Procedures

#### Service Recovery

```bash
#!/bin/bash
# recover-service.sh

echo "Starting service recovery..."

# Stop services
docker-compose down

# Clear problematic cache
redis-cli FLUSHALL

# Restore from backup if needed
if [ "$1" = "restore" ]; then
    echo "Restoring from backup..."
    # Restore logic here
fi

# Start services
docker-compose up -d

# Wait for health check
sleep 30
curl -f http://localhost:8000/health

echo "Service recovery completed"
```

---

*This deployment guide covers the most common deployment scenarios. For specific cloud provider configurations or custom setups, refer to the respective platform documentation.*