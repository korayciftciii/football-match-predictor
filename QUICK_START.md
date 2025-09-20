# 🚀 Hızlı Başlangıç Rehberi

Bu rehber, Football Match Predictor projesini hızlıca çalıştırmanız için gerekli adımları içerir.

## ✅ Ön Gereksinimler

- ✅ Python 3.11+ yüklü
- ✅ Redis server çalışıyor
- ⏳ API anahtarları (aşağıda açıklanmıştır)

## 🔑 API Anahtarları

### 1. API-Football (Zorunlu)
1. [API-Football](https://www.api-football.com/) sitesine gidin
2. Ücretsiz hesap oluşturun (günde 100 istek)
3. API anahtarınızı alın

### 2. OpenRouter (Zorunlu)
1. [OpenRouter](https://openrouter.ai/) sitesine gidin
2. Hesap oluşturun
3. API anahtarınızı alın

### 3. Telegram Bot (Zorunlu)
1. Telegram'da [@BotFather](https://t.me/botfather) ile konuşun
2. `/newbot` komutu ile yeni bot oluşturun
3. Bot token'ınızı alın
4. Bot ile konuşup chat ID'nizi öğrenin

### 4. Hugging Face (İsteğe Bağlı)
1. [Hugging Face](https://huggingface.co/settings/tokens) sitesine gidin
2. Hesap oluşturun
3. API token'ınızı alın (gelişmiş ML modelleri için)

## 🛠️ Kurulum Adımları

### 1. Sanal Ortam Oluşturun
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 2. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 3. Ortam Değişkenlerini Ayarlayın
```bash
# .env dosyasını oluşturun
copy .env.example .env

# .env dosyasını düzenleyin ve API anahtarlarınızı ekleyin
```

### 4. .env Dosyası Örneği
```env
# API Anahtarları
API_FOOTBALL_KEY=your_api_football_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# Redis (varsayılan ayarlar)
REDIS_URL=redis://localhost:6379/0

# Uygulama Ayarları
LOG_LEVEL=INFO
CACHE_TTL=3600
PREDICTION_CACHE_HOURS=3

# İsteğe Bağlı
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
```

## 🚀 Uygulamayı Çalıştırın

### Yöntem 1: Doğrudan Python
```bash
python -m app.main
```

### Yöntem 2: Uvicorn ile (Geliştirme)
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Yöntem 3: Docker ile
```bash
# Docker Compose ile
docker-compose up -d

# Sadece uygulama
docker build -t football-predictor .
docker run -p 8000:8000 --env-file .env football-predictor
```

## 🧪 Test Etme

### 1. API Endpoint'lerini Test Edin
```bash
# Sağlık kontrolü
curl http://localhost:8000/health

# Bugünün maçları
curl http://localhost:8000/matches/today

# Maç tahmini (örnek match ID ile)
curl http://localhost:8000/matches/12345/predictions
```

### 2. Telegram Bot'u Test Edin
1. Telegram'da botunuza `/start` gönderin
2. Bugünün maçlarını görmelisiniz
3. Bir maça tıklayarak tahmin alın

### 3. Web Arayüzü
- Tarayıcınızda `http://localhost:8000/docs` adresine gidin
- Swagger UI ile API'yi test edin

## 🔍 Sorun Giderme

### Redis Bağlantı Hatası
```bash
# Redis'in çalıştığını kontrol edin
redis-cli ping

# Windows'ta Redis başlatma
redis-server

# Docker ile Redis
docker run -d -p 6379:6379 redis:alpine
```

### API Anahtarı Hataları
```bash
# Konfigürasyonu test edin
python -c "from app.config import settings; print('Config loaded successfully')"

# API bağlantısını test edin
python -c "
import asyncio
from app.services.fetch_data import FootballDataFetcher
async def test():
    fetcher = FootballDataFetcher()
    matches = await fetcher.get_todays_matches()
    print(f'Found {len(matches)} matches')
asyncio.run(test())
"
```

### Telegram Bot Hataları
```bash
# Bot token'ını test edin
python -c "
import asyncio
from app.bots.telegram_bot import FootballPredictionBot
async def test():
    bot = FootballPredictionBot()
    me = await bot.application.bot.get_me()
    print(f'Bot: {me.first_name} (@{me.username})')
asyncio.run(test())
"
```

### Bağımlılık Hataları
```bash
# Bağımlılıkları yeniden yükleyin
pip install --upgrade -r requirements.txt

# Sanal ortamı yeniden oluşturun
deactivate
rmdir /s venv  # Windows
rm -rf venv    # macOS/Linux
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

## 📊 Beklenen Çıktı

### Başarılı Başlatma
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Telegram Bot Çıktısı
```
🏈 Antalyaspor vs Kayserispor
🏆 Süper Lig
🕐 20.09.2025 17:00

📊 TAHMİNLER:
⚽️ Skor: 2-1
📈 Güven: %72
🎯 Gol Atacak Oyuncu: M. Thiam
📈 Olasılık: %45
🟨 Sarı Kart: 4 adet
⛳️ Korner: 8 adet
🕐 İlk Yarı: 1-0

🤖 AI ANALİZİ:
Galatasaray'ın ev sahibi avantajı ve son dönemdeki iyi performansı...
```

## 🎯 Sonraki Adımlar

1. **Model İyileştirme**: `docs/model-improvement-guide.md` dosyasını inceleyin
2. **Özellik Ekleme**: Yeni tahmin türleri ekleyin
3. **Veri Genişletme**: Daha fazla veri kaynağı entegre edin
4. **Performans Optimizasyonu**: Caching ve hız iyileştirmeleri yapın

## 📞 Destek

Sorun yaşıyorsanız:
1. `logs/app.log` dosyasını kontrol edin
2. GitHub Issues'da sorun bildirin
3. Dokümantasyonu inceleyin: `docs/` klasörü

---

**Başarılar! ⚽️🎯**