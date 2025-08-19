# Docker Deployment Guide

## Problem Solved

The `/sessions/create` endpoint was timing out (504 Gateway Timeout) when running in Docker due to:

1. **Heavy model loading on every session creation** - ASR, TTS, and VAD models were loaded synchronously for each session
2. **Large model downloads** - Whisper model (~1.5GB) downloaded on first use
3. **No caching** - Models re-downloaded if not cached
4. **Docker environment constraints** - Limited resources and slower I/O

## Solution Implemented

### 1. Model Preloading and Caching
- Models are now preloaded at startup in background threads
- Global model cache prevents reloading models for each session
- Health check endpoint shows model loading status

### 2. Optimized Session Creation
- Increased timeout from 30s to 60s for Docker environments
- Better error messages indicating model loading status
- Graceful fallback if models aren't ready

### 3. Docker Optimizations
- Persistent volume for model cache (`/root/.cache/huggingface`)
- Memory limits and reservations
- Health checks with proper startup period
- Curl included for health check monitoring

## Deployment Instructions

### 1. Build and Run with Docker Compose

```bash
# Build and start the service
docker-compose up --build

# Run in background
docker-compose up -d --build
```

### 2. Check Service Status

```bash
# Check if service is running
docker-compose ps

# Check logs
docker-compose logs -f interview-agent

# Check health status
curl http://localhost:8000/health
```

### 3. Wait for Models to Load

The service will show "initializing" status until all models are loaded:

```json
{
  "status": "initializing",
  "models_ready": {
    "asr_models": true,
    "tts_processor": false,
    "vad_processor": true
  },
  "all_models_ready": false
}
```

Once all models are ready:
```json
{
  "status": "healthy",
  "models_ready": {
    "asr_models": true,
    "tts_processor": true,
    "vad_processor": true
  },
  "all_models_ready": true
}
```

### 4. Monitor Model Loading

```bash
# Watch logs for model loading progress
docker-compose logs -f interview-agent | grep -E "(Preloading|loaded|cached)"

# Use the wait script
python wait_for_models.py http://localhost:8000 300
```

## Environment Variables

Create a `.env` file in the `poc` directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here
```

## Performance Improvements

### Before (Causing Timeouts)
- Session creation: 30-60 seconds (model loading)
- Each session loads models independently
- No caching, repeated downloads

### After (Optimized)
- Session creation: 2-5 seconds (cached models)
- Models loaded once at startup
- Persistent cache across container restarts

## Troubleshooting

### 1. Models Still Loading
If you get 504 timeouts, check:
```bash
curl http://localhost:8000/health
```

Wait for `"all_models_ready": true` before creating sessions.

### 2. Memory Issues
If the container runs out of memory:
```bash
# Check memory usage
docker stats

# Increase memory limit in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 6G  # Increase from 4G
```

### 3. Model Cache Issues
If models aren't being cached:
```bash
# Check if cache volume is mounted
docker-compose exec interview-agent ls -la /root/.cache/huggingface

# Clear cache if needed
docker-compose down
docker volume rm poc_model_cache
docker-compose up --build
```

### 4. Health Check Failures
If health checks are failing:
```bash
# Check if curl is available
docker-compose exec interview-agent which curl

# Check health endpoint manually
docker-compose exec interview-agent curl -f http://localhost:8000/health
```

## Production Considerations

1. **Resource Allocation**: Ensure adequate CPU and memory
2. **Model Cache**: Use persistent volumes for model caching
3. **Monitoring**: Monitor health endpoint for model status
4. **Graceful Degradation**: Service works even if some models fail to load
5. **Restart Policy**: Use `restart: unless-stopped` for reliability

## Expected Timeline

- **First startup**: 2-5 minutes (model downloads)
- **Subsequent startups**: 30-60 seconds (cached models)
- **Session creation**: 2-5 seconds (cached models)
- **Health check ready**: 2-5 minutes after startup
