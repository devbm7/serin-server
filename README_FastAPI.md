# Interview Agent FastAPI Pipeline

This is a FastAPI-based version of the interview agent pipeline that provides RESTful API endpoints for creating and managing interview sessions with AI-powered conversation capabilities.

## Features

- **RESTful API**: Complete HTTP API for session management
- **WebSocket Support**: Real-time communication for audio processing
- **Session Management**: Create, manage, and track interview sessions
- **Model Selection**: Choose from available ASR and LLM models
- **Audio Processing**: Speech-to-text, text-to-speech, and voice activity detection
- **Conversation History**: Track and retrieve conversation history
- **Resume Integration**: Upload and process candidate resumes
- **Cloud Storage**: Upload recordings to Supabase storage with local fallback

## API Endpoints

### Core Endpoints

- `GET /` - Health check and API info
- `GET /models/available` - Get available ASR and LLM models
- `POST /sessions/create` - Create a new interview session
- `GET /sessions` - List all sessions
- `GET /sessions/{session_id}` - Get session information
- `POST /sessions/{session_id}/start` - Start interview with opening question
- `POST /sessions/{session_id}/process-audio` - Process audio chunk
- `POST /sessions/{session_id}/end` - End interview session
- `DELETE /sessions/{session_id}` - Delete session
- `WS /ws/{session_id}` - WebSocket for real-time communication

### Recording Management Endpoints

- `POST /recordings/upload/{session_id}` - Upload recording to Supabase storage
- `GET /recordings/{session_id}` - Get recording information
- `GET /recordings/{session_id}/download` - Download recording or get Supabase URL
- `GET /recordings/{session_id}/list` - List all recordings for a session
- `DELETE /recordings/{session_id}/{filename}` - Delete recording from storage

## Installation

1. Install dependencies:
```bash
pip install -r requirements_fastapi.txt
```

2. Configure Supabase (optional but recommended):
   - Copy `env.example` to `.env` and add your Supabase credentials
   - Create an `interview-recordings` bucket in your Supabase project
   - See `README_Supabase_Integration.md` for detailed setup instructions

3. Ensure you have the required models and services running:
   - Ollama server (for LLM models)
   - Required model files (ASR, TTS, VAD)

4. Update the configuration in `pipeline_config.yaml` if needed.

## Usage

### Starting the Server

```bash
python fastapi_pipeline.py
```

The server will start on `http://localhost:8000` by default.

### API Documentation

Once the server is running, you can access:
- Interactive API docs: `http://localhost:8000/docs`
- Alternative API docs: `http://localhost:8000/redoc`

### Example Usage with Python Client

```python
from fastapi_client_example import InterviewAgentClient

# Create client
client = InterviewAgentClient("http://localhost:8000")

# Get available models
models = client.get_available_models()

# Create a session
session_id = client.create_session(
    interview_topic="Machine Learning",
    resume_file="path/to/resume.pdf"
)

# Start the interview
start_result = client.start_interview()

# Process audio (example with simulated audio)
audio_data = simulate_audio_data()  # Your audio data
result = client.process_audio(audio_data)

# End the session
client.end_session()
```

### WebSocket Usage

For real-time audio processing:

```python
import websockets
import json
import asyncio

async def websocket_client():
    uri = "ws://localhost:8000/ws/session_123"
    async with websockets.connect(uri) as websocket:
        # Send audio data
        audio_data = {
            "audio_data": "base64_encoded_audio",
            "sample_rate": 16000
        }
        await websocket.send(json.dumps(audio_data))
        
        # Receive response
        response = await websocket.recv()
        print(json.loads(response))

asyncio.run(websocket_client())
```

## Request/Response Models

### Create Session Request

```json
{
  "request": {
    "interview_topic": "Machine Learning",
    "resume_file": "path/to/resume.pdf"
  },
  "model_request": {
    "asr_model": "openai/whisper-small",
    "llm_provider": "ollama",
    "llm_model": "llama3"
  }
}
```

### Audio Processing Request

```json
{
  "audio_data": "base64_encoded_audio_data",
  "sample_rate": 16000
}
```

### Interview Response

```json
{
  "transcription": "User's transcribed speech",
  "response": "AI's response text",
  "audio_response": "base64_encoded_tts_audio"
}
```

## Configuration

The API uses the same configuration file (`pipeline_config.yaml`) as the original pipeline:

```yaml
audio:
  sample_rate: 16000
  channels: 1
  chunk_size: 512
  silence_threshold_seconds: 2

models:
  default_asr: "openai/whisper-small"
  default_llm:
    provider: "ollama"
    model: "llama3"
  tts_voice: "af_heart"

paths:
  resume_file: "Resume v3.01.pdf"
  sessions_dir: "sessions"
  log_file: "pipeline_debug.log"
```

## Session Management

The API maintains session state including:
- Conversation history
- Model instances (ASR, LLM, TTS, VAD)
- Session metadata (topic, resume content, timestamps)
- Processing status

Sessions are automatically cleaned up when ended or deleted.

## Error Handling

The API provides comprehensive error handling:
- HTTP status codes for different error types
- Detailed error messages
- Validation of request data
- Graceful handling of model failures

## Security Considerations

- CORS is enabled for all origins (configure as needed for production)
- Input validation using Pydantic models
- Session isolation
- Error messages don't expose sensitive information

## Performance

- Async/await for non-blocking operations
- Model instances are reused within sessions
- Efficient audio processing with VAD
- Background processing for long-running operations

## Development

### Adding New Endpoints

1. Define Pydantic models for request/response
2. Add route handler with proper error handling
3. Update session manager if needed
4. Add tests and documentation

### Testing

```bash
# Test the API endpoints
python fastapi_client_example.py

# Or use curl
curl -X GET "http://localhost:8000/models/available"
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure all required models are downloaded
2. **Audio Processing**: Check audio format and sample rate
3. **LLM Connection**: Verify Ollama server is running
4. **Memory Issues**: Monitor model memory usage for large models

### Logs

Check the log file specified in `pipeline_config.yaml` for detailed error information.

## Migration from CLI Version

The FastAPI version maintains compatibility with the original pipeline:
- Same model configurations
- Same processing logic
- Same configuration file
- Same dependencies

The main differences are:
- RESTful API interface instead of CLI
- Session management
- WebSocket support for real-time communication
- Structured request/response formats 