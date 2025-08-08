# Complete FastAPI Client Interface

This document describes the complete client interface for the Interview Agent FastAPI, including the comprehensive client library and interactive CLI application.

## Overview

The complete client interface provides:
- **FastAPI Client Library** (`fastapi_client.py`) - Comprehensive Python client
- **Interactive CLI Client** (`interactive_client.py`) - User-friendly command-line interface
- **Session Management** - Complete interview session lifecycle
- **Audio Processing** - Real-time audio recording and processing
- **Model Selection** - Flexible model configuration
- **Error Handling** - Robust error handling and recovery

## Features

### 🚀 Core Features
- **Session Management**: Create, manage, and track interview sessions
- **Audio Processing**: Record, simulate, and process audio with ASR
- **Real-time Communication**: WebSocket support for live audio processing
- **Model Selection**: Choose from available ASR and LLM models
- **Conversation History**: Track and retrieve conversation history
- **Audio Playback**: Play AI responses with TTS audio

### 🛠️ Client Features
- **Connection Testing**: Verify server connectivity
- **Error Recovery**: Graceful handling of network and API errors
- **Configuration Management**: Flexible session configuration
- **Audio Utilities**: Audio recording, simulation, and playback
- **Session Persistence**: Maintain session state across operations

## Installation

1. **Install Dependencies**:
```bash
pip install -r requirements_fastapi.txt
```

2. **Start the FastAPI Server**:
```bash
python start_fastapi_server.py
```

3. **Use the Client**:
```bash
# Interactive CLI
python interactive_client.py

# Or use the client library directly
python fastapi_client.py
```

## Client Library Usage

### Basic Usage

```python
from fastapi_client import InterviewAgentClient, SessionConfig, InterviewSession

# Create client
client = InterviewAgentClient("http://localhost:8000")

# Create session configuration
config = SessionConfig(
    interview_topic="Machine Learning",
    resume_file="path/to/resume.pdf",
    asr_model="openai/whisper-small",
    llm_provider="ollama",
    llm_model="llama3.2:1b"  # Default model
)

# Create and start interview session
session = InterviewSession(client, config)
response = session.start()

print(f"AI: {response.response}")
```

### Advanced Usage

```python
# Process audio
audio_data = record_audio(5.0)  # Record 5 seconds
response = session.respond_to_audio(audio_data, play_audio=True)

# Get conversation history
history = session.get_conversation_history()
for message in history:
    print(f"{message['role']}: {message['content']}")

# End session
session.end()
```

### WebSocket Usage

```python
import asyncio

async def handle_message(message):
    print(f"Received: {message}")

async def handle_error(error):
    print(f"Error: {error}")

# Connect to WebSocket
await client.websocket_connect(
    on_message=handle_message,
    on_error=handle_error
)
```

## Interactive CLI Client

The interactive CLI provides a user-friendly interface for all client operations.

### Starting the Interactive Client

```bash
python interactive_client.py
```

### Menu Options

1. **📊 View Available Models** - List all available ASR and LLM models
2. **🆕 Create New Interview Session** - Create a new interview session with custom configuration
3. **🎤 Start Interview** - Start the interview with an opening question
4. **🎵 Process Audio (Simulated)** - Process simulated audio data
5. **🎙️ Record and Process Audio** - Record real audio and process it
6. **📝 View Session Information** - Display current session details
7. **💬 View Conversation History** - Show conversation history
8. **📋 List All Sessions** - List all available sessions
9. **🗑️ Delete Session** - Delete a specific session
10. **⏹️ End Current Session** - End the active session
11. **🔄 Test Server Connection** - Test connectivity to the server
0. **🚪 Exit** - Exit the application

### Example Interactive Session

```
============================================================
           INTERVIEW AGENT FASTAPI CLIENT
============================================================

📋 MAIN MENU:
1.  📊 View Available Models
2.  🆕 Create New Interview Session
3.  🎤 Start Interview
4.  🎵 Process Audio (Simulated)
5.  🎙️  Record and Process Audio
6.  📝 View Session Information
7.  💬 View Conversation History
8.  📋 List All Sessions
9.  🗑️  Delete Session
10. ⏹️  End Current Session
11. 🔄 Test Server Connection
0.  🚪 Exit
------------------------------------------------------------
Enter your choice (0-11): 2

🆕 CREATING NEW INTERVIEW SESSION
----------------------------------------
Enter interview topic (e.g., Machine Learning): Python Development
Enter resume file path (or press Enter for default): 
🔧 MODEL CONFIGURATION:
ASR model (press Enter for default 'openai/whisper-small'): 
LLM provider (press Enter for default 'ollama'): 
LLM model (press Enter for default 'llama3.2:1b'): 

🔄 Creating session...
✅ Session created successfully!
```

## Configuration

### Session Configuration

```python
@dataclass
class SessionConfig:
    interview_topic: str                    # Interview topic
    resume_file: Optional[str] = None       # Resume file path
    asr_model: str = "openai/whisper-small" # ASR model
    llm_provider: str = "ollama"           # LLM provider
    llm_model: str = "llama3.2:1b"         # LLM model (default)
```

### Client Configuration

```python
client = InterviewAgentClient(
    base_url="http://localhost:8000",  # Server URL
    timeout=30                         # Request timeout
)
```

## Audio Processing

### Recording Audio

```python
from fastapi_client import record_audio

# Record 5 seconds of audio
audio_data = record_audio(duration=5.0, sample_rate=16000)
```

### Simulating Audio

```python
from fastapi_client import simulate_audio_data

# Generate 2 seconds of 440Hz sine wave
audio_data = simulate_audio_data(duration=2.0, frequency=440.0)
```

### Processing Audio

```python
# Process audio and get response
response = session.respond_to_audio(audio_data, play_audio=True)

if response.transcription:
    print(f"Transcription: {response.transcription}")

if response.response:
    print(f"AI Response: {response.response}")
```

## Error Handling

The client provides comprehensive error handling:

```python
from fastapi_client import FastAPIClientError

try:
    response = session.start()
except FastAPIClientError as e:
    print(f"Client error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Common Error Types

- **Connection Errors**: Server unreachable
- **Session Errors**: Invalid session state
- **Audio Processing Errors**: Audio format or processing issues
- **Model Errors**: Model loading or inference failures

## API Reference

### InterviewAgentClient

#### Methods

- `get_available_models()` - Get available models
- `create_session(config)` - Create new session
- `start_interview()` - Start interview
- `process_audio(audio_data, sample_rate)` - Process audio
- `get_session_info()` - Get session information
- `end_session()` - End session
- `list_sessions()` - List all sessions
- `delete_session(session_id)` - Delete session
- `play_audio_response(audio_response)` - Play audio
- `save_audio_response(audio_response, filename)` - Save audio

### InterviewSession

#### Methods

- `start()` - Start interview session
- `respond_to_audio(audio_data, sample_rate, play_audio)` - Process audio
- `end()` - End session
- `get_conversation_history()` - Get conversation history

## Examples

### Complete Interview Flow

```python
from fastapi_client import InterviewAgentClient, SessionConfig, InterviewSession, record_audio

# Setup
client = InterviewAgentClient("http://localhost:8000")
config = SessionConfig(
    interview_topic="Software Engineering",
    resume_file="resume.pdf"
)

# Create session
session = InterviewSession(client, config)

# Start interview
response = session.start()
print(f"AI: {response.response}")

# Record and process audio
print("Recording your response...")
audio_data = record_audio(5.0)
response = session.respond_to_audio(audio_data, play_audio=True)

# Continue conversation
print("Recording another response...")
audio_data = record_audio(3.0)
response = session.respond_to_audio(audio_data, play_audio=True)

# End session
session.end()
```

### Batch Processing

```python
# Process multiple audio files
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]

for audio_file in audio_files:
    audio_data = load_audio_file(audio_file)
    response = session.respond_to_audio(audio_data, play_audio=False)
    print(f"File: {audio_file}")
    print(f"Transcription: {response.transcription}")
    print(f"Response: {response.response}")
    print("-" * 50)
```

## Troubleshooting

### Common Issues

1. **Connection Failed**
   - Ensure the FastAPI server is running
   - Check the server URL
   - Verify network connectivity

2. **Audio Recording Issues**
   - Check microphone permissions
   - Ensure sounddevice is installed
   - Verify audio device configuration

3. **Model Loading Errors**
   - Ensure all required models are downloaded
   - Check model file paths
   - Verify model compatibility

4. **Session Errors**
   - Create a new session if current session is invalid
   - Check session status before operations
   - Ensure proper session cleanup

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Testing

Test the complete setup:

```bash
# 1. Start server
python start_fastapi_server.py

# 2. Test client
python fastapi_client.py

# 3. Use interactive client
python interactive_client.py
```

## Performance Considerations

- **Audio Processing**: Large audio files may take longer to process
- **Model Loading**: Models are loaded per session for isolation
- **Memory Usage**: Monitor memory usage with large models
- **Network Latency**: Consider server location for real-time applications

## Security

- **Input Validation**: All inputs are validated before processing
- **Session Isolation**: Sessions are isolated from each other
- **Error Messages**: Sensitive information is not exposed in error messages
- **Connection Security**: Use HTTPS in production environments

## Contributing

To extend the client interface:

1. Add new methods to `InterviewAgentClient`
2. Update `InterviewSession` for high-level operations
3. Add corresponding menu options to `InteractiveClient`
4. Update documentation and examples
5. Add tests for new functionality 