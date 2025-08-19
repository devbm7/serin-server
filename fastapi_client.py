"""
Complete FastAPI Client for Interview Agent
Provides a comprehensive interface to the FastAPI endpoints
"""

import requests
import json
import base64
import numpy as np
import time
import asyncio
import websockets
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SessionConfig:
    """Configuration for interview sessions"""
    interview_topic: str
    resume_file: str  # Can be user ID, resume path, or local file path
    asr_model: str = "openai/whisper-medium"
    llm_provider: str = "ollama"
    llm_model: str = "llama3.2:1b"

@dataclass
class InterviewResponse:
    """Response from interview processing"""
    transcription: str
    response: str
    audio_response: Optional[str] = None
    session_id: Optional[str] = None

class FastAPIClientError(Exception):
    """Custom exception for FastAPI client errors"""
    pass

class InterviewAgentClient:
    """
    Complete client for the Interview Agent FastAPI
    
    Provides methods for:
    - Session management
    - Audio processing
    - Real-time communication via WebSocket
    - Model selection
    - Interview flow control
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 3000):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session_id = None
        self.session_info = None
        self.is_connected = False
        
        # Test connection
        try:
            self._test_connection()
        except Exception as e:
            logger.warning(f"Could not connect to server at {base_url}: {e}")
    
    def _test_connection(self) -> bool:
        """Test connection to the FastAPI server"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Connected to Interview Agent API: {data.get('version', 'unknown')}")
            return True
        except requests.exceptions.RequestException as e:
            raise FastAPIClientError(f"Failed to connect to server: {e}")
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get available ASR and LLM models"""
        try:
            response = requests.get(f"{self.base_url}/models/available", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise FastAPIClientError(f"Failed to get available models: {e}")
    
    def create_session(self, config: SessionConfig) -> str:
        """
        Create a new interview session
        
        Args:
            config: Session configuration
            
        Returns:
            Session ID
        """
        try:
            # Create the request payload matching ModelSelectionRequest
            model_request = {
                "interview_topic": config.interview_topic,
                "resume_file": config.resume_file,
                "asr_model": config.asr_model,
                "llm_provider": config.llm_provider,
                "llm_model": config.llm_model
            }
            
            response = requests.post(
                f"{self.base_url}/sessions/create",
                json=model_request,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            self.session_id = result["session_id"]
            logger.info(f"Created session: {self.session_id}")
            return self.session_id
            
        except requests.exceptions.RequestException as e:
            raise FastAPIClientError(f"Failed to create session: {e}")
    
    def start_interview(self) -> InterviewResponse:
        """
        Start the interview with an opening question
        
        Returns:
            InterviewResponse with opening question
        """
        if not self.session_id:
            raise FastAPIClientError("No active session. Create a session first.")
        
        try:
            response = requests.post(
                f"{self.base_url}/sessions/{self.session_id}/start",
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info("Interview started successfully")
            
            return InterviewResponse(
                transcription=result.get('transcription', ''),
                response=result.get('response', ''),
                audio_response=result.get('audio_response'),
                session_id=self.session_id
            )
            
        except requests.exceptions.RequestException as e:
            raise FastAPIClientError(f"Failed to start interview: {e}")
    
    def process_audio(self, audio_data: Union[bytes, np.ndarray], sample_rate: int = 16000) -> InterviewResponse:
        """
        Process an audio chunk and get the response
        
        Args:
            audio_data: Audio data as bytes or numpy array
            sample_rate: Audio sample rate
            
        Returns:
            InterviewResponse with transcription and AI response
        """
        if not self.session_id:
            raise FastAPIClientError("No active session. Create a session first.")
        
        try:
            # Convert audio data to bytes if it's a numpy array
            if isinstance(audio_data, np.ndarray):
                audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
            else:
                audio_bytes = audio_data
            
            # Convert to base64
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            audio_chunk = {
                "audio_data": audio_base64,
                "sample_rate": sample_rate
            }
            
            response = requests.post(
                f"{self.base_url}/sessions/{self.session_id}/process-audio",
                json=audio_chunk,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            return InterviewResponse(
                transcription=result.get('transcription', ''),
                response=result.get('response', ''),
                audio_response=result.get('audio_response'),
                session_id=self.session_id
            )
            
        except requests.exceptions.RequestException as e:
            raise FastAPIClientError(f"Failed to process audio: {e}")
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information"""
        if not self.session_id:
            raise FastAPIClientError("No active session. Create a session first.")
        
        try:
            response = requests.get(f"{self.base_url}/sessions/{self.session_id}", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise FastAPIClientError(f"Failed to get session info: {e}")
    
    def end_session(self) -> Dict[str, Any]:
        """End the current session"""
        if not self.session_id:
            raise FastAPIClientError("No active session. Create a session first.")
        
        try:
            response = requests.post(f"{self.base_url}/sessions/{self.session_id}/end", timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Session ended: {result['message']}")
            self.session_id = None
            return result
            
        except requests.exceptions.RequestException as e:
            raise FastAPIClientError(f"Failed to end session: {e}")
    
    def list_sessions(self) -> Dict[str, Any]:
        """List all sessions"""
        try:
            response = requests.get(f"{self.base_url}/sessions", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise FastAPIClientError(f"Failed to list sessions: {e}")
    
    def delete_session(self, session_id: str) -> Dict[str, Any]:
        """Delete a specific session"""
        try:
            response = requests.delete(f"{self.base_url}/sessions/{session_id}", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise FastAPIClientError(f"Failed to delete session: {e}")
    
    async def websocket_connect(self, on_message=None, on_error=None):
        """
        Connect to WebSocket for real-time communication
        
        Args:
            on_message: Callback function for received messages
            on_error: Callback function for errors
        """
        if not self.session_id:
            raise FastAPIClientError("No active session. Create a session first.")
        
        uri = f"ws://{self.base_url.replace('http://', '').replace('https://', '')}/ws/{self.session_id}"
        
        try:
            async with websockets.connect(uri) as websocket:
                self.is_connected = True
                logger.info(f"Connected to WebSocket: {uri}")
                
                while self.is_connected:
                    try:
                        # Receive message
                        message = await websocket.recv()
                        data = json.loads(message)
                        
                        if on_message:
                            await on_message(data)
                        else:
                            logger.info(f"Received: {data}")
                            
                    except websockets.exceptions.ConnectionClosed:
                        logger.info("WebSocket connection closed")
                        break
                    except Exception as e:
                        if on_error:
                            await on_error(e)
                        else:
                            logger.error(f"WebSocket error: {e}")
                        break
                        
        except Exception as e:
            self.is_connected = False
            if on_error:
                await on_error(e)
            else:
                logger.error(f"Failed to connect to WebSocket: {e}")
    
    async def websocket_send_audio(self, audio_data: Union[bytes, np.ndarray], sample_rate: int = 16000):
        """Send audio data via WebSocket"""
        if not self.is_connected:
            raise FastAPIClientError("Not connected to WebSocket")
        
        # Convert audio data to bytes if it's a numpy array
        if isinstance(audio_data, np.ndarray):
            audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
        else:
            audio_bytes = audio_data
        
        # Convert to base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        message = {
            "audio_data": audio_base64,
            "sample_rate": sample_rate
        }
        
        # This would need to be implemented with the actual WebSocket connection
        # For now, we'll use the HTTP endpoint
        return self.process_audio(audio_bytes, sample_rate)
    
    def play_audio_response(self, audio_response: str):
        """Play audio response (base64 encoded)"""
        if not audio_response:
            return
        
        try:
            audio_bytes = base64.b64decode(audio_response)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Try to play audio using sounddevice
            try:
                import sounddevice as sd
                sd.play(audio_array, 24000)  # Assuming 24kHz sample rate
                sd.wait()
            except ImportError:
                logger.warning("sounddevice not available, cannot play audio")
            except Exception as e:
                logger.error(f"Failed to play audio: {e}")
                
        except Exception as e:
            logger.error(f"Failed to decode audio response: {e}")
    
    def save_audio_response(self, audio_response: str, filename: str):
        """Save audio response to file"""
        if not audio_response:
            return
        
        try:
            audio_bytes = base64.b64decode(audio_response)
            with open(filename, 'wb') as f:
                f.write(audio_bytes)
            logger.info(f"Audio saved to: {filename}")
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")

class InterviewSession:
    """
    High-level interface for managing complete interview sessions
    """
    
    def __init__(self, client: InterviewAgentClient, config: SessionConfig):
        self.client = client
        self.config = config
        self.conversation_history = []
    
    def start(self) -> InterviewResponse:
        """Start the interview session"""
        # Create session
        self.client.create_session(self.config)
        
        # Start interview
        response = self.client.start_interview()
        
        # Add to conversation history
        if response.response:
            self.conversation_history.append({
                "role": "assistant",
                "content": response.response
            })
        
        return response
    
    def respond_to_audio(self, audio_data: Union[bytes, np.ndarray], 
                        sample_rate: int = 16000,
                        play_audio: bool = True) -> InterviewResponse:
        """Process audio and get response"""
        response = self.client.process_audio(audio_data, sample_rate)
        
        # Add to conversation history
        if response.transcription:
            self.conversation_history.append({
                "role": "user",
                "content": response.transcription
            })
        
        if response.response:
            self.conversation_history.append({
                "role": "assistant",
                "content": response.response
            })
        
        # Play audio if requested
        if play_audio and response.audio_response:
            self.client.play_audio_response(response.audio_response)
        
        return response
    
    def end(self) -> Dict[str, Any]:
        """End the interview session"""
        return self.client.end_session()
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversation_history.copy()

def simulate_audio_data(duration: float = 1.0, frequency: float = 440.0, sample_rate: int = 16000) -> np.ndarray:
    """Simulate audio data for testing"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * frequency * t)
    return audio

def record_audio(duration: float = 5.0, sample_rate: int = 16000) -> np.ndarray:
    """Record audio from microphone"""
    try:
        import sounddevice as sd
        logger.info(f"Recording {duration} seconds of audio...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        return audio.flatten()
    except ImportError:
        logger.error("sounddevice not available for recording")
        return simulate_audio_data(duration, 440, sample_rate)
    except Exception as e:
        logger.error(f"Failed to record audio: {e}")
        return simulate_audio_data(duration, 440, sample_rate)

def main():
    """Example usage of the complete FastAPI client"""
    print("=== Interview Agent FastAPI Client Demo ===")
    
    # Create client
    client = InterviewAgentClient("http://localhost:8000")
    
    try:
        # Get available models
        print("\n1. Getting available models...")
        models = client.get_available_models()
        print("Available models:")
        print(json.dumps(models, indent=2))
        
        # Create session configuration
        config = SessionConfig(
            interview_topic="Machine Learning",
            resume_file="user123",  # Can be user ID, resume path, or local file path
            asr_model="openai/whisper-small",
            llm_provider="ollama",
            llm_model="llama3.2:1b"
        )
        
        # Create interview session
        print("\n2. Creating interview session...")
        session = InterviewSession(client, config)
        
        # Start interview
        print("\n3. Starting interview...")
        start_response = session.start()
        print(f"Opening question: {start_response.response}")
        
        # Simulate audio processing
        print("\n4. Simulating audio processing...")
        audio_data = simulate_audio_data(2.0, 440)
        response = session.respond_to_audio(audio_data, play_audio=False)
        
        if response.transcription:
            print(f"Transcription: {response.transcription}")
        
        if response.response:
            print(f"AI Response: {response.response}")
        
        # Get session info
        print("\n5. Getting session information...")
        session_info = client.get_session_info()
        print("Session info:")
        print(json.dumps(session_info, indent=2))
        
        # Get conversation history
        print("\n6. Conversation history:")
        history = session.get_conversation_history()
        for i, message in enumerate(history):
            print(f"{i+1}. {message['role']}: {message['content']}")
        
        # End session
        print("\n7. Ending session...")
        session.end()
        print("Session ended successfully")
        
    except FastAPIClientError as e:
        print(f"Client error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main() 