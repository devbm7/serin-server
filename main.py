import asyncio
import base64
import json
import logging
import numpy as np
import os
import queue
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import PyPDF2

# Import Supabase configuration
try:
    from supabase_config import supabase_config
    SUPABASE_AVAILABLE = True
    logging.info("Supabase configuration loaded successfully")
except ImportError as e:
    SUPABASE_AVAILABLE = False
    logging.warning(f"Supabase configuration not available: {e}")
# LiteLLM
try:
    import litellm
    # Set verbose to False if it's a function, otherwise ignore
    if hasattr(litellm, 'set_verbose') and callable(litellm.set_verbose):
        litellm.set_verbose(False)
except ImportError:
    litellm = None
    logging.error("LiteLLM is not installed. Please install it with 'pip install litellm'.")

# TTS
try:
    from kokoro import KPipeline
except ImportError:
    KPipeline = None
    logging.error("Kokoro TTS is not installed. Please install it with 'pip install kokoro-tts'.")

# VAD
try:
    from silero_vad import load_silero_vad, get_speech_timestamps
except ImportError:
    load_silero_vad = None
    get_speech_timestamps = None
    logging.error("Silero VAD is not installed. Please install it with 'pip install silero-vad'.")

# Audio preprocessing
try:
    from scipy import signal
except ImportError:
    signal = None
    logging.warning("scipy not installed, audio preprocessing will be disabled.")

# Import configuration
try:
    from config import get_gemini_api_key, validate_gemini_config
    # Validate Gemini configuration on startup
    if validate_gemini_config():
        print("✅ Gemini configuration validated successfully")
    else:
        print("⚠️  Gemini configuration issues detected")
except ImportError:
    print("⚠️  Could not import config module")
    # Fallback to direct environment variable
    if 'GEMINI_API_KEY' not in os.environ:
        os.environ['GEMINI_API_KEY'] = "AIzaSyD-987BdBsdKnCa7oWZktY9_1K27hS-qY8"
        print("⚠️  Using default Gemini API key. For production, set GEMINI_API_KEY environment variable.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def read_resume_content(resume_file: str) -> str:
    with open(resume_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Pydantic models
class ModelSelectionRequest(BaseModel):
    interview_topic: str
    resume_file: str
    asr_model: str = "openai/whisper-medium"  # Updated default
    llm_provider: str = "ollama"
    llm_model: str = "llama3.2:1b"

class AudioChunk(BaseModel):
    type: str
    audio_data: str
    sample_rate: int
    chunk_size: int

class InterviewResponse(BaseModel):
    transcription: str
    response: str
    audio_response: Optional[str] = None

# Add new models for recording upload after the existing models
class RecordingUploadResponse(BaseModel):
    success: bool
    file_path: str
    file_size: int
    message: str

class RecordingInfo(BaseModel):
    session_id: str
    file_path: str
    file_size: int
    upload_time: datetime
    duration: Optional[float] = None
    public_url: Optional[str] = None
    bucket: Optional[str] = None
    storage_type: str = "supabase"  # "supabase" or "local"


# Global session manager
class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.session_counters: Dict[str, int] = {}
        # Add recordings storage
        self.recordings: Dict[str, RecordingInfo] = {}
        
        # Create recordings directory for fallback local storage
        self.recordings_dir = Path("recordings")
        self.recordings_dir.mkdir(exist_ok=True)
        logger.info(f"Local recordings directory: {self.recordings_dir.absolute()}")
        
        # Log Supabase availability
        if SUPABASE_AVAILABLE and supabase_config.client:
            logger.info("Supabase storage is available for recordings")
        else:
            logger.warning("Supabase storage is not available, using local storage only")
    
    def create_session(self, session_id: str, model_request: ModelSelectionRequest) -> Dict:
        """Create a new interview session with all necessary components."""
        logger.info(f"Creating session {session_id}")
        
        # Initialize session data
        session_data = {
            "id": session_id,
            "status": "active",
            "created_at": datetime.now(),
            "model_request": model_request,
            "conversation_history": [],
            "audio_queue": queue.Queue(maxsize=1000),  # Audio buffer queue
            "current_utterance": [],  # Current utterance buffer
            "silence_chunks": 0,  # Silence counter
            "is_processing": False,  # Processing flag
            "processing_thread": None,  # Background processing thread
            "chunks_per_second": 16000 / 512,  # Calculate chunks per second (31.25 chunks/sec for 32ms chunks)
            "silent_chunks_threshold": int(2.0 * (16000 / 512)),  # 2 seconds of silence (62.5 chunks)
        }
        
        # Initialize processors
        try:
            logger.info(f"Initializing ASR processor for session {session_id}")
            session_data["asr_processor"] = ASRProcessor(model_request.asr_model)
            
            logger.info(f"Initializing VAD processor for session {session_id}")
            session_data["vad_processor"] = VADProcessor()
            
            logger.info(f"Initializing LLM client for session {session_id}")
            session_data["llm_client"] = LiteLLMClient()
            
            logger.info(f"Initializing TTS processor for session {session_id}")
            session_data["tts_processor"] = TTSProcessor()
            
            # Load resume content
            logger.info(f"Loading resume content for session {session_id}")
            session_data["resume_content"] = read_resume_content(model_request.resume_file)
            
            # Create initial conversation prompt
            logger.info(f"Creating initial prompt for session {session_id}")
            initial_prompt = self.create_initial_prompt(
                model_request.interview_topic,
                session_data["resume_content"]
            )
            session_data["conversation_history"] = [{"role": "system", "content": initial_prompt}, {"role": "user", "content": "Hello"}]
            # session_data["conversation_history"].append({"role": "user", "content": "Hello"})

            # --- NEW: Generate opening question and TTS ---
            try:
                model_info = {
                    "provider": model_request.llm_provider,
                    "model": model_request.llm_model
                }
                # Generate the first question
                opening_question = generate_next_question(
                    session_data["llm_client"],
                    session_data["conversation_history"],
                    model_info,
                    model_request.interview_topic,
                    session_data["resume_content"]
                )
                if opening_question:
                    # Get conversational response (natural phrasing)
                    conversational_response = get_conversational_response(
                        session_data["llm_client"],
                        opening_question,
                        session_data["conversation_history"],
                        model_info
                    )
                    if conversational_response:
                        session_data["conversation_history"].append({"role": "assistant", "content": conversational_response})
                        # Generate TTS
                        audio_base64, _ = session_data["tts_processor"].synthesize(conversational_response)
                        from pydantic import BaseModel
                        class _TmpResponse(BaseModel):
                            transcription: str
                            response: str
                            audio_response: str
                        session_data["last_response"] = _TmpResponse(
                            transcription="",  # No user speech yet
                            response=conversational_response,
                            audio_response=audio_base64 or ""
                        )
            except Exception as e:
                logger.error(f"Failed to generate opening question: {e}")
            
            self.sessions[session_id] = session_data
            logger.info(f"Session {session_id} created successfully")
            return session_data
            
        except Exception as e:
            logger.error(f"Failed to create session {session_id}: {e}")
            # Clean up any partially created session
            if session_id in self.sessions:
                del self.sessions[session_id]
            raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")
    
    def get_session(self, session_id: str) -> Dict:
        """Get session data."""
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        return self.sessions[session_id]
    
    def delete_session(self, session_id: str):
        """Delete a session and clean up resources."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Stop processing thread if running
            if session.get("processing_thread") and session["processing_thread"].is_alive():
                session["processing_thread"].join(timeout=1)
            
            # Clear audio queue
            while not session["audio_queue"].empty():
                try:
                    session["audio_queue"].get_nowait()
                except queue.Empty:
                    break
            
            del self.sessions[session_id]
            logger.info(f"Session {session_id} deleted")
    
    def load_resume_content(self, resume_file: str) -> str:
        """Load resume content from file."""
        try:
            # For now, return a placeholder. In production, implement actual file reading
            return f"Resume content from {resume_file}"
        except Exception as e:
            logger.error(f"Failed to load resume: {e}")
            return "Resume content not available"
    
    def create_initial_prompt(self, interview_topic: str, resume_content: str) -> str:
        """Create the initial system prompt for the interview."""
        return f"""You are Serin, an interviewer. Your goal is to understand the user's background and skills. Based on what the user says, ask a relevant follow-up question to gather more details. For example, if the user says 'I made a machine learning model,' you should ask something like, 'Great! Which type of machine learning model did you make?' Do not ask more than one question. The one question should be a followup to what the user said. Do not assume anything else that what is given to you. If you do not understand the user's response, ask for clarification. Do not ask for clarification if you understand the user's response. If you think that resume does not have adequate information for the given topic, you can ask a general question around the topic. Do not mention about reading the resume in the conversation. First, introduce yourself as Serin. Then, ask the question.

Here is the candidate's resume:
--- START RESUME ---
{resume_content}
--- END RESUME ---

The topic for today's interview is: {interview_topic}

Based on the resume, ask one insightful question related to the topic to start the conversation."""
    
    def get_transcription_confidence(self, transcription: str, audio_quality: float) -> float:
        """Calculate transcription confidence based on multiple factors."""
        if not transcription:
            return 0.0
        
        # Factor 1: Length (too short or too long may indicate issues)
        length_score = min(len(transcription) / 50.0, 1.0)  # Normalize to 50 chars
        
        # Factor 2: Common ASR errors (repetitions, nonsense words)
        error_indicators = ['um', 'uh', 'ah', 'er', '...', '??', '!!']
        error_count = sum(transcription.lower().count(indicator) for indicator in error_indicators)
        error_score = max(0.0, 1.0 - (error_count * 0.2))
        
        # Factor 3: Word count (reasonable range)
        word_count = len(transcription.split())
        word_score = 1.0 if 2 <= word_count <= 50 else 0.5
        
        # Factor 4: Audio quality
        quality_score = min(audio_quality * 2, 1.0)  # Scale quality to 0-1
        
        # Combine factors with weights
        confidence = (
            length_score * 0.3 +
            error_score * 0.3 +
            word_score * 0.2 +
            quality_score * 0.2
        )
        
        return min(confidence, 1.0)

    def save_recording(self, session_id: str, recording_data: bytes, filename: str) -> RecordingInfo:
        """Save recording to Supabase storage or local storage as fallback."""
        try:
            # Try to upload to Supabase first
            if SUPABASE_AVAILABLE and supabase_config.client:
                try:
                    upload_result = supabase_config.upload_recording(session_id, recording_data, filename)
                    
                    # Create recording info with Supabase data
                    recording_info = RecordingInfo(
                        session_id=session_id,
                        file_path=upload_result["file_path"],
                        file_size=upload_result["file_size"],
                        upload_time=datetime.now(),
                        public_url=upload_result["public_url"],
                        bucket=upload_result["bucket"],
                        storage_type="supabase"
                    )
                    
                    # Store recording info
                    self.recordings[session_id] = recording_info
                    
                    logger.info(f"Recording uploaded to Supabase for session {session_id}: {upload_result['file_path']} ({upload_result['file_size']} bytes)")
                    return recording_info
                    
                except Exception as supabase_error:
                    logger.warning(f"Supabase upload failed, falling back to local storage: {supabase_error}")
            
            # Fallback to local storage
            session_dir = self.recordings_dir / session_id
            session_dir.mkdir(exist_ok=True)
            
            # Save the recording file locally
            file_path = session_dir / filename
            with open(file_path, 'wb') as f:
                f.write(recording_data)
            
            # Get file size
            file_size = len(recording_data)
            
            # Create recording info for local storage
            recording_info = RecordingInfo(
                session_id=session_id,
                file_path=str(file_path),
                file_size=file_size,
                upload_time=datetime.now(),
                storage_type="local"
            )
            
            # Store recording info
            self.recordings[session_id] = recording_info
            
            logger.info(f"Recording saved locally for session {session_id}: {file_path} ({file_size} bytes)")
            return recording_info
            
        except Exception as e:
            logger.error(f"Failed to save recording for session {session_id}: {e}")
            raise

    def get_recording_info(self, session_id: str) -> Optional[RecordingInfo]:
        """Get recording info for a session."""
        return self.recordings.get(session_id)


# Global session manager instance
session_manager = SessionManager()

class VADProcessor:
    def __init__(self):
        logger.info("Initializing VAD Processor")
        self.model = None
        
        if not load_silero_vad:
            logger.warning("Silero VAD not installed. Using fallback VAD.")
            return
            
        try:
            # Use the imported load_silero_vad function instead of torch.hub.load directly
            self.model = load_silero_vad()
            logger.info("VAD model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load VAD model: {e}")
            logger.warning("Using fallback VAD without Silero model")
            self.model = None

    def detect_speech_in_chunk(self, audio_chunk_float, sample_rate=16000):
        """Detect speech in a single audio chunk."""
        if len(audio_chunk_float) == 0:
            return False
        
        # Fallback VAD if Silero model is not available
        if self.model is None:
            # Simple energy-based VAD as fallback
            energy = np.sqrt(np.mean(audio_chunk_float**2))
            return energy > 0.01
        
        try:
            # Silero VAD requires exactly 512 samples for 16kHz audio (32ms)
            expected_samples = 512 if sample_rate == 16000 else 256
            
            # Check if chunk is too short for Silero VAD
            if len(audio_chunk_float) < 512:  # Minimum required by Silero
                logger.debug(f"Audio chunk too short for Silero VAD: {len(audio_chunk_float)} samples")
                # Use energy-based VAD for short chunks
                energy = np.sqrt(np.mean(audio_chunk_float**2))
                return energy > 0.01
            
            if len(audio_chunk_float) != expected_samples:
                logger.warning(f"Audio chunk size mismatch: got {len(audio_chunk_float)}, expected {expected_samples}")
                # Pad or truncate to expected size
                if len(audio_chunk_float) > expected_samples:
                    audio_chunk_float = audio_chunk_float[:expected_samples]
                else:
                    # Pad with zeros
                    padding = np.zeros(expected_samples - len(audio_chunk_float))
                    audio_chunk_float = np.concatenate([audio_chunk_float, padding])
            
            audio_tensor = torch.from_numpy(audio_chunk_float)
            speech_prob = self.model(audio_tensor, sample_rate).item()
            return speech_prob > 0.1
        except Exception as e:
            logger.error(f"VAD error: {e}")
            # Fallback to energy-based VAD
            energy = np.sqrt(np.mean(audio_chunk_float**2))
            return energy > 0.01

    def has_speech(self, audio_array, sample_rate=16000):
        """Check if audio array contains speech using more sophisticated VAD."""
        if len(audio_array) / sample_rate < 0.15:
            return False
        
        # Fallback VAD if Silero model is not available
        if self.model is None or not get_speech_timestamps:
            # Simple energy-based VAD as fallback
            energy = np.sqrt(np.mean(audio_array**2))
            return energy > 0.01
        
        try:
            speech_timestamps = get_speech_timestamps(
                torch.from_numpy(audio_array),
                self.model,
                sampling_rate=sample_rate,
                min_speech_duration_ms=150
            )
            return len(speech_timestamps) > 0
        except Exception as e:
            logger.error(f"VAD speech detection error: {e}")
            # Fallback to energy-based VAD
            energy = np.sqrt(np.mean(audio_array**2))
            return energy > 0.01
    
    def calculate_audio_quality(self, audio_array):
        """Calculate audio quality metrics (RMS, SNR, Dynamic Range, Zero-Crossing Rate)."""
        if len(audio_array) == 0:
            return 0.0
        
        try:
            # RMS (Root Mean Square) - overall volume
            rms = np.sqrt(np.mean(audio_array**2))
            
            # Dynamic range
            dynamic_range = np.max(audio_array) - np.min(audio_array)
            
            # Zero-crossing rate (indicates speech vs noise)
            zero_crossings = np.sum(np.diff(np.sign(audio_array)) != 0)
            zcr = zero_crossings / len(audio_array)
            
            # Spectral centroid (brightness of sound)
            if signal is not None:
                freqs, psd = signal.welch(audio_array, fs=16000)
                spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
            else:
                spectral_centroid = 1000  # Default value
            
            # Simple SNR approximation (signal vs noise floor)
            sorted_audio = np.sort(np.abs(audio_array))
            noise_floor = np.mean(sorted_audio[:len(sorted_audio)//4])  # Bottom 25%
            signal_level = np.mean(sorted_audio[3*len(sorted_audio)//4:])  # Top 25%
            snr = signal_level / (noise_floor + 1e-10)
            
            # Log detailed quality metrics
            logger.debug(f"Audio Quality - RMS: {rms:.4f}, SNR: {snr:.2f}, ZCR: {zcr:.4f}, Centroid: {spectral_centroid:.0f}Hz")
            
            # Combine metrics (normalize and weight)
            quality_score = (
                min(rms * 10, 1.0) * 0.3 +           # RMS weight
                min(dynamic_range * 5, 1.0) * 0.2 +  # Dynamic range weight
                min(np.log10(snr + 1) / 2, 1.0) * 0.3 +  # SNR weight
                min(zcr * 10, 1.0) * 0.1 +           # Zero-crossing rate weight
                min(spectral_centroid / 4000, 1.0) * 0.1  # Spectral centroid weight
            )
            
            return quality_score
        except Exception as e:
            logger.error(f"Audio quality calculation error: {e}")
            return 0.0

class ASRProcessor:
    def __init__(self, model_name="openai/whisper-medium"):  # Updated default
        logger.info(f"Initializing ASR Processor with model: {model_name}")
        self.model_name = model_name
        try:
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            logger.info(f"ASR model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load ASR model: {e}")
            raise

    def preprocess_audio(self, audio_array, sample_rate=16000):
        """Preprocess audio for better ASR performance."""
        if signal is None:
            return audio_array
        
        try:
            # Remove DC offset
            audio_array = audio_array - np.mean(audio_array)
            
            # Apply pre-emphasis filter to boost high frequencies
            pre_emphasis = 0.97
            emphasized_audio = np.append(audio_array[0], audio_array[1:] - pre_emphasis * audio_array[:-1])
            
            # Apply high-pass filter to remove low-frequency noise (50-300 Hz)
            nyquist = sample_rate / 2
            low_cutoff = 50  # Hz
            high_cutoff = 8000  # Hz
            b, a = signal.butter(4, [low_cutoff / nyquist, high_cutoff / nyquist], btype='band')
            filtered_audio = signal.filtfilt(b, a, emphasized_audio)
            
            # Normalize audio to prevent clipping
            max_val = np.max(np.abs(filtered_audio))
            if max_val > 0:
                normalized_audio = filtered_audio / max_val * 0.95  # Leave some headroom
            
            # Apply noise reduction using spectral subtraction (simple approach)
            # Calculate noise floor from first 100ms
            noise_samples = min(int(0.1 * sample_rate), len(normalized_audio))
            noise_floor = np.mean(np.abs(normalized_audio[:noise_samples]))
            
            # Apply noise gate
            noise_threshold = noise_floor * 3
            normalized_audio[np.abs(normalized_audio) < noise_threshold] = 0
            
            return normalized_audio
        except Exception as e:
            logger.error(f"Audio preprocessing error: {e}")
            return audio_array

    def transcribe(self, audio_array, sample_rate=16000):
        """Transcribe audio to text."""
        if len(audio_array) == 0:
            return ""
        
        start_time = time.time()
        try:
            # Preprocess audio
            processed_audio = self.preprocess_audio(audio_array, sample_rate)
            
            # Transcribe
            inputs = self.processor(processed_audio, sampling_rate=sample_rate, return_tensors="pt").to(self.device)
            with torch.no_grad():
                predicted_ids = self.model.generate(inputs.input_features)
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True, normalize=True)
            result = transcription[0].strip() if transcription else ""
            
            logger.info(f"ASR transcription completed in {time.time() - start_time:.3f}s: '{result}'")
            return result
        except Exception as e:
            logger.error(f"ASR error: {e}")
            return ""

class LiteLLMClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.api_base = base_url
        logger.info(f"Initializing LiteLLM client with base URL: {base_url}")

    def chat(self, messages, model_info, prompt_type="chat"):
        if litellm is None:
            logger.error("LiteLLM package not available.")
            return "Sorry, the language model backend is not available."
        
        start_time = time.time()
        provider = model_info.get("provider", "ollama")
        model = model_info.get("model", "llama3.2:1b")
        model_string = f"{provider}/{model}"

        kwargs = {
            "model": model_string,
            "messages": messages,
            "stream": False
        }

        if provider == "ollama":
            kwargs["api_base"] = self.api_base
        elif provider == "gemini":
            # Get Gemini API key from configuration
            try:
                from config import get_gemini_api_key
                gemini_key = get_gemini_api_key()
            except ImportError:
                gemini_key = os.environ.get('GEMINI_API_KEY')
            
            if not gemini_key:
                error_msg = "GEMINI_API_KEY not available. Please set the environment variable or check configuration."
                logger.error(error_msg)
                return "Sorry, the Gemini API key is not configured. Please set the GEMINI_API_KEY environment variable."
            
            # Set the API key for LiteLLM
            kwargs["api_key"] = gemini_key
            logger.info(f"Using Gemini model: {model_string} with API key: {gemini_key[:10]}...{gemini_key[-4:]}")

        try:
            logger.info(f"Making LiteLLM request to {model_string} with {len(messages)} messages")
            response = litellm.completion(**kwargs)
            content = response["choices"][0]["message"]["content"]
            logger.info(f"LiteLLM {prompt_type} completed in {time.time() - start_time:.3f}s")
            return content
        except Exception as e:
            logger.error(f"LiteLLM error for {model_string}: {e}")
            logger.error(f"Request kwargs: {kwargs}")
            return f"Sorry, I'm having trouble connecting to the AI model ({model_string}) for {prompt_type}. Error: {str(e)}"

class TTSProcessor:
    def __init__(self):
        if not KPipeline:
            raise ImportError("Kokoro TTS not installed.")
        logger.info("Initializing TTS Processor")
        try:
            self.pipeline = KPipeline(lang_code='a')
            logger.info("TTS pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TTS pipeline: {e}")
            raise

    def synthesize(self, text, voice="af_heart"):
        if not text:
            return None, None
        
        start_time = time.time()
        try:
            generator = self.pipeline(text, voice=voice)
            audio_chunks = [audio for _, _, audio in generator]
            
            if audio_chunks:
                full_audio = np.concatenate(audio_chunks)
                logger.info(f"TTS synthesis completed in {time.time() - start_time:.3f}s")
                
                # Convert to WAV format
                import wave
                import io
                
                # Convert to 16-bit PCM
                audio_int16 = (full_audio * 32767).astype(np.int16)
                
                # Create WAV file in memory
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(24000)  # Sample rate
                    wav_file.writeframes(audio_int16.tobytes())
                
                # Get WAV data and encode to base64
                wav_data = wav_buffer.getvalue()
                audio_base64 = base64.b64encode(wav_data).decode('utf-8')
                
                logger.info(f"WAV file created: {len(wav_data)} bytes")
                
                return audio_base64, 24000
            
            return None, None
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None, None

def process_utterance(session_id: str, audio_data: np.ndarray) -> InterviewResponse:
    """Process a complete utterance through the pipeline."""
    session = session_manager.get_session(session_id)
    
    try:
        # Transcribe audio
        transcription = session["asr_processor"].transcribe(audio_data)
        
        if not transcription:
            return InterviewResponse(transcription="", response="", audio_response=None)
        
        # Add user message to conversation history
        session["conversation_history"].append({"role": "user", "content": transcription})
        
        # Generate next question
        model_info = {
            "provider": session["model_request"].llm_provider,
            "model": session["model_request"].llm_model
        }
        
        next_question = generate_next_question(
            session["llm_client"], 
            session["conversation_history"], 
            model_info,
            session["model_request"].interview_topic,
            session["resume_content"]
        )
        
        if next_question:
            # Get conversational response
            conversational_response = get_conversational_response(
                session["llm_client"], 
                next_question, 
                session["conversation_history"], 
                model_info
            )
            
            if conversational_response:
                session["conversation_history"].append({"role": "assistant", "content": conversational_response})
                
                # Generate TTS audio
                audio_base64, _ = session["tts_processor"].synthesize(conversational_response)
                
                return InterviewResponse(
                    transcription=transcription,
                    response=conversational_response,
                    audio_response=audio_base64
                )
        
        return InterviewResponse(transcription=transcription, response="", audio_response=None)
        
    except Exception as e:
        logger.error(f"Error processing utterance for session {session_id}: {e}")
        return InterviewResponse(transcription="", response="", audio_response=None)

def generate_next_question(llm_client, conversation_history, model_info, interview_topic, resume_content):
    """Generate the next interview question."""
    logger.info("Generating next question")
    qgen_prompt = f"""You are a question generation module. Your task is to generate the next interview question based on the conversation history, the interview topic, and the candidate's resume. The question should be a logical follow-up to the previous turn in the conversation. Do not ask more than one question. Do not ask for clarification. Just generate the question.

Here is the candidate's resume:
--- START RESUME ---
{resume_content}
--- END RESUME ---

The topic for today's interview is: {interview_topic}

Based on the conversation history, generate the next question."""
    messages = conversation_history + [{"role": "system", "content": qgen_prompt}]
    return llm_client.chat(messages, model_info, prompt_type="qgen")

def get_conversational_response(llm_client, question, conversation_history, model_info):
    """Get a conversational response from the LLM."""
    logger.info("Getting conversational response")
    conversational_prompt = f"You are Serin, an interviewer. Your goal is to ask the following question in a natural and conversational way: '{question}'. Do not add any other information to the response. Just ask the question."
    messages = conversation_history + [{"role": "system", "content": conversational_prompt}]
    return llm_client.chat(messages, model_info, prompt_type="chat")

def continuous_audio_processor(session_id: str):
    """Background thread for continuous audio processing."""
    session = session_manager.get_session(session_id)
    audio_queue = session["audio_queue"]
    vad_processor = session["vad_processor"]
    
    logger.info(f"Starting continuous audio processing for session {session_id}")
    
    while session["status"] == "active":
        try:
            # Get audio chunk from queue with timeout
            audio_data_bytes = audio_queue.get(timeout=0.1)
            if audio_data_bytes is None:  # Sentinel value to stop
                break
            
            # Convert bytes to numpy array
            audio_chunk_np = np.frombuffer(audio_data_bytes, dtype=np.int16)
            audio_chunk_float = audio_chunk_np.astype(np.float32) / 32768.0
            
            # Log chunk size for debugging
            logger.debug(f"Received audio chunk: {len(audio_chunk_float)} samples")
            
            # VAD detection
            if vad_processor.detect_speech_in_chunk(audio_chunk_float):
                session["current_utterance"].append(audio_chunk_np)
                session["silence_chunks"] = 0
            else:
                if len(session["current_utterance"]) > 0:
                    session["silence_chunks"] += 1
                    if session["silence_chunks"] > session["silent_chunks_threshold"]:
                        # Process complete utterance
                        full_utterance_np = np.concatenate(session["current_utterance"], axis=0)
                        full_utterance_float = full_utterance_np.astype(np.float32) / 32768.0
                        
                        # Reset utterance buffer
                        session["current_utterance"] = []
                        session["silence_chunks"] = 0
                        
                        # Check if utterance contains speech
                        if vad_processor.has_speech(full_utterance_float):
                            logger.info(f"Processing utterance for session {session_id}")
                            session["is_processing"] = True
                            
                            # Process utterance
                            response = process_utterance(session_id, full_utterance_float)
                            
                            # Send response via WebSocket (handled by main thread)
                            session["last_response"] = response
                            session["is_processing"] = False
                        else:
                            logger.info(f"VAD rejected utterance for session {session_id}")
                            
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Error in continuous audio processing for session {session_id}: {e}")
            continue
    
    logger.info(f"Continuous audio processing stopped for session {session_id}")

# Add new endpoints for recording upload and download after the existing endpoints
@app.post("/recordings/upload/{session_id}")
async def upload_recording(
    session_id: str,
    recording: UploadFile = File(...),
    recording_type: str = Form("session")  # "session" or "audio"
):
    """Upload recording file to Supabase storage."""
    try:
        # Validate session exists
        if session_id not in session_manager.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Read recording data
        recording_data = await recording.read()
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{recording_type}_recording_{session_id}_{timestamp}.webm"
        
        # Save recording (will try Supabase first, fallback to local)
        recording_info = session_manager.save_recording(session_id, recording_data, filename)
        
        # Prepare response message
        if recording_info.storage_type == "supabase":
            message = f"Recording uploaded to Supabase successfully: {filename}"
        else:
            message = f"Recording saved locally (Supabase unavailable): {filename}"
        
        return RecordingUploadResponse(
            success=True,
            file_path=recording_info.file_path,
            file_size=recording_info.file_size,
            message=message
        )
        
    except Exception as e:
        logger.error(f"Failed to upload recording for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recordings/{session_id}")
async def get_recording_info(session_id: str):
    """Get recording information for a session."""
    try:
        recording_info = session_manager.get_recording_info(session_id)
        if not recording_info:
            raise HTTPException(status_code=404, detail="Recording not found")
        
        return recording_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get recording info for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recordings/{session_id}/list")
async def list_session_recordings(session_id: str):
    """List all recordings for a session."""
    try:
        # Validate session exists
        if session_id not in session_manager.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        recordings = []
        
        # Get local recordings
        session_dir = session_manager.recordings_dir / session_id
        if session_dir.exists():
            for file_path in session_dir.glob("*.webm"):
                recordings.append({
                    "filename": file_path.name,
                    "file_path": str(file_path),
                    "file_size": file_path.stat().st_size,
                    "storage_type": "local",
                    "upload_time": datetime.fromtimestamp(file_path.stat().st_mtime)
                })
        
        # Get Supabase recordings if available
        if SUPABASE_AVAILABLE and supabase_config.client:
            try:
                supabase_files = supabase_config.list_session_recordings(session_id)
                for filename in supabase_files:
                    public_url = supabase_config.get_recording_url(session_id, filename)
                    recordings.append({
                        "filename": filename,
                        "file_path": f"{session_id}/{filename}",
                        "public_url": public_url,
                        "storage_type": "supabase",
                        "bucket": supabase_config.bucket_name
                    })
            except Exception as e:
                logger.warning(f"Failed to list Supabase recordings: {e}")
        
        return {
            "session_id": session_id,
            "recordings": recordings,
            "total_count": len(recordings)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list recordings for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recordings/{session_id}/download")
async def download_recording(session_id: str):
    """Download recording file or return Supabase URL."""
    try:
        recording_info = session_manager.get_recording_info(session_id)
        if not recording_info:
            raise HTTPException(status_code=404, detail="Recording not found")
        
        # If stored in Supabase, return the public URL
        if recording_info.storage_type == "supabase" and recording_info.public_url:
            return {
                "storage_type": "supabase",
                "public_url": recording_info.public_url,
                "file_path": recording_info.file_path,
                "file_size": recording_info.file_size
            }
        
        # If stored locally, serve the file
        file_path = Path(recording_info.file_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Recording file not found")
        
        return FileResponse(
            path=file_path,
            filename=file_path.name,
            media_type="video/webm"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download recording for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/recordings/{session_id}/{filename}")
async def delete_recording(session_id: str, filename: str):
    """Delete a recording file."""
    try:
        # Validate session exists
        if session_id not in session_manager.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        success = False
        message = ""
        
        # Try to delete from Supabase first
        if SUPABASE_AVAILABLE and supabase_config.client:
            try:
                if supabase_config.delete_recording(session_id, filename):
                    success = True
                    message = f"Recording deleted from Supabase: {filename}"
                else:
                    message = f"Failed to delete recording from Supabase: {filename}"
            except Exception as e:
                logger.warning(f"Supabase delete failed: {e}")
                message = f"Supabase delete failed: {str(e)}"
        
        # Try to delete from local storage
        session_dir = session_manager.recordings_dir / session_id
        local_file_path = session_dir / filename
        if local_file_path.exists():
            try:
                local_file_path.unlink()
                success = True
                message += f" Recording deleted from local storage: {filename}"
            except Exception as e:
                logger.error(f"Failed to delete local recording: {e}")
                message += f" Local delete failed: {str(e)}"
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Recording not found: {filename}")
        
        return {
            "success": success,
            "message": message,
            "filename": filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete recording {filename} for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# API endpoints
@app.post("/sessions/create")
async def create_session(model_request: ModelSelectionRequest):
    """Create a new interview session."""
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        session_data = session_manager.create_session(session_id, model_request)
        
        # Start background processing thread
        processing_thread = threading.Thread(
            target=continuous_audio_processor,
            args=(session_id,),
            daemon=True
        )
        processing_thread.start()
        session_data["processing_thread"] = processing_thread
        
        logger.info(f"Created session {session_id}")
        return {"session_id": session_id, "status": "created"}
        
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session information."""
    try:
        session = session_manager.get_session(session_id)
        return {
            "session_id": session_id,
            "status": session["status"],
            "created_at": session["created_at"],
            "conversation_length": len(session["conversation_history"])
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    try:
        session_manager.delete_session(session_id)
        return {"status": "deleted"}
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_available_models():
    """Get available ASR and LLM models."""
    from model_selector import get_available_asr_models, get_available_llm_models
    
    return {
        "asr_models": get_available_asr_models(),
        "llm_models": get_available_llm_models()
    }

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    logger.info(f"WebSocket connection established for session {session_id}")
    await websocket.accept()
    
    try:
        session = session_manager.get_session(session_id)
        if session["status"] != "active":
            logger.warning(f"Session {session_id} is not active, status: {session['status']}")
            await websocket.send_text(json.dumps({"error": "Session is not active"}))
            return
        
        logger.info(f"WebSocket ready for session {session_id}")

        # --- NEW: Send opening AI message if available ---
        if session.get("last_response"):
            response = session["last_response"]
            session["last_response"] = None
            response_json = response.json()
            logger.debug(f"Sending opening response to session {session_id}: {len(response_json)} chars")
            await websocket.send_text(response_json)
            logger.info(f"Opening response sent successfully to session {session_id}")
        
        while True:
            logger.debug(f"Waiting for audio data from session {session_id}")
            data = await websocket.receive_text()
            logger.debug(f"Received audio data from session {session_id}: {len(data)} chars")
            
            try:
                audio_data = json.loads(data)
                logger.debug(f"Parsed audio data: type={audio_data.get('type')}, sample_rate={audio_data.get('sample_rate')}")
                
                if audio_data.get('type') == 'audio_chunk':
                    # Decode audio data
                    audio_bytes = base64.b64decode(audio_data['audio_data'])
                    
                    # Add to audio queue for processing
                    try:
                        session["audio_queue"].put_nowait(audio_bytes)
                        logger.debug(f"Added audio chunk to queue for session {session_id}")
                    except queue.Full:
                        logger.warning(f"Audio queue full for session {session_id}, dropping chunk")
                    
                    # Check if there's a response ready
                    if session.get("last_response") and not session["is_processing"]:
                        response = session["last_response"]
                        session["last_response"] = None
                        response_json = response.json()
                        logger.debug(f"Sending response to session {session_id}: {len(response_json)} chars")
                        await websocket.send_text(response_json)
                        logger.info(f"Response sent successfully to session {session_id}")
            
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error for session {session_id}: {e}")
                await websocket.send_text(json.dumps({"error": "Invalid JSON format"}))
            except Exception as e:
                logger.error(f"Error processing audio for session {session_id}: {e}")
                await websocket.send_text(json.dumps({"error": str(e)}))
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        await websocket.send_text(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 