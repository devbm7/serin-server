import streamlit as st
import threading
import queue
import numpy as np
import soundfile as sf
import tempfile
import os
import base64
import time
import logging
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline_debug.log')
    ]
)
logger = logging.getLogger(__name__)

# Audio recording
import pyaudio

# ASR
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

# Ollama
import requests
import json

# TTS
from kokoro import KPipeline

# VAD
from silero_vad import load_silero_vad, get_speech_timestamps

# Global message queue for inter-thread communication
message_queue = queue.Queue()

# --- Initialize Session State ---
def initialize_session_state():
    logger.info("Initializing session state")
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'recorder' not in st.session_state:
        st.session_state.recorder = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'status_message' not in st.session_state:
        st.session_state.status_message = "Ready to start"
    if 'new_messages' not in st.session_state:
        st.session_state.new_messages = deque(maxlen=10)  # Thread-safe queue for new messages
    if 'silence_threshold' not in st.session_state:
        st.session_state.silence_threshold = 1.0
    if 'tts_voice' not in st.session_state:
        st.session_state.tts_voice = "af_heart"
    if 'ollama_model' not in st.session_state:
        st.session_state.ollama_model = "llama3.2:1b"
    if 'processing_thread' not in st.session_state:
        st.session_state.processing_thread = None
    if 'shared_state' not in st.session_state:
        st.session_state.shared_state = {
            'is_recording': False,
            'processing': False,
            'status_message': "Ready to start"
        }
    logger.info("Session state initialized successfully")

initialize_session_state()

# --- Core Application Classes ---

class AudioRecorder:
    """
    A class to handle audio recording using PyAudio in a separate thread.
    Uses a thread-safe queue to pass audio data from the recording thread
    to the processing thread.
    """
    def __init__(self, sample_rate=16000, channels=1, chunk_size=512):  # Changed from 1024 to 512
        logger.info(f"Initializing AudioRecorder: sample_rate={sample_rate}, channels={channels}, chunk_size={chunk_size}")
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue(maxsize=50)  # Reduced queue size to prevent memory issues
        self.is_recording = False
        self.stream = None
        self.p = pyaudio.PyAudio()
        self.silence_threshold = st.session_state.silence_threshold
        self.chunks_recorded = 0
        logger.info("AudioRecorder initialized successfully")

    def start_continuous_recording(self):
        logger.info("Starting continuous recording")
        self.is_recording = True
        self.chunks_recorded = 0
        # Clear the queue before starting
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                continue
        
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            self.stream.start_stream()
            logger.info("Recording started successfully")
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            self.is_recording = False

    def stop_recording(self):
        logger.info("Stopping recording")
        self.is_recording = False
        if self.stream and self.stream.is_active():
            try:
                self.stream.stop_stream()
                self.stream.close()
                logger.info("Audio stream closed successfully")
            except Exception as e:
                logger.error(f"Error stopping recording: {e}")
        # Use a sentinel value to signal the processing thread to stop
        try:
            self.audio_queue.put(None, timeout=1)
            logger.info("Sentinel value added to queue")
        except queue.Full:
            logger.warning("Queue was full when adding sentinel value")
        logger.info(f"Recording stopped. Total chunks recorded: {self.chunks_recorded}")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            self.chunks_recorded += 1
            try:
                # Non-blocking put to prevent blocking the audio callback
                self.audio_queue.put_nowait(in_data)
                if self.chunks_recorded % 200 == 0:  # Log every 200 chunks (less frequent)
                    logger.debug(f"Audio callback: recorded {self.chunks_recorded} chunks, queue size: {self.audio_queue.qsize()}")
            except queue.Full:
                # If queue is full, skip this chunk to prevent blocking
                logger.debug("Audio queue full, skipping chunk")
        return (None, pyaudio.paContinue)

class VADProcessor:
    def __init__(self):
        logger.info("Initializing VAD Processor")
        # This model is lightweight and fast
        try:
            self.model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                           model='silero_vad',
                                           force_reload=False)
            logger.info("VAD model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load VAD model: {e}")
            raise

    def detect_speech_in_chunk(self, audio_chunk_float, sample_rate=16000):
        if len(audio_chunk_float) == 0:
            return False
        try:
            # Calculate audio level for debugging
            audio_level = np.abs(audio_chunk_float).mean()
            
            audio_tensor = torch.from_numpy(audio_chunk_float)
            speech_prob = self.model(audio_tensor, sample_rate).item()
            is_speech = speech_prob > 0.1  # Lowered threshold from 0.3 to 0.1 for noisy environments
            
            # Log audio level and VAD results
            if audio_level > 0.005:  # Lowered threshold to see more audio activity
                logger.debug(f"VAD chunk detection: level={audio_level:.4f}, prob={speech_prob:.3f}, is_speech={is_speech}")
            
            return is_speech
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return False

    def has_speech(self, audio_array, sample_rate=16000, min_speech_duration=0.15):  # Reduced from 0.25 to 0.15
        if len(audio_array) / sample_rate < min_speech_duration:
            logger.debug(f"Audio too short: {len(audio_array) / sample_rate:.2f}s < {min_speech_duration}s")
            return False
        
        try:
            # For noisy environments, use a more lenient approach
            # Check if there's any significant audio activity
            audio_level = np.abs(audio_array).mean()
            if audio_level < 0.001:  # Very low audio level
                logger.debug(f"Audio level too low: {audio_level:.4f}")
                return False
            
            speech_timestamps = get_speech_timestamps(
                torch.from_numpy(audio_array),
                self.model,
                sampling_rate=sample_rate,
                min_speech_duration_ms=150,  # Reduced from default
                min_silence_duration_ms=100,  # Reduced from default
                speech_pad_ms=30  # Reduced padding
            )
            has_speech = len(speech_timestamps) > 0
            logger.debug(f"VAD speech detection: duration={len(audio_array) / sample_rate:.2f}s, level={audio_level:.4f}, timestamps={len(speech_timestamps)}, has_speech={has_speech}")
            return has_speech
        except Exception as e:
            logger.error(f"VAD speech detection error: {e}")
            return False

class ASRProcessor:
    def __init__(self):
        logger.info("Initializing ASR Processor")
        # Using a smaller, faster model for better real-time performance
        model_name = "openai/whisper-small"
        try:
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            logger.info(f"ASR model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load ASR model: {e}")
            raise

    def transcribe(self, audio_array, sample_rate=16000):
        if len(audio_array) == 0:
            logger.warning("Empty audio array provided to ASR")
            return ""
        try:
            logger.info(f"Starting transcription: audio_length={len(audio_array)}, duration={len(audio_array)/sample_rate:.2f}s")
            inputs = self.processor(audio_array, sampling_rate=sample_rate, return_tensors="pt")
            inputs = inputs.to(self.device)
            with torch.no_grad():
                predicted_ids = self.model.generate(inputs.input_features)
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True, normalize=True)
            result = transcription[0].strip() if transcription else ""
            logger.info(f"Transcription completed: '{result}'")
            return result
        except Exception as e:
            logger.error(f"ASR error: {e}")
            return ""

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        logger.info(f"Initializing Ollama client with base URL: {base_url}")

    def chat_stream(self, messages, model):
        url = f"{self.base_url}/api/chat"
        payload = {"model": model, "messages": messages, "stream": True}
        logger.info(f"Sending chat request to Ollama: model={model}, messages_count={len(messages)}")
        try:
            response = requests.post(url, json=payload, stream=True, timeout=30)
            response.raise_for_status()
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if 'message' in data and 'content' in data['message']:
                            content = data['message']['content']
                            full_response += content
                            yield content
                        if data.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
            logger.info(f"Ollama response completed: length={len(full_response)}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama connection error: {e}")
            yield "Sorry, I'm having trouble connecting to the AI model."

class TTSProcessor:
    def __init__(self):
        logger.info("Initializing TTS Processor")
        try:
            self.pipeline = KPipeline(lang_code='a')
            logger.info("TTS pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TTS pipeline: {e}")
            raise

    def synthesize(self, text, voice):
        if not text: 
            logger.warning("Empty text provided to TTS")
            return None, None
        try:
            logger.info(f"Starting TTS synthesis: text_length={len(text)}, voice={voice}")
            generator = self.pipeline(text, voice=voice)
            audio_chunks = [audio for _, _, audio in generator]
            if audio_chunks:
                full_audio = np.concatenate(audio_chunks)
                logger.info(f"TTS synthesis completed: audio_length={len(full_audio)}")
                return full_audio, 24000
            logger.warning("No audio chunks generated by TTS")
            return None, None
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None, None

# --- Model and Utility Loading ---

@st.cache_resource
def load_models():
    logger.info("Loading all models")
    try:
        asr = ASRProcessor()
        ollama = OllamaClient()
        tts = TTSProcessor()
        vad = VADProcessor()
        logger.info("All models loaded successfully")
        return asr, ollama, tts, vad
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

def audio_to_base64(audio_data, sample_rate):
    logger.debug(f"Converting audio to base64: length={len(audio_data)}, sample_rate={sample_rate}")
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        sf.write(tmp_file.name, audio_data, sample_rate)
        with open(tmp_file.name, 'rb') as f:
            audio_base64 = base64.b64encode(f.read()).decode()
    os.unlink(tmp_file.name)
    logger.debug("Audio converted to base64 successfully")
    return audio_base64

# --- Core Processing Logic ---

def process_audio_continuously(asr, ollama, tts, vad, recorder, silence_threshold, ollama_model, tts_voice, shared_state, message_queue):
    """
    The main background thread function.
    Consumes audio from the queue and uses a VAD to detect utterances.
    """
    logger.info("Starting continuous audio processing thread")
    current_utterance = []
    silence_chunks = 0
    chunks_per_second = recorder.sample_rate / recorder.chunk_size
    silent_chunks_threshold = int(silence_threshold * chunks_per_second)
    
    logger.info(f"Processing parameters: chunks_per_second={chunks_per_second}, silent_chunks_threshold={silent_chunks_threshold}")
    logger.info(f"Initial shared_state: {shared_state}")
    
    chunks_processed = 0
    
    while shared_state['is_recording']:
        try:
            logger.debug(f"Waiting for audio data... (chunks_processed={chunks_processed})")
            audio_data_bytes = recorder.audio_queue.get(timeout=0.1)
            if audio_data_bytes is None: 
                logger.info("Received sentinel value, stopping processing")
                break

            chunks_processed += 1
            audio_chunk_np = np.frombuffer(audio_data_bytes, dtype=np.int16)
            audio_chunk_float = audio_chunk_np.astype(np.float32) / 32768.0
            
            # Monitor audio levels
            audio_level = np.abs(audio_chunk_float).mean()
            if chunks_processed % 50 == 0:  # Log every 50 chunks
                logger.info(f"Audio level check: chunk={chunks_processed}, level={audio_level:.4f}")
            
            if chunks_processed % 100 == 0:
                logger.debug(f"Processed {chunks_processed} audio chunks")
            
            if vad.detect_speech_in_chunk(audio_chunk_float):
                current_utterance.append(audio_chunk_np)
                silence_chunks = 0
                if len(current_utterance) == 1:  # First chunk of speech
                    logger.debug("Speech detected, starting utterance collection")
            else:
                if len(current_utterance) > 0:
                    silence_chunks += 1
                    if silence_chunks > silent_chunks_threshold:
                        logger.info(f"Silence threshold reached ({silence_chunks} chunks), processing utterance")
                        # Set processing flag
                        shared_state['processing'] = True
                        shared_state['status_message'] = "🤖 Processing..."
                        
                        full_utterance_np = np.concatenate(current_utterance, axis=0)
                        full_utterance_float = full_utterance_np.astype(np.float32) / 32768.0
                        
                        utterance_duration = len(full_utterance_float) / recorder.sample_rate
                        logger.info(f"Utterance collected: duration={utterance_duration:.2f}s, chunks={len(current_utterance)}")
                        
                        current_utterance = []
                        silence_chunks = 0
                        
                        if vad.has_speech(full_utterance_float):
                            logger.info("VAD confirmed speech, processing utterance")
                            process_utterance(asr, ollama, tts, full_utterance_float, ollama_model, tts_voice, message_queue)
                        else:
                            # Fallback: if VAD rejects but audio level is significant, process anyway
                            audio_level = np.abs(full_utterance_float).mean()
                            if audio_level > 0.002:  # Significant audio activity
                                logger.info(f"VAD rejected but audio level is significant ({audio_level:.4f}), processing anyway")
                                process_utterance(asr, ollama, tts, full_utterance_float, ollama_model, tts_voice, message_queue)
                            else:
                                logger.info("VAD rejected utterance as non-speech")
                        
                        # Reset processing flag
                        shared_state['processing'] = False
                        shared_state['status_message'] = "🔴 Listening..."
        
        except queue.Empty:
            # Add a small delay to prevent busy waiting
            time.sleep(0.01)
            continue
        except Exception as e:
            logger.error(f"Error in continuous processing: {e}")
            # Add a small delay before continuing to prevent rapid error loops
            time.sleep(0.1)
            continue
    
    logger.info(f"Continuous audio processing thread ended. Total chunks processed: {chunks_processed}")

def process_utterance(asr, ollama, tts, audio_data, ollama_model, tts_voice, message_queue):
    """Processes a single, complete audio utterance."""
    logger.info("Starting utterance processing")
    try:
        transcription = asr.transcribe(audio_data)
        if transcription:
            logger.info(f"User transcription: '{transcription}'")
            user_message = {"role": "user", "content": transcription}
            message_queue.put(user_message)
            logger.info("User message added to queue")
            
            # Prepare messages for Ollama
            all_messages = list(st.session_state.messages) + [user_message]
            ollama_messages = [{"role": msg["role"], "content": msg["content"]} for msg in all_messages]
            logger.info(f"Prepared {len(ollama_messages)} messages for Ollama")
            
            full_response = "".join(ollama.chat_stream(ollama_messages, model=ollama_model))
            
            if full_response:
                logger.info(f"Ollama response: '{full_response[:100]}...'")
                audio_output, sample_rate = tts.synthesize(full_response, voice=tts_voice)
                if audio_output is not None:
                    audio_base64 = audio_to_base64(audio_output, sample_rate)
                    assistant_message = {
                        "role": "assistant", 
                        "content": full_response,
                        "audio": audio_base64
                    }
                    message_queue.put(assistant_message)
                    logger.info("Assistant message with audio added to queue")
                else:
                    logger.warning("TTS failed to generate audio")
            else:
                logger.warning("Ollama returned empty response")
        else:
            logger.warning("ASR returned empty transcription")
    except Exception as e:
        logger.error(f"Error in processing utterance: {e}")
    finally:
        logger.info("Utterance processing completed")

# --- Streamlit UI ---

def main():
    logger.info("Starting Streamlit application")
    st.set_page_config(page_title="Conversational AI", layout="wide")
    st.title("🎙️ Conversational AI Pipeline")
    
    # Load models
    logger.info("Loading models in main thread")
    asr, ollama, tts, vad = load_models()
    
    # Poll the message queue for new messages from the background thread
    while not message_queue.empty():
        msg = message_queue.get()
        st.session_state.new_messages.append(msg)
        st.session_state.messages.append(msg)
    
    if st.session_state.new_messages:
        logger.info(f"Processed {len(st.session_state.new_messages)} new messages")
    
    # Sync shared state with session state
    st.session_state.shared_state['is_recording'] = st.session_state.is_recording
    st.session_state.processing = st.session_state.shared_state['processing']
    st.session_state.status_message = st.session_state.shared_state['status_message']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Conversation")
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    if message["role"] == "assistant" and "audio" in message:
                        audio_html = f"""
                        <audio controls autoplay>
                            <source src="data:audio/wav;base64,{message["audio"]}" type="audio/wav">
                        </audio>"""
                        st.markdown(audio_html, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Voice Controls")
        
        # Status display
        status_container = st.empty()
        status_container.info(st.session_state.status_message)
        
        if not st.session_state.is_recording:
            if st.button("🎤 Start Listening", type="primary"):
                logger.info("Start Listening button clicked")
                st.session_state.is_recording = True
                st.session_state.shared_state['is_recording'] = True
                st.session_state.recorder = AudioRecorder()
                st.session_state.recorder.start_continuous_recording()
                st.session_state.status_message = "🔴 Listening..."
                st.session_state.shared_state['status_message'] = "🔴 Listening..."
                
                # Start processing thread with parameters passed directly
                logger.info("Starting processing thread")
                processing_thread = threading.Thread(
                    target=process_audio_continuously,
                    args=(
                        asr, 
                        ollama, 
                        tts, 
                        vad, 
                        st.session_state.recorder,
                        st.session_state.silence_threshold,
                        st.session_state.ollama_model,
                        st.session_state.tts_voice,
                        st.session_state.shared_state,
                        message_queue  # <-- add this
                    ),
                    daemon=True
                )
                processing_thread.start()
                st.session_state.processing_thread = processing_thread
                logger.info("Processing thread started successfully")
        else:
            if st.button("⏹️ Stop Listening"):
                logger.info("Stop Listening button clicked")
                st.session_state.is_recording = False
                st.session_state.shared_state['is_recording'] = False
                if st.session_state.recorder:
                    st.session_state.recorder.stop_recording()
                st.session_state.status_message = "Ready to start"
                st.session_state.shared_state['status_message'] = "Ready to start"
                st.session_state.processing = False
                st.session_state.shared_state['processing'] = False
                logger.info("Recording stopped by user")
        
        if st.button("🗑️ Clear Conversation"):
            logger.info("Clear Conversation button clicked")
            st.session_state.messages = []
            st.session_state.new_messages.clear()
            
        st.subheader("Settings")
        st.session_state.silence_threshold = st.slider(
            "Silence Threshold (s)", 0.5, 5.0, st.session_state.silence_threshold, 0.1)
        
        st.session_state.tts_voice = st.selectbox(
            "TTS Voice", ["af_heart", "af_neutral", "af_calm"], 
            index=["af_heart", "af_neutral", "af_calm"].index(st.session_state.tts_voice))
            
        st.session_state.ollama_model = st.text_input(
            "Ollama Model", value=st.session_state.ollama_model)

        if st.session_state.recorder:
            st.session_state.recorder.silence_threshold = st.session_state.silence_threshold
        
        st.subheader("Model Status")
        try:
            requests.get("http://localhost:11434/api/tags", timeout=2)
            st.success("✅ Ollama Connected")
            logger.debug("Ollama connection check successful")
        except:
            st.error("❌ Ollama Not Available")
            logger.warning("Ollama connection check failed")
    
    # Auto-rerun only when recording to update the UI
    if st.session_state.is_recording:
        time.sleep(0.1)
        st.rerun()

if __name__ == "__main__":
    main()
