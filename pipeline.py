import threading
import queue
import numpy as np
import soundfile as sf
import tempfile
import os
import base64
import time
import logging
import json
from datetime import datetime
from collections import deque
import sys
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme
from model_selector import get_available_asr_models, get_available_mainllm_models
import PyPDF2
import cv2
import supervision as sv
from rfdetr import RFDETRBase, RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES
import yaml
import wave

os.environ['LITELLM_LOG'] = 'ERROR'
os.environ['GEMINI_API_KEY'] = 'GEMINI'

# Load config
with open('pipeline_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config['paths']['log_file'])
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
# LiteLLM
try:
    import litellm
    # litellm.set_verbose(False)
except ImportError:
    litellm = None
    logger.error("LiteLLM is not installed. Please install it with 'pip install litellm'.")

# TTS
try:
    from kokoro import KPipeline
except ImportError:
    KPipeline = None
    logger.error("Kokoro TTS is not installed. Please install it with 'pip install kokoro-tts'.")


# VAD
try:
    from silero_vad import load_silero_vad, get_speech_timestamps
except ImportError:
    load_silero_vad = None
    get_speech_timestamps = None
    logger.error("Silero VAD is not installed. Please install it with 'pip install silero-vad'.")


# For playing audio (TTS response)
try:
    import sounddevice as sd
except ImportError:
    sd = None
    logger.warning("sounddevice not installed, TTS audio will not be played.")

# Rich console for pretty CLI output
custom_theme = Theme({
    "user": "bold cyan",
    "assistant": "bold magenta",
    "prompt": "bold yellow",
    "info": "dim white"
})
console = Console(theme=custom_theme)

# Available ASR models
AVAILABLE_ASR_MODELS = get_available_asr_models()
AVAILABLE_MAINLLM_MODELS = get_available_mainllm_models()

def select_llm_model():
    """Display available LLM models and let user select one."""
    console.print("\n[bold green]==== LLM Model Selection ====[/bold green]")
    console.print("[info]Please select an LLM model for this session:[/info]\n")
    
    model_choices = {}
    for i, (key, model_info) in enumerate(AVAILABLE_MAINLLM_MODELS.items(), 1):
        model_choices[str(i)] = model_info
        console.print(f"[bold]{i}.[/bold] {model_info['model']}")
        console.print(f"     {model_info['provider']}\n")
    
    console.print("[bold]0.[/bold] Use default model (llama3)\n")
    
    while True:
        try:
            choice = input(f"Enter your choice (0-{len(model_choices)}): ").strip()
            if choice == "0":
                console.print("[bold green]Using default LLM model: llama3[/bold green]\n")
                return {"provider": "ollama", "model": "llama3"}
            elif choice in model_choices:
                selected_model_info = model_choices[choice]
                console.print(f"[bold green]Selected LLM model: {selected_model_info['model']}[/bold green]\n")
                return selected_model_info
            else:
                console.print(f"[bold red]Invalid choice. Please enter a number between 0-{len(model_choices)}.[/bold red]")
        except KeyboardInterrupt:
            console.print("\n[bold red]Using default model: llama3[/bold red]\n")
            return {"provider": "ollama", "model": "llama3"}
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")
            console.print("[bold red]Using default model: llama3[/bold red]\n")
            return {"provider": "ollama", "model": "llama3"}

def select_asr_model():
    """Display available ASR models and let user select one."""
    console.print("\n[bold green]==== ASR Model Selection ====[/bold green]")
    console.print("[info]Please select an ASR model for this session:[/info]\n")
    
    model_keys = list(AVAILABLE_ASR_MODELS.keys())
    for key in model_keys:
        model_info = AVAILABLE_ASR_MODELS[key]
        console.print(f"[bold]{key}.[/bold] {model_info['name']}")
        console.print(f"     {model_info['description']}")
        console.print(f"     Language: {model_info['language']}\n")
    
    console.print("[bold]0.[/bold] Use default model (openai/whisper-small)\n")
    
    while True:
        try:
            choice = input(f"Enter your choice (0-{len(model_keys)}): ").strip()
            if choice == "0":
                console.print("[bold green]Using default ASR model: openai/whisper-small[/bold green]\n")
                return "openai/whisper-small"
            elif choice in AVAILABLE_ASR_MODELS:
                selected_model = AVAILABLE_ASR_MODELS[choice]["name"]
                console.print(f"[bold green]Selected ASR model: {selected_model}[/bold green]\n")
                return selected_model
            else:
                console.print(f"[bold red]Invalid choice. Please enter a valid number.[/bold red]")
        except KeyboardInterrupt:
            console.print("\n[bold red]Using default model: openai/whisper-small[/bold red]\n")
            return "openai/whisper-small"
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")
            console.print("[bold red]Using default model: openai/whisper-small[/bold red]\n")
            return "openai/whisper-small"

# --- Core Application Classes ---

class AudioRecorder:
    def __init__(self, sample_rate=16000, channels=1, chunk_size=512):
        logger.info(f"Initializing AudioRecorder: sample_rate={sample_rate}, channels={channels}, chunk_size={chunk_size}")
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue(maxsize=50)
        self.is_recording = False
        self.stream = None
        self.p = pyaudio.PyAudio()
        self.chunks_recorded = 0
        self.frames = []
        logger.info("AudioRecorder initialized successfully")

    def start_continuous_recording(self):
        logger.info("Starting continuous recording")
        self.is_recording = True
        self.chunks_recorded = 0
        self.frames = []
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
        try:
            self.audio_queue.put(None, timeout=1)
            logger.info("Sentinel value added to queue")
        except queue.Full:
            logger.warning("Queue was full when adding sentinel value")
        logger.info(f"Recording stopped. Total chunks recorded: {self.chunks_recorded}")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            self.chunks_recorded += 1
            self.frames.append(in_data)
            try:
                self.audio_queue.put_nowait(in_data)
            except queue.Full:
                logger.debug("Audio queue full, skipping chunk")
        return (None, pyaudio.paContinue)

    def save_to_file(self, filename):
        logger.info(f"Saving audio to {filename}")
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        logger.info(f"Audio saved successfully to {filename}")
        return filename

class VADProcessor:
    def __init__(self):
        if not load_silero_vad:
            raise ImportError("Silero VAD not installed.")
        logger.info("Initializing VAD Processor")
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
            audio_tensor = torch.from_numpy(audio_chunk_float)
            speech_prob = self.model(audio_tensor, sample_rate).item()
            return speech_prob > 0.1
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return False

    def has_speech(self, audio_array, sample_rate=16000):
        if len(audio_array) / sample_rate < 0.15:
            return False
        if not get_speech_timestamps:
            return True # Fallback if VAD function is missing
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
            return False

class ASRProcessor:
    def __init__(self, model_name="openai/whisper-small"):
        logger.info(f"Initializing ASR Processor with model: {model_name}")
        self.model_name = model_name
        console.print(f"[info]Loading ASR model: {model_name}...[/info]")
        try:
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            console.print(f"[bold green]✓ ASR model loaded successfully on {self.device}[/bold green]")
            logger.info(f"ASR model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load ASR model: {e}")
            console.print(f"[bold red]✗ Failed to load ASR model: {e}[/bold red]")
            raise

    def transcribe(self, audio_array, sample_rate=16000):
        if len(audio_array) == 0:
            return ""
        start_time = time.time()
        try:
            inputs = self.processor(audio_array, sampling_rate=sample_rate, return_tensors="pt").to(self.device)
            with torch.no_grad():
                predicted_ids = self.model.generate(inputs.input_features)
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True, normalize=True)
            result = transcription[0].strip() if transcription else ""
            logger.info(f"ASR transcription completed in {time.time() - start_time:.3f}s: '{result}'")
            console.print(Panel(result, title="User", border_style="cyan"))
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

        # Construct model string for litellm
        model_string = f"{provider}/{model}"

        # Prepare arguments for litellm
        kwargs = {
            "model": model_string,
            "messages": messages,
            "stream": False
        }

        if provider == "ollama":
            kwargs["api_base"] = self.api_base
        elif provider == "gemini":
            if 'GEMINI_API_KEY' not in os.environ or not os.environ['GEMINI_API_KEY']:
                error_msg = "GEMINI_API_KEY environment variable not set."
                logger.error(error_msg)
                console.print(f"[bold red]{error_msg}[/bold red]")
                return "Sorry, the Gemini API key is not configured."

        try:
            response = litellm.completion(**kwargs)
            content = response["choices"][0]["message"]["content"]
            logger.info(f"LiteLLM {prompt_type} completed in {time.time() - start_time:.3f}s")
            if prompt_type == "chat":
                console.print(Panel(content, title="Serin", border_style="magenta"))
            return content
        except Exception as e:
            logger.error(f"LiteLLM error: {e}")
            return f"Sorry, I'm having trouble connecting to the AI model for {prompt_type}."

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

    def synthesize(self, text, voice):
        if not text:
            return None, None
        start_time = time.time()
        try:
            generator = self.pipeline(text, voice=voice)
            audio_chunks = [audio for _, _, audio in generator]
            if audio_chunks:
                full_audio = np.concatenate(audio_chunks)
                logger.info(f"TTS synthesis completed in {time.time() - start_time:.3f}s")
                return full_audio, 24000
            return None, None
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None, None

# --- Model and Utility Loading ---
def load_models():
    try:
        asr_model_name = select_asr_model()
        asr = ASRProcessor(model_name=asr_model_name)
        
        llm_model_info = select_llm_model()
        
        tts = TTSProcessor()
        vad = VADProcessor()
        
        return asr, llm_model_info, tts, vad
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        console.print(f"[bold red]A critical error occurred during model loading: {e}[/bold red]")
        console.print("[bold red]Please ensure all dependencies are installed and models are accessible.[/bold red]")
        sys.exit(1)


def read_resume(file_path=config['paths']['resume_file']):
    """Reads content from a PDF resume file."""
    try:
        with open(file_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            content = ""
            for page in pdf_reader.pages:
                content += page.extract_text()
            logger.info(f"Successfully read resume from {file_path}")
            return content
    except FileNotFoundError:
        logger.error(f"Resume file not found at {file_path}")
        console.print(f"[bold red]Error: Resume file not found at {file_path}[/bold red]")
        return None
    except Exception as e:
        logger.error(f"Error reading resume: {e}")
        console.print(f"[bold red]An error occurred while reading the resume: {e}[/bold red]")
        return None

def play_audio(audio_data, sample_rate):
    if sd is not None and audio_data is not None:
        try:
            sd.play(audio_data, sample_rate)
            sd.wait()
        except Exception as e:
            logger.error(f"Error playing audio: {e}")

def get_conversational_response(ollama, question, conversation_history, model_info):
    logger.info("Getting conversational response")
    conversational_prompt = f"You are Serin, an interviewer. Your goal is to ask the following question in a natural and conversational way: '{question}'. Do not add any other information to the response. Just ask the question."
    messages = conversation_history + [{"role": "system", "content": conversational_prompt}]
    return ollama.chat(messages, model_info, prompt_type="chat")

def generate_next_question(ollama, conversation_history, model_info, interview_topic, resume_content):
    logger.info("Generating next question")
    qgen_prompt = f"""You are a question generation module. Your task is to generate the next interview question based on the conversation history, the interview topic, and the candidate's resume. The question should be a logical follow-up to the previous turn in the conversation. Do not ask more than one question. Do not ask for clarification. Just generate the question.

Here is the candidate's resume:
--- START RESUME ---
{resume_content}
--- END RESUME ---

The topic for today's interview is: {interview_topic}

Based on the conversation history, generate the next question."""
    messages = conversation_history + [{"role": "system", "content": qgen_prompt}]
    return ollama.chat(messages, model_info, prompt_type="qgen")

# --- Core Processing Logic ---
def process_audio_continuously(asr, llm_model_info, tts, vad, recorder, silence_threshold, tts_voice, conversation_history, interview_topic, resume_content):
    logger.info("Starting continuous audio processing thread")
    current_utterance = []
    silence_chunks = 0
    chunks_per_second = recorder.sample_rate / recorder.chunk_size
    silent_chunks_threshold = int(silence_threshold * chunks_per_second)
    
    ollama = LiteLLMClient()
    
    while recorder.is_recording:
        try:
            audio_data_bytes = recorder.audio_queue.get(timeout=0.1)
            if audio_data_bytes is None:
                break
                
            audio_chunk_np = np.frombuffer(audio_data_bytes, dtype=np.int16)
            audio_chunk_float = audio_chunk_np.astype(np.float32) / 32768.0
            
            if vad.detect_speech_in_chunk(audio_chunk_float):
                current_utterance.append(audio_chunk_np)
                silence_chunks = 0
            else:
                if len(current_utterance) > 0:
                    silence_chunks += 1
                    if silence_chunks > silent_chunks_threshold:
                        full_utterance_np = np.concatenate(current_utterance, axis=0)
                        full_utterance_float = full_utterance_np.astype(np.float32) / 32768.0
                        current_utterance = []
                        silence_chunks = 0
                        
                        if vad.has_speech(full_utterance_float):
                            logger.info("VAD confirmed speech, processing utterance")
                            conversation_history = process_utterance(asr, ollama, tts, full_utterance_float, llm_model_info, tts_voice, conversation_history, interview_topic, resume_content)
                        else:
                            logger.info("VAD rejected utterance as non-speech")
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Error in continuous processing: {e}")
            continue

def process_utterance(asr, ollama, tts, audio_data, llm_model_info, tts_voice, conversation_history, interview_topic, resume_content):
    logger.info("Starting utterance processing")
    try:
        transcription = asr.transcribe(audio_data)
        
        if transcription:
            conversation_history.append({"role": "user", "content": transcription})
            
            # Generate the next question
            next_question = generate_next_question(ollama, conversation_history, llm_model_info, interview_topic, resume_content)
            
            if next_question:
                # Get the conversational response
                conversational_response = get_conversational_response(ollama, next_question, conversation_history, llm_model_info)
                
                if conversational_response:
                    conversation_history.append({"role": "assistant", "content": conversational_response})
                    
                    audio_output, sample_rate = tts.synthesize(conversational_response, voice=tts_voice)
                    if audio_output is not None:
                        play_audio(audio_output, sample_rate)
        else:
            logger.warning("ASR returned empty transcription")
    except Exception as e:
        logger.error(f"Error in processing utterance: {e}")
    return conversation_history

def save_session_to_json(asr_processor, llm_model_info, interview_topic, conversation_history, camera_data, audio_file, video_file):
    """Saves the session details to a timestamped JSON file."""
    session_dir = config['paths']['sessions_dir']
    os.makedirs(session_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(session_dir, f"session.json")
    session_data = {
        "asr_model": asr_processor.model_name,
        "litellm_model": llm_model_info,
        "interview_topic": interview_topic,
        "conversation_history": conversation_history,
        "camera_capture": camera_data,
        "audio_file": audio_file,
        "video_file": video_file
    }
    
    try:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
        else:
            all_data = []

        all_data.append({"time": timestamp, "session_data": session_data})

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
            console.print(f"[info]Session saved to {filename}[/info]")
            logger.info(f"Session data saved to {filename}")
    except Exception as e:
        console.print(f"[bold red]Error saving session: {e}[/bold red]")
        logger.error(f"Failed to save session data to {filename}: {e}")


class CameraRecorder:
    def __init__(self):
        logger.info("Initializing CameraRecorder")
        self.is_recording = False
        self.capture_data = {}
        self.model = RFDETRNano()
        self.cap = None
        self.thread = None
        self.out = None
        self.video_filename = None

    def start_recording(self, filename):
        logger.info("Starting camera recording")
        self.is_recording = True
        self.video_filename = filename
        self.cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(self.video_filename, fourcc, 20.0, (640, 480))
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()

    def stop_recording(self):
        logger.info("Stopping camera recording")
        self.is_recording = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()
        return self.video_filename

    def _record_loop(self):
        while self.is_recording:
            success, frame = self.cap.read()
            if not success:
                break

            self.out.write(frame)  # Save the frame to the video file

            rgb_frame = frame[:, :, ::-1].copy()
            detections = self.model.predict(rgb_frame, threshold=0.5)

            labels = [
                f"{COCO_CLASSES[class_id]} {confidence:.2f}"
                for class_id, confidence
                in zip(detections.class_id, detections.confidence)
            ]
            count_of_person = 0
            count_of_devices = 0
            for label in labels:
                if 'person' in label:
                    count_of_person += 1
                if 'laptop' in label or 'phone' in label or 'remote' in label or 'tv' in label:
                    count_of_devices += 1
            # check if more than one person is detected or devices such as phone, remote, laptop is detected
            if count_of_devices > 0 or count_of_person > 1:
                current_time = datetime.now()
                self.capture_data.update({str(current_time): labels})

            # Optional: Display the annotated frame
            # annotated_frame = frame.copy()
            # annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
            # annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)
            # cv2.imshow("Webcam", annotated_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    def get_capture_data(self):
        return self.capture_data

# --- CLI Main Loop ---
def main():
    console.print("\n[bold green]==== Conversational AI CLI ====[/bold green]", style="info")
    console.print("[info]Welcome! You'll be prompted to select ASR and LLM models for this session.[/info]")
    
    # --- Initial Setup ---
    asr, llm_model_info, tts, vad = load_models()
    recorder = AudioRecorder()
    camera_recorder = CameraRecorder()
    silence_threshold = config['audio']['silence_threshold_seconds']
    tts_voice = config['models']['tts_voice']
    ollama = LiteLLMClient()

    resume_content = read_resume()
    if not resume_content:
        return

    console.print("\n[prompt]Please enter the topic for this interview session (e.g., Machine Learning, Android Development):[/prompt]")
    interview_topic = input("> ").strip()

    initial_prompt = f"""You are Serin, an interviewer. Your goal is to understand the user's background and skills. Based on what the user says, ask a relevant follow-up question to gather more details. For example, if the user says 'I made a machine learning model,' you should ask something like, 'Great! Which type of machine learning model did you make?' Do not ask more than one question. The one question should be a followup to what the user said. Do not assume anything else that what is given to you. If you do not understand the user's response, ask for clarification. Do not ask for clarification if you understand the user's response. If you think that resume does not have adequate information for the given topic, you can ask a general question around the topic. Do not mention about reading the resume in the conversation. First, introduce yourself as Serin. Then, ask the question.

Here is the candidate's resume:
--- START RESUME ---
{resume_content}
--- END RESUME ---

The topic for today's interview is: {interview_topic}

Based on the resume, ask one insightful question related to the topic to start the conversation."""
    conversation_history = [{"role": "system", "content": initial_prompt}, {"role": "user", "content": "Hello!"}]
    
    processing_thread = None # Initialize thread variable

    try:
        # --- Start Session ---
        console.print("\n[prompt]Press Enter to start the interview session.[/prompt]")
        input("")

        # Start recordings for the whole session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sessions_dir = config['paths']['sessions_dir']
        os.makedirs(sessions_dir, exist_ok=True)
        audio_filename = os.path.join(sessions_dir, f"session_{timestamp}.wav")
        video_filename = os.path.join(sessions_dir, f"session_{timestamp}.avi")
        
        recorder.start_continuous_recording()
        camera_recorder.start_recording(video_filename)

        # --- Opening Question ---
        console.print("\n[info]Generating opening question...[/info]")
        opening_question = ollama.chat(conversation_history, model_info=llm_model_info)
        if opening_question:
            conversation_history.append({"role": "assistant", "content": opening_question})
            audio_output, sample_rate = tts.synthesize(opening_question, voice=tts_voice)
            if audio_output is not None:
                play_audio(audio_output, sample_rate)

        # --- Main Conversation Loop (background thread) ---
        processing_thread = threading.Thread(
            target=process_audio_continuously,
            args=(asr, llm_model_info, tts, vad, recorder, silence_threshold, tts_voice, conversation_history, interview_topic, resume_content),
            daemon=True
        )
        processing_thread.start()

        console.print(Panel("[prompt]The interview has started. Speak freely. Press Ctrl+C to end the session.[/prompt]", style="bold green", border_style="green"))
        
        # Keep the main thread alive to listen for Ctrl+C
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        console.print("\n[bold red]Interview ended by user.[/bold red]")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main loop: {e}")
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
    finally:
        # --- Save Session ---
        console.print("\n[info]Stopping recordings and saving session data...[/info]")
        
        if recorder.is_recording:
            recorder.stop_recording()
        
        if processing_thread and processing_thread.is_alive():
            processing_thread.join()

        video_file = camera_recorder.stop_recording()
        audio_file = recorder.save_to_file(audio_filename)
        
        if len(conversation_history) > 2:
            camera_data = camera_recorder.get_capture_data()
            save_session_to_json(asr, llm_model_info, interview_topic, conversation_history, camera_data, audio_file, video_file)
        else:
            console.print("[info]No conversation to save.[/info]")


if __name__ == "__main__":
    main()
