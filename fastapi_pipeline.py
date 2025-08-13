import asyncio
import base64
import json
import logging
import numpy as np
import os
import queue
import threading
import time
import uuid
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

# Video processing imports
try:
    import cv2
    import supervision as sv
    from rfdetr import RFDETRNano
    from rfdetr.util.coco_classes import COCO_CLASSES
    VIDEO_PROCESSING_AVAILABLE = True
    logging.info("Video processing libraries loaded successfully")
except ImportError as e:
    VIDEO_PROCESSING_AVAILABLE = False
    logging.warning(f"Video processing libraries not available: {e}")

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
        print("Gemini configuration validated successfully")
    else:
        print("Gemini configuration issues detected")
except ImportError:
    print("Could not import config module")
    # Fallback to direct environment variable
    if 'GEMINI_API_KEY' not in os.environ:
        os.environ['GEMINI_API_KEY'] = "AIzaSyD-987BdBsdKnCa7oWZktY9_1K27hS-qY8"
        print("Using default Gemini API key. For production, set GEMINI_API_KEY environment variable.")

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
    """Read resume content from either local file or Supabase storage."""
    try:
        # First, try to treat it as a local file
        if os.path.exists(resume_file):
            with open(resume_file, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
            return text
        
        # If not a local file, try to download from Supabase
        if SUPABASE_AVAILABLE and supabase_config.client:
            # Check if it looks like a user ID (UUID format)
            import re
            uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
            
            if uuid_pattern.match(resume_file):
                # It's a user ID, try to get resume from user profile
                logger.info(f"Treating {resume_file} as user ID, fetching from profile")
                return load_resume_from_user_profile(resume_file)
            else:
                # It might be a direct path to resume in Supabase storage
                logger.info(f"Treating {resume_file} as direct path to resume in Supabase")
                return load_resume_from_supabase_path(resume_file)
        
        # Fallback: return error message
        logger.warning(f"Resume file not found locally and Supabase not available: {resume_file}")
        return "Resume content not available"
        
    except Exception as e:
        logger.error(f"Failed to read resume content from {resume_file}: {e}")
        return "Resume content not available"

def load_resume_from_user_profile(user_id: str) -> str:
    """Load resume content from user profile in Supabase."""
    try:
        # Get user profile from database
        response = supabase_config.client.table('user_profiles').select('*').eq('user_id', user_id).execute()
        
        if response.data and len(response.data) > 0:
            user_profile = response.data[0]
            resume_url = user_profile.get('resume_url')
            
            if resume_url:
                # Download resume from Supabase storage
                resume_data = supabase_config.download_resume_by_path(resume_url)
                if resume_data:
                    # Read PDF content
                    import io
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(resume_data))
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    logger.info(f"Successfully loaded resume content for user {user_id}")
                    return text
                else:
                    logger.warning(f"Resume file not found in Supabase: {resume_url}")
                    return "Resume file not found in storage"
            else:
                logger.warning(f"No resume_url found for user {user_id}")
                return "No resume found for this user"
        else:
            logger.warning(f"User profile not found for user {user_id}")
            return "User profile not found"
            
    except Exception as e:
        logger.error(f"Failed to load resume from user profile for user {user_id}: {e}")
        return "Resume content not available (profile error)"

def load_resume_from_supabase_path(resume_path: str) -> str:
    """Load resume content directly from Supabase storage path."""
    try:
        # Extract the actual file path from the URL if it's a full URL
        import re
        if resume_path.startswith('http'):
            # Extract path from URL like: https://.../storage/v1/object/public/resumes/user_id/filename.pdf
            # We want: user_id/filename.pdf
            match = re.search(r'/resumes/(.+)$', resume_path)
            if match:
                resume_path = match.group(1)
                logger.info(f"Extracted path from URL: {resume_path}")
            else:
                logger.error(f"Could not extract path from URL: {resume_path}")
                return "Resume content not available (invalid URL format)"
        
        # Download resume from Supabase storage
        resume_data = supabase_config.download_resume_by_path(resume_path)
        if resume_data:
            # Read PDF content
            import io
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(resume_data))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            logger.info(f"Successfully loaded resume content from path: {resume_path}")
            return text
        else:
            logger.warning(f"Resume file not found in Supabase: {resume_path}")
            return "Resume file not found in storage"
            
    except Exception as e:
        logger.error(f"Failed to load resume from Supabase path {resume_path}: {e}")
        return "Resume content not available (storage error)"

# Pydantic models
class ModelSelectionRequest(BaseModel):
    job_role: str  # From job template
    user_id: str   # User ID for resume
    resume_url: str  # Direct path to resume in Supabase storage
    asr_model: str = "openai/whisper-medium"  # Updated default
    llm_provider: str = "gemini"  # Changed to gemini
    llm_model: str = "gemini-2.5-flash"  # Changed to gemini-2.5-flash

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


class VideoAnnotationProcessor:
    def __init__(self):
        if not VIDEO_PROCESSING_AVAILABLE:
            raise ImportError("Video processing libraries not available.")
        logger.info("Initializing Video Annotation Processor")
        try:
            self.model = RFDETRNano()
            logger.info("Video annotation model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize video annotation model: {e}")
            raise

    def annotate_video(self, input_video_path: str, output_video_path: str) -> Dict:
        """Annotate video with object detection and return detection statistics."""
        logger.info(f"Starting video annotation: {input_video_path} -> {output_video_path}")
        
        try:
            # Check if input file exists
            if not os.path.exists(input_video_path):
                raise FileNotFoundError(f"Input video file not found: {input_video_path}")
            
            logger.info(f"Input video exists: {input_video_path}")
            
            # Get detailed video properties using ffprobe for accurate timing
            import subprocess
            import json
            
            # Use ffprobe to get accurate video properties
            ffprobe_cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                input_video_path
            ]
            
            try:
                result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    video_info = json.loads(result.stdout)
                    
                    # Extract video stream info
                    video_stream = None
                    audio_stream = None
                    for stream in video_info.get('streams', []):
                        if stream.get('codec_type') == 'video':
                            video_stream = stream
                        elif stream.get('codec_type') == 'audio':
                            audio_stream = stream
                    
                    if video_stream:
                        # Get accurate frame rate from video stream
                        fps_str = video_stream.get('r_frame_rate', '0/1')
                        if '/' in fps_str:
                            num, den = map(int, fps_str.split('/'))
                            fps = num / den if den > 0 else 30.0
                        else:
                            fps = float(fps_str)
                        
                        width = int(video_stream.get('width', 0))
                        height = int(video_stream.get('height', 0))
                        duration = float(video_info.get('format', {}).get('duration', 0))
                        
                        logger.info(f"FFprobe video properties: {width}x{height}, {fps:.3f} FPS, duration: {duration:.2f}s")
                    else:
                        raise ValueError("No video stream found in input file")
                        
                else:
                    logger.warning(f"FFprobe failed, falling back to OpenCV: {result.stderr}")
                    raise RuntimeError("FFprobe failed")
                    
            except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
                logger.warning(f"FFprobe error, falling back to OpenCV: {e}")
                # Fallback to OpenCV
                cap = cv2.VideoCapture(input_video_path)
                if not cap.isOpened():
                    raise ValueError(f"Could not open input video: {input_video_path}")
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps if fps > 0 else 0
                cap.release()
                
                logger.info(f"OpenCV video properties: {width}x{height}, {fps:.3f} FPS, {total_frames} frames, duration: {duration:.2f}s")
            
            # Open input video for processing
            cap = cv2.VideoCapture(input_video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open input video: {input_video_path}")
            
            # Get total frames for progress tracking
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create temporary video file for annotated frames (without audio)
            temp_video_path = output_video_path.replace('.webm', '_temp.mp4')
            
            # Use the original frame rate for the temporary video to preserve timing
            original_fps = fps
            
            # Setup video writer for temporary file with original frame rate
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, original_fps, (width, height))
            
            if not out.isOpened():
                raise ValueError(f"Could not create temporary video writer: {temp_video_path}")
            
            logger.info(f"Temporary video writer created with {original_fps:.3f} FPS: {temp_video_path}")
            
            # Detection statistics
            detection_stats = {
                "total_frames": total_frames,
                "processed_frames": 0,
                "detection_frames": 0,
                "person_detections": 0,
                "device_detections": 0,
                "capture_data": {},
                "original_fps": original_fps,
                "duration": duration
            }
            
            frame_count = 0
            
            while True:
                success, frame = cap.read()
                if not success:
                    break
                
                frame_count += 1
                detection_stats["processed_frames"] += 1
                
                # Convert BGR to RGB for model input
                rgb_frame = frame[:, :, ::-1].copy()
                
                # Run object detection
                detections = self.model.predict(rgb_frame, threshold=0.5)
                
                # Create labels
                labels = [
                    f"{COCO_CLASSES[class_id]} {confidence:.2f}"
                    for class_id, confidence
                    in zip(detections.class_id, detections.confidence)
                ]
                
                # Count detections
                count_of_person = 0
                count_of_devices = 0
                
                for label in labels:
                    if 'person' in label:
                        count_of_person += 1
                    if 'laptop' in label or 'phone' in label or 'remote' in label or 'tv' in label:
                        count_of_devices += 1
                
                # Update statistics
                if count_of_person > 0 or count_of_devices > 0:
                    detection_stats["detection_frames"] += 1
                    detection_stats["person_detections"] += count_of_person
                    detection_stats["device_detections"] += count_of_devices
                    
                    # Record detection data for frames with multiple people or devices
                    if count_of_person > 1 or count_of_devices > 0:
                        current_time = datetime.now()
                        detection_stats["capture_data"][str(current_time)] = labels
                
                # Annotate frame
                annotated_frame = frame.copy()
                annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
                annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)
                
                # Write annotated frame
                out.write(annotated_frame)
                
                # Log progress every 100 frames
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count}/{total_frames} frames")
            
            # Cleanup video capture and writer
            cap.release()
            out.release()
            
            logger.info(f"Temporary annotated video created: {temp_video_path}")
            
            # Verify temporary video properties
            temp_cap = cv2.VideoCapture(temp_video_path)
            temp_fps = temp_cap.get(cv2.CAP_PROP_FPS)
            temp_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            temp_cap.release()
            logger.info(f"Temporary video properties: {temp_fps:.3f} FPS, {temp_frames} frames, duration: {temp_frames / temp_fps:.2f}s")
            
            if abs(temp_frames - total_frames) > 1:
                logger.warning(f"Frame count mismatch: original={total_frames}, annotated={temp_frames}")
            if abs(temp_fps - original_fps) > 0.1:
                logger.warning(f"FPS mismatch: original={original_fps:.3f}, annotated={temp_fps:.3f}")
            
            # Use FFmpeg to combine annotated video with original audio and convert to WebM
            # Key changes for better synchronization:
            # 1. Use -vsync 1 (cfr) for constant frame rate
            # 2. Don't force frame rate - preserve original timing
            # 3. Use -async 1 for audio synchronization
            # 4. Use -copyts to preserve timestamps
            try:
                # First try VP9 encoding (better quality but slower)
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-i', temp_video_path,  # Annotated video (no audio)
                    '-i', input_video_path,  # Original video (with audio)
                    '-c:v', 'libvpx-vp9',  # Transcode video to VP9 for WebM compatibility
                    '-crf', '30',  # Quality setting for VP9 (lower = better quality)
                    '-b:v', '0',  # Use CRF mode for VP9
                    '-vsync', '1',  # Use constant frame rate (cfr)
                    '-async', '1',  # Audio synchronization
                    '-copyts',  # Preserve timestamps
                    '-c:a', 'copy',  # Copy audio stream from original video
                    '-map', '0:v:0',  # Use video from first input (annotated)
                    '-map', '1:a:0',  # Use audio from second input (original)
                    '-y',  # Overwrite output file
                    output_video_path
                ]
                
                logger.info(f"Running FFmpeg command (VP9): {' '.join(ffmpeg_cmd)}")
                
                # Run FFmpeg command
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    logger.info(f"FFmpeg successfully created annotated video with audio (VP9): {output_video_path}")
                    
                    # Clean up temporary file
                    if os.path.exists(temp_video_path):
                        os.remove(temp_video_path)
                        logger.info(f"Temporary file cleaned up: {temp_video_path}")
                else:
                    logger.warning(f"VP9 encoding failed: {result.stderr}")
                    logger.info("Trying VP8 encoding as fallback...")
                    
                    # Fallback to VP8 encoding (faster but lower quality)
                    ffmpeg_cmd_vp8 = [
                        'ffmpeg',
                        '-i', temp_video_path,  # Annotated video (no audio)
                        '-i', input_video_path,  # Original video (with audio)
                        '-c:v', 'libvpx',  # Transcode video to VP8 for WebM compatibility
                        '-crf', '10',  # Quality setting for VP8
                        '-b:v', '1M',  # Bitrate for VP8
                        '-vsync', '1',  # Use constant frame rate (cfr)
                        '-async', '1',  # Audio synchronization
                        '-copyts',  # Preserve timestamps
                        '-c:a', 'copy',  # Copy audio stream from original video
                        '-map', '0:v:0',  # Use video from first input (annotated)
                        '-map', '1:a:0',  # Use audio from second input (original)
                        '-y',  # Overwrite output file
                        output_video_path
                    ]
                    
                    logger.info(f"Running FFmpeg command (VP8): {' '.join(ffmpeg_cmd_vp8)}")
                    
                    # Run FFmpeg command with VP8
                    result_vp8 = subprocess.run(ffmpeg_cmd_vp8, capture_output=True, text=True, timeout=300)
                    
                    if result_vp8.returncode == 0:
                        logger.info(f"FFmpeg successfully created annotated video with audio (VP8): {output_video_path}")
                        
                        # Clean up temporary file
                        if os.path.exists(temp_video_path):
                            os.remove(temp_video_path)
                            logger.info(f"Temporary file cleaned up: {temp_video_path}")
                    else:
                        logger.error(f"VP8 encoding also failed: {result_vp8.stderr}")
                        # Fall back to temporary file if both VP9 and VP8 fail
                        if os.path.exists(temp_video_path):
                            import shutil
                            shutil.move(temp_video_path, output_video_path)
                            logger.warning(f"Fell back to temporary file (no audio): {output_video_path}")
                        else:
                            raise RuntimeError("Both VP9 and VP8 encoding failed and temporary file not found")
                        
            except (subprocess.TimeoutExpired, FileNotFoundError, RuntimeError) as e:
                logger.error(f"FFmpeg error: {e}")
                # Fall back to temporary file if FFmpeg is not available or fails
                if os.path.exists(temp_video_path):
                    import shutil
                    shutil.move(temp_video_path, output_video_path)
                    logger.warning(f"Fell back to temporary file (no audio): {output_video_path}")
                else:
                    raise RuntimeError("FFmpeg failed and temporary file not found")
            
            # Verify final output video properties
            try:
                final_cap = cv2.VideoCapture(output_video_path)
                final_fps = final_cap.get(cv2.CAP_PROP_FPS)
                final_frames = int(final_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                final_cap.release()
                
                logger.info(f"Final video properties: {final_fps:.3f} FPS, {final_frames} frames, duration: {final_frames / final_fps:.2f}s")
                
                # Check for timing consistency
                if abs(final_frames - total_frames) > 1:
                    logger.warning(f"Final frame count mismatch: original={total_frames}, final={final_frames}")
                if abs(final_fps - original_fps) > 0.1:
                    logger.warning(f"Final FPS mismatch: original={original_fps:.3f}, final={final_fps:.3f}")
                    
            except Exception as e:
                logger.warning(f"Could not verify final video properties: {e}")
            
            logger.info(f"Video annotation completed: {detection_stats['processed_frames']} frames processed")
            logger.info(f"Detection summary: {detection_stats['person_detections']} persons, {detection_stats['device_detections']} devices")
            
            return detection_stats
            
        except Exception as e:
            logger.error(f"Video annotation error: {e}")
            raise

    def save_detection_data(self, session_id: str, detection_stats: Dict, output_path: str):
        """Save detection statistics to JSON file."""
        try:
            data = {
                "session_id": session_id,
                "capture_data": detection_stats["capture_data"],
                "statistics": {
                    "total_frames": detection_stats["total_frames"],
                    "processed_frames": detection_stats["processed_frames"],
                    "detection_frames": detection_stats["detection_frames"],
                    "person_detections": detection_stats["person_detections"],
                    "device_detections": detection_stats["device_detections"]
                },
                "timestamp": datetime.now().isoformat()
            }
            
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Detection data saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save detection data: {e}")
            raise


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
        
        # Create sessions directory for detection data
        self.sessions_dir = Path("sessions")
        self.sessions_dir.mkdir(exist_ok=True)
        logger.info(f"Sessions directory: {self.sessions_dir.absolute()}")
        
        # Initialize video annotation processor if available
        self.video_processor = None
        if VIDEO_PROCESSING_AVAILABLE:
            try:
                self.video_processor = VideoAnnotationProcessor()
                logger.info("Video annotation processor initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize video annotation processor: {e}")
        else:
            logger.warning("Video processing libraries not available, video annotation will be skipped")
        
        # Log Supabase availability
        if SUPABASE_AVAILABLE and supabase_config.client:
            logger.info("Supabase storage is available for recordings")
        else:
            logger.warning("Supabase storage is not available, using local storage only")
        
        # Session timeout settings (30 minutes)
        self.session_timeout_seconds = 30 * 60  # 30 minutes
        self.last_activity: Dict[str, datetime] = {}
        
        # Start session cleanup thread
        self.cleanup_thread = threading.Thread(target=self._session_cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        logger.info("Session cleanup thread started")
    
    def _session_cleanup_worker(self):
        """Background thread to clean up inactive sessions."""
        while True:
            try:
                current_time = datetime.now()
                sessions_to_cleanup = []
                
                for session_id, last_activity in self.last_activity.items():
                    time_diff = (current_time - last_activity).total_seconds()
                    if time_diff > self.session_timeout_seconds:
                        sessions_to_cleanup.append(session_id)
                
                for session_id in sessions_to_cleanup:
                    logger.info(f"Session {session_id} timed out, auto-saving and cleaning up")
                    try:
                        if session_id in self.sessions:
                            session_data = self.sessions[session_id]
                            self.save_session_to_database(session_id, session_data)
                            logger.info(f"Session {session_id} auto-saved due to timeout")
                        
                        # Clean up session
                        self.delete_session(session_id)
                        logger.info(f"Session {session_id} cleaned up due to timeout")
                        
                    except Exception as e:
                        logger.error(f"Failed to cleanup timed out session {session_id}: {e}")
                
                # Sleep for 5 minutes before next cleanup check
                time.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in session cleanup worker: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def update_session_activity(self, session_id: str):
        """Update the last activity time for a session."""
        self.last_activity[session_id] = datetime.now()
    
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
            
            # Load resume content using the resume_url
            logger.info(f"Loading resume content for session {session_id}")
            session_data["resume_content"] = load_resume_from_supabase_path(model_request.resume_url)
            
            # Create initial conversation prompt
            logger.info(f"Creating initial prompt for session {session_id}")
            initial_prompt = self.create_initial_prompt(
                model_request.job_role,
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
                    model_request.job_role,
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
            # Initialize activity tracking
            self.update_session_activity(session_id)
            logger.info(f"Session {session_id} created successfully")
            return session_data
            
        except Exception as e:
            logger.error(f"Failed to create session {session_id}: {e}")
            # Clean up any partially created session
            if session_id in self.sessions:
                del self.sessions[session_id]
            if session_id in self.last_activity:
                del self.last_activity[session_id]
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
            
            # Save session information to database before cleanup
            try:
                self.save_session_to_database(session_id, session)
            except Exception as e:
                logger.error(f"Failed to save session data for {session_id}: {e}")
            
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
            # Clean up activity tracking
            if session_id in self.last_activity:
                del self.last_activity[session_id]
            logger.info(f"Session {session_id} deleted")

    def save_session_to_database(self, session_id: str, session_data: dict):
        """Save session information to the interview_sessions table."""
        if not SUPABASE_AVAILABLE or not supabase_config.client:
            logger.warning("Supabase not available, skipping session save")
            return
        
        try:
            # Extract conversation history (excluding system prompt and initial "Hello")
            conversation_history = []
            for message in session_data.get("conversation_history", []):
                if message.get("role") == "system":
                    continue  # Skip system prompt
                if message.get("role") == "user" and message.get("content") == "Hello":
                    continue  # Skip initial "Hello"
                conversation_history.append(message)
            
            # Get recording URL if available
            recording_url = None
            recording_info = self.get_recording_info(session_id)
            if recording_info and recording_info.public_url:
                recording_url = recording_info.public_url
                logger.info(f"Found recording URL for session {session_id}: {recording_url}")
            else:
                logger.warning(f"No recording URL found for session {session_id}")
            
            # Get detection data for anomalies
            detection_data = self.get_detection_data(session_id)
            anomalies_detected = []
            if detection_data and detection_data.get("capture_data"):
                # Analyze detection data for anomalies
                capture_data = detection_data["capture_data"]
                logger.info(f"Analyzing detection data for session {session_id}: {len(capture_data)} frames with detections")
                
                for timestamp, detections in capture_data.items():
                    # Count persons and devices more accurately
                    person_count = 0
                    device_count = 0
                    
                    for detection in detections:
                        detection_lower = detection.lower()
                        if "person" in detection_lower:
                            person_count += 1
                        elif any(device in detection_lower for device in ["laptop", "phone", "cell phone", "tv", "remote", "computer", "monitor", "keyboard", "mouse"]):
                            device_count += 1
                    
                    logger.debug(f"Frame {timestamp}: {person_count} persons, {device_count} devices - {detections}")
                    
                    # Flag anomalies: multiple people or devices detected
                    if person_count > 1:
                        logger.info(f"Anomaly detected: Multiple persons ({person_count}) at {timestamp}")
                        anomalies_detected.append({
                            "timestamp": timestamp,
                            "type": "multiple_persons",
                            "count": person_count,
                            "detections": detections,
                            "description": f"Multiple persons detected in frame: {person_count} people"
                        })
                    if device_count > 0:
                        logger.info(f"Anomaly detected: Devices ({device_count}) at {timestamp}")
                        anomalies_detected.append({
                            "timestamp": timestamp,
                            "type": "devices_detected",
                            "count": device_count,
                            "detections": detections,
                            "description": f"Digital devices detected in frame: {device_count} devices"
                        })
                
                logger.info(f"Total anomalies detected for session {session_id}: {len(anomalies_detected)}")
                
                # Log summary of anomalies
                if anomalies_detected:
                    multiple_persons_count = sum(1 for anomaly in anomalies_detected if anomaly["type"] == "multiple_persons")
                    devices_count = sum(1 for anomaly in anomalies_detected if anomaly["type"] == "devices_detected")
                    logger.info(f"Anomaly summary: {multiple_persons_count} frames with multiple persons, {devices_count} frames with devices")
            else:
                logger.info(f"No detection data found for session {session_id}")
            
            # Prepare session information JSON (without resume_url and recording_url)
            session_information = {
                "job_role": session_data.get("model_request").job_role if session_data.get("model_request") else None,
                "conversation_history": conversation_history,
                "anomalies_detected": anomalies_detected,
                "asr_model": session_data.get("model_request").asr_model if session_data.get("model_request") else None,
                "llm_provider": session_data.get("model_request").llm_provider if session_data.get("model_request") else None,
                "llm_model": session_data.get("model_request").llm_model if session_data.get("model_request") else None
            }
            
            # Convert datetime to ISO format string for JSON serialization
            start_time = session_data.get("created_at")
            if start_time and hasattr(start_time, 'isoformat'):
                start_time = start_time.isoformat()
            
            # Prepare data for interview_sessions table
            # Note: session_id is the primary key, so we don't include it in the data
            # The database will generate a new UUID for the primary key
            session_record = {
                "interviewee_id": session_data.get("model_request").user_id if session_data.get("model_request") else None,  # User ID from session creation
                "template_id": None,  # Could be extracted from job_role if needed
                "start_time": start_time,  # Use created_at as start_time (converted to ISO string)
                "status": "completed",
                "session_information": session_information,
                "resume_url": session_data.get("model_request").resume_url if session_data.get("model_request") else None,  # Separate column
                "recording_url": recording_url  # Separate column
            }
            
            # Save to database
            saved_session_id = supabase_config.save_interview_session(session_record)
            if saved_session_id:
                # Store the mapping between original session ID and database session ID
                if not hasattr(self, 'session_id_mapping'):
                    self.session_id_mapping = {}
                self.session_id_mapping[session_id] = saved_session_id
                
                logger.info(f"Session {session_id} information saved to database with ID: {saved_session_id}")
                logger.info(f"  - Start time: {session_record['start_time']}")
                logger.info(f"  - Resume URL: {session_record['resume_url']}")
                logger.info(f"  - Recording URL: {session_record['recording_url']}")
                logger.info(f"  - Conversation messages: {len(conversation_history)}")
                logger.info(f"  - Anomalies detected: {len(anomalies_detected)}")
            else:
                logger.error(f"Failed to save session {session_id} information to database")
                
        except Exception as e:
            logger.error(f"Error saving session {session_id} to database: {e}")
            raise
    
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
        """Save recording to Supabase storage or local storage as fallback, with video annotation for detection only."""
        try:
            # First, save the raw recording locally for processing
            session_dir = self.recordings_dir / session_id
            session_dir.mkdir(exist_ok=True)
            
            # Save the raw recording file locally
            raw_file_path = session_dir / filename
            with open(raw_file_path, 'wb') as f:
                f.write(recording_data)
            
            raw_file_size = len(recording_data)
            logger.info(f"Raw recording saved locally for session {session_id}: {raw_file_path} ({raw_file_size} bytes)")
            
            # Check if this is a video file and video processing is available
            video_extensions = ['.mp4', '.avi', '.mov', '.webm', '.mkv']
            is_video = any(filename.lower().endswith(ext) for ext in video_extensions)
            
            # Always use raw footage for upload, but process for detection if it's a video
            final_filename = filename
            final_data = recording_data
            final_size = raw_file_size
            
            if is_video and self.video_processor:
                try:
                    logger.info(f"Processing video for anomaly detection (session {session_id})")
                    
                    # Create temporary annotated video filename for detection processing only
                    name_without_ext = Path(filename).stem
                    temp_annotated_filename = f"{name_without_ext}_temp_annotated.webm"
                    temp_annotated_file_path = session_dir / temp_annotated_filename
                    
                    logger.info(f"Raw video path: {raw_file_path}")
                    logger.info(f"Temporary annotated video path: {temp_annotated_file_path}")
                    
                    # Annotate the video for detection purposes only
                    detection_stats = self.video_processor.annotate_video(
                        str(raw_file_path), 
                        str(temp_annotated_file_path)
                    )
                    
                    # Save detection data
                    detection_data_path = self.sessions_dir / f"capture_data_{session_id}.json"
                    self.video_processor.save_detection_data(session_id, detection_stats, str(detection_data_path))
                    
                    logger.info(f"Video detection processing completed for session {session_id}")
                    logger.info(f"Detection stats: {detection_stats['person_detections']} persons, {detection_stats['device_detections']} devices")
                    
                    # Clean up temporary annotated file (we don't need to keep it)
                    if temp_annotated_file_path.exists():
                        temp_annotated_file_path.unlink()
                        logger.info(f"Temporary annotated video cleaned up: {temp_annotated_file_path}")
                    
                except Exception as video_error:
                    logger.error(f"Video detection processing failed for session {session_id}: {video_error}")
                    import traceback
                    logger.error(f"Video detection error details: {traceback.format_exc()}")
                    # Continue with raw footage upload even if detection fails
            
            # Try to upload raw footage to Supabase
            if SUPABASE_AVAILABLE and supabase_config.client:
                try:
                    upload_result = supabase_config.upload_recording(session_id, final_data, final_filename)
                    
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
                    
                    logger.info(f"Raw recording uploaded to Supabase for session {session_id}: {upload_result['file_path']} ({upload_result['file_size']} bytes)")
                    return recording_info
                    
                except Exception as supabase_error:
                    logger.warning(f"Supabase upload failed, falling back to local storage: {supabase_error}")
            
            # Fallback to local storage
            final_file_path = session_dir / final_filename
            with open(final_file_path, 'wb') as f:
                f.write(final_data)
            
            # Create recording info for local storage
            recording_info = RecordingInfo(
                session_id=session_id,
                file_path=str(final_file_path),
                file_size=final_size,
                upload_time=datetime.now(),
                storage_type="local"
            )
            
            # Store recording info
            self.recordings[session_id] = recording_info
            
            logger.info(f"Raw recording saved locally for session {session_id}: {final_file_path} ({final_size} bytes)")
            return recording_info
            
        except Exception as e:
            logger.error(f"Failed to save recording for session {session_id}: {e}")
            raise

    def get_recording_info(self, session_id: str) -> Optional[RecordingInfo]:
        """Get recording info for a session."""
        return self.recordings.get(session_id)
    
    def get_detection_data(self, session_id: str) -> Optional[Dict]:
        """Get detection data for a session."""
        try:
            detection_data_path = self.sessions_dir / f"capture_data_{session_id}.json"
            if detection_data_path.exists():
                with open(detection_data_path, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Failed to get detection data for session {session_id}: {e}")
            return None


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
            session["model_request"].job_role,
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
        
        # Generate filename based on content type
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        content_type = recording.content_type or "video/webm"
        
        # Determine file extension based on content type
        if "video/mp4" in content_type:
            extension = ".mp4"
        elif "video/webm" in content_type:
            extension = ".webm"
        elif "video/avi" in content_type:
            extension = ".avi"
        elif "video/mov" in content_type:
            extension = ".mov"
        else:
            extension = ".webm"  # Default fallback
        
        filename = f"{recording_type}_recording_{session_id}_{timestamp}{extension}"
        
        # Save recording (will try Supabase first, fallback to local)
        recording_info = session_manager.save_recording(session_id, recording_data, filename)
        
        # Try to update existing session in database with recording URL
        if recording_info.public_url:
            try:
                # Check if session exists in database and update with recording URL
                if SUPABASE_AVAILABLE and supabase_config.client:
                    # Get the database session ID from the mapping
                    db_session_id = None
                    if hasattr(session_manager, 'session_id_mapping') and session_id in session_manager.session_id_mapping:
                        db_session_id = session_manager.session_id_mapping[session_id]
                        logger.info(f"Found database session ID mapping: {session_id} -> {db_session_id}")
                        update_success = supabase_config.update_session_recording_url(db_session_id, recording_info.public_url)
                    else:
                        # Try to find the session by looking for recent sessions without recording URLs
                        logger.info(f"No session ID mapping found for {session_id}, trying to find session in database")
                        update_success = supabase_config.update_session_recording_url_by_original_id(session_id, recording_info.public_url)
                    
                    if update_success:
                        logger.info(f"Updated existing session with recording URL")
                    else:
                        logger.info(f"No existing session found to update with recording URL for {session_id}")
            except Exception as e:
                logger.warning(f"Failed to update session with recording URL: {e}")
        
        # Check if anomaly detection was performed
        detection_data = session_manager.get_detection_data(session_id)
        anomaly_info = ""
        if detection_data and detection_data.get("capture_data"):
            anomalies_detected = []
            capture_data = detection_data["capture_data"]
            
            for timestamp, detections in capture_data.items():
                person_count = sum(1 for detection in detections if "person" in detection.lower())
                device_count = sum(1 for detection in detections if any(device in detection.lower() for device in ["laptop", "phone", "cell phone", "tv", "remote", "computer", "monitor", "keyboard", "mouse"]))
                
                if person_count > 1:
                    anomalies_detected.append(f"Multiple persons ({person_count})")
                if device_count > 0:
                    anomalies_detected.append(f"Digital devices ({device_count})")
            
            if anomalies_detected:
                anomaly_info = f" | Anomalies detected: {', '.join(set(anomalies_detected))}"
            else:
                anomaly_info = " | No anomalies detected"
        
        # Prepare response message
        if recording_info.storage_type == "supabase":
            message = f"Raw recording uploaded to Supabase successfully: {recording_info.file_path}{anomaly_info}"
        else:
            message = f"Raw recording saved locally (Supabase unavailable): {recording_info.file_path}{anomaly_info}"
        
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
            # Look for all video files
            video_extensions = ["*.webm", "*.mp4", "*.avi", "*.mov", "*.mkv"]
            for extension in video_extensions:
                for file_path in session_dir.glob(extension):
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

@app.get("/recordings/{session_id}/detection-data")
async def get_detection_data(session_id: str):
    """Get detection data for a session."""
    try:
        # Validate session exists
        if session_id not in session_manager.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get detection data from session manager
        detection_data = session_manager.get_detection_data(session_id)
        
        if detection_data is None:
            return {
                "session_id": session_id,
                "detection_data_available": False,
                "message": "No detection data found for this session"
            }
        
        return {
            "session_id": session_id,
            "detection_data_available": True,
            "detection_data": detection_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get detection data for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recordings/{session_id}/anomalies")
async def get_anomaly_detection_summary(session_id: str):
    """Get anomaly detection summary for a session."""
    try:
        # Validate session exists
        if session_id not in session_manager.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get detection data from session manager
        detection_data = session_manager.get_detection_data(session_id)
        
        if detection_data is None:
            return {
                "session_id": session_id,
                "anomalies_detected": False,
                "message": "No detection data found for this session"
            }
        
        # Analyze detection data for anomalies
        anomalies_detected = []
        if detection_data.get("capture_data"):
            capture_data = detection_data["capture_data"]
            
            for timestamp, detections in capture_data.items():
                # Count persons and devices more accurately
                person_count = 0
                device_count = 0
                
                for detection in detections:
                    detection_lower = detection.lower()
                    if "person" in detection_lower:
                        person_count += 1
                    elif any(device in detection_lower for device in ["laptop", "phone", "cell phone", "tv", "remote", "computer", "monitor", "keyboard", "mouse"]):
                        device_count += 1
                
                # Flag anomalies: multiple people or devices detected
                if person_count > 1:
                    anomalies_detected.append({
                        "timestamp": timestamp,
                        "type": "multiple_persons",
                        "count": person_count,
                        "detections": detections,
                        "description": f"Multiple persons detected in frame: {person_count} people"
                    })
                if device_count > 0:
                    anomalies_detected.append({
                        "timestamp": timestamp,
                        "type": "devices_detected",
                        "count": device_count,
                        "detections": detections,
                        "description": f"Digital devices detected in frame: {device_count} devices"
                    })
        
        # Calculate summary statistics
        total_frames_with_detections = len(detection_data.get("capture_data", {}))
        total_frames_processed = detection_data.get("statistics", {}).get("processed_frames", 0)
        total_person_detections = detection_data.get("statistics", {}).get("person_detections", 0)
        total_device_detections = detection_data.get("statistics", {}).get("device_detections", 0)
        
        multiple_persons_count = sum(1 for anomaly in anomalies_detected if anomaly["type"] == "multiple_persons")
        devices_count = sum(1 for anomaly in anomalies_detected if anomaly["type"] == "devices_detected")
        
        return {
            "session_id": session_id,
            "anomalies_detected": len(anomalies_detected) > 0,
            "total_anomalies": len(anomalies_detected),
            "anomaly_types": {
                "multiple_persons": multiple_persons_count,
                "devices_detected": devices_count
            },
            "detection_statistics": {
                "total_frames_processed": total_frames_processed,
                "total_frames_with_detections": total_frames_with_detections,
                "total_person_detections": total_person_detections,
                "total_device_detections": total_device_detections
            },
            "anomalies": anomalies_detected
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get anomaly detection summary for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# API endpoints
@app.post("/sessions/create")
async def create_session(model_request: ModelSelectionRequest):
    """Create a new interview session."""
    session_id = str(uuid.uuid4())  # Generate proper UUID
    
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

@app.post("/sessions/{session_id}/save")
async def save_session_information(session_id: str):
    """Manually save session information to database."""
    try:
        if session_id not in session_manager.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_data = session_manager.sessions[session_id]
        session_manager.save_session_to_database(session_id, session_data)
        
        return {
            "status": "saved",
            "session_id": session_id,
            "message": "Session information saved to database successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions/{session_id}/save-missed")
async def save_missed_session(session_id: str):
    """Manually save a missed session that may not be in memory anymore."""
    try:
        # Check if session is still in memory
        if session_id in session_manager.sessions:
            session_data = session_manager.sessions[session_id]
            session_manager.save_session_to_database(session_id, session_data)
            return {
                "status": "saved",
                "session_id": session_id,
                "message": "Session information saved to database successfully"
            }
        else:
            # Session not in memory, try to reconstruct from available data
            logger.warning(f"Session {session_id} not in memory, attempting to reconstruct")
            
            # Check if we have recording info
            recording_info = session_manager.get_recording_info(session_id)
            if not recording_info:
                raise HTTPException(status_code=404, detail="Session not found and no recording available")
            
            # Get recording URL
            recording_url = recording_info.public_url if recording_info.public_url else None
            
            # Convert datetime to ISO format string for JSON serialization
            upload_time = recording_info.upload_time
            if upload_time and hasattr(upload_time, 'isoformat'):
                upload_time = upload_time.isoformat()
            
            # Create minimal session data from recording info
            session_data = {
                "id": session_id,
                "status": "completed",
                "created_at": recording_info.upload_time,
                "model_request": None,  # We don't have this info
                "conversation_history": [],  # We don't have this info
                "resume_content": "Not available"
            }
            
            # Prepare minimal session information
            session_information = {
                "job_role": "Unknown",  # We don't have this info
                "conversation_history": [],
                "anomalies_detected": [],
                "asr_model": "Unknown",
                "llm_provider": "Unknown",
                "llm_model": "Unknown"
            }
            
            # Prepare data for interview_sessions table
            session_record = {
                "interviewee_id": None,
                "template_id": None,
                "start_time": upload_time,  # Use upload_time as start_time (converted to ISO string)
                "status": "completed",
                "session_information": session_information,
                "resume_url": None,  # We don't have this info
                "recording_url": recording_url
            }
            
            # Try to save what we have
            saved_session_id = supabase_config.save_interview_session(session_record)
            if saved_session_id:
                logger.info(f"Partial session {session_id} saved with recording URL: {recording_url}")
                return {
                    "status": "saved_partial",
                    "session_id": session_id,
                    "message": f"Partial session information saved (recording only: {recording_url})"
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to save partial session")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save missed session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions/{session_id}/end")
async def end_session(session_id: str):
    """End a session gracefully - save information and then delete."""
    try:
        if session_id not in session_manager.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Save session information first
        session_data = session_manager.sessions[session_id]
        session_manager.save_session_to_database(session_id, session_data)
        
        # Then delete the session
        session_manager.delete_session(session_id)
        
        return {
            "status": "ended",
            "session_id": session_id,
            "message": "Session ended and information saved to database"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to end session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/info")
async def get_session_information(session_id: str):
    """Get session information that would be saved to database."""
    try:
        if session_id not in session_manager.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_data = session_manager.sessions[session_id]
        
        # Extract conversation history (excluding system prompt and initial "Hello")
        conversation_history = []
        for message in session_data.get("conversation_history", []):
            if message.get("role") == "system":
                continue  # Skip system prompt
            if message.get("role") == "user" and message.get("content") == "Hello":
                continue  # Skip initial "Hello"
            conversation_history.append(message)
        
        # Get recording URL if available
        recording_url = None
        recording_info = session_manager.get_recording_info(session_id)
        if recording_info and recording_info.public_url:
            recording_url = recording_info.public_url
        
        # Get detection data for anomalies
        detection_data = session_manager.get_detection_data(session_id)
        anomalies_detected = []
        if detection_data and detection_data.get("capture_data"):
            # Analyze detection data for anomalies
            capture_data = detection_data["capture_data"]
            for timestamp, detections in capture_data.items():
                # Count persons and devices more accurately
                person_count = 0
                device_count = 0
                
                for detection in detections:
                    detection_lower = detection.lower()
                    if "person" in detection_lower:
                        person_count += 1
                    elif any(device in detection_lower for device in ["laptop", "phone", "cell phone", "tv", "remote", "computer", "monitor", "keyboard", "mouse"]):
                        device_count += 1
                
                # Flag anomalies: multiple people or devices detected
                if person_count > 1:
                    anomalies_detected.append({
                        "timestamp": timestamp,
                        "type": "multiple_persons",
                        "count": person_count,
                        "detections": detections,
                        "description": f"Multiple persons detected in frame: {person_count} people"
                    })
                if device_count > 0:
                    anomalies_detected.append({
                        "timestamp": timestamp,
                        "type": "devices_detected",
                        "count": device_count,
                        "detections": detections,
                        "description": f"Digital devices detected in frame: {device_count} devices"
                    })
        
        # Prepare session information (without date_of_interview, resume_url, recording_url)
        session_information = {
            "job_role": session_data.get("model_request").job_role if session_data.get("model_request") else None,
            "conversation_history": conversation_history,
            "anomalies_detected": anomalies_detected,
            "asr_model": session_data.get("model_request").asr_model if session_data.get("model_request") else None,
            "llm_provider": session_data.get("model_request").llm_provider if session_data.get("model_request") else None,
            "llm_model": session_data.get("model_request").llm_model if session_data.get("model_request") else None
        }
        
        return {
            "session_id": session_id,
            "start_time": session_data.get("created_at"),
            "session_information": session_information,
            "resume_url": session_data.get("model_request").resume_url if session_data.get("model_request") else None,
            "recording_url": recording_url
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session information for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/job-templates")
async def get_job_templates():
    """Get available job templates from Supabase."""
    try:
        if SUPABASE_AVAILABLE and supabase_config.client:
            # Query the job_templates table for active templates
            response = supabase_config.client.table('job_templates').select('*').eq('is_active', True).execute()
            
            if response.data:
                return {
                    "job_templates": response.data
                }
            else:
                return {
                    "job_templates": []
                }
        else:
            # Return sample data if Supabase is not available
            return {
                "job_templates": [
                    {
                        "template_id": "sample-1",
                        "template_name": "Software Engineer",
                        "job_role": "Software Engineer",
                        "difficulty_level": "Intermediate",
                        "estimated_duration": 30,
                        "description": "Full-stack development role"
                    },
                    {
                        "template_id": "sample-2", 
                        "template_name": "Data Scientist",
                        "job_role": "Data Scientist",
                        "difficulty_level": "Advanced",
                        "estimated_duration": 45,
                        "description": "Machine learning and data analysis role"
                    }
                ]
            }
    except Exception as e:
        logger.error(f"Failed to fetch job templates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch job templates: {str(e)}")

@app.get("/user-resumes/{user_id}")
async def get_user_resumes(user_id: str):
    """Get resumes for a specific user from Supabase."""
    try:
        if SUPABASE_AVAILABLE and supabase_config.client:
            # Query the user_profiles table for the specific user's resume
            response = supabase_config.client.table('user_profiles').select('user_id, first_name, last_name, resume_url, resume_filename').eq('user_id', user_id).not_.is_('resume_url', 'null').execute()
            
            if response.data:
                return {
                    "user_resumes": response.data
                }
            else:
                return {
                    "user_resumes": []
                }
        else:
            # Return sample data if Supabase is not available
            return {
                "user_resumes": [
                    {
                        "user_id": "23a0b603-e437-42d6-b1e0-6e0a1b983150",
                        "first_name": "John",
                        "last_name": "Doe",
                        "resume_url": "23a0b603-e437-42d6-b1e0-6e0a1b983150/resume_1754526365292.pdf",
                        "resume_filename": "resume_1754526365292.pdf"
                    }
                ]
            }
    except Exception as e:
        logger.error(f"Failed to fetch user resumes for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch user resumes: {str(e)}")

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
        
        # Update session activity
        session_manager.update_session_activity(session_id)

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
            
            # Update session activity on each message
            session_manager.update_session_activity(session_id)
            
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
        # --- NEW: Auto-save session when WebSocket disconnects ---
        try:
            if session_id in session_manager.sessions:
                logger.info(f"Auto-saving session {session_id} due to WebSocket disconnect")
                session_data = session_manager.sessions[session_id]
                session_manager.save_session_to_database(session_id, session_data)
                logger.info(f"Session {session_id} auto-saved successfully")
        except Exception as e:
            logger.error(f"Failed to auto-save session {session_id} on disconnect: {e}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        await websocket.send_text(json.dumps({"error": str(e)}))
        # --- NEW: Auto-save session on any WebSocket error ---
        try:
            if session_id in session_manager.sessions:
                logger.info(f"Auto-saving session {session_id} due to WebSocket error")
                session_data = session_manager.sessions[session_id]
                session_manager.save_session_to_database(session_id, session_data)
                logger.info(f"Session {session_id} auto-saved successfully after error")
        except Exception as save_error:
            logger.error(f"Failed to auto-save session {session_id} after error: {save_error}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 