#!/usr/bin/env python3
"""
Configuration management for the Interview Agent
"""

import os
from typing import Optional

def load_environment():
    """Load environment variables from .env file if it exists"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("Loaded environment variables from .env file")
    except ImportError:
        print("python-dotenv not installed. Install with: pip install python-dotenv")
    except Exception as e:
        print(f"Could not load .env file: {e}")

def get_gemini_api_key() -> Optional[str]:
    """Get Gemini API key from environment or set default"""
    api_key = os.environ.get('GEMINI_API_KEY')
    
    if not api_key:
        # Set default key for development
        default_key = "AIzaSyD-987BdBsdKnCa7oWZktY9_1K27hS-qY8"
        os.environ['GEMINI_API_KEY'] = default_key
        print("Using default Gemini API key. For production, set GEMINI_API_KEY environment variable.")
        return default_key
    
    return api_key

def validate_gemini_config() -> bool:
    """Validate Gemini configuration"""
    api_key = get_gemini_api_key()
    if not api_key:
        print("Gemini API key not available")
        return False
    
    try:
        import litellm
        # Test the API key with a simple request
        response = litellm.completion(
            model="gemini/gemini-2.5-flash",
            messages=[{"role": "user", "content": "Test"}],
            api_key=api_key,
            max_tokens=10
        )
        print("Gemini API key is valid")
        return True
    except Exception as e:
        print(f"Gemini API key validation failed: {e}")
        return False

def get_model_preload_config() -> dict:
    """Get model preloading configuration"""
    return {
        "preload_asr_models": os.environ.get('PRELOAD_ASR_MODELS', 'true').lower() == 'true',
        "asr_models_to_preload": os.environ.get('ASR_MODELS_TO_PRELOAD', 'openai/whisper-medium,openai/whisper-small').split(','),
        "preload_vad": os.environ.get('PRELOAD_VAD', 'true').lower() == 'true',
        "preload_tts": os.environ.get('PRELOAD_TTS', 'true').lower() == 'true',
        "preload_llm_client": os.environ.get('PRELOAD_LLM_CLIENT', 'true').lower() == 'true',
        "preload_video_processor": os.environ.get('PRELOAD_VIDEO_PROCESSOR', 'true').lower() == 'true',
        "enable_gpu_cleanup": os.environ.get('ENABLE_GPU_CLEANUP', 'true').lower() == 'true'
    }

# Load environment on import
load_environment() 