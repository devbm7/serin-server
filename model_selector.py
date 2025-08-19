def get_available_asr_models():
    # Available ASR models
    available_asr_models = {
        "1": {
            "name": "openai/whisper-tiny",
            "description": "Fastest, smallest model (39M parameters)",
            "language": "Multilingual"
        },
        "2": {
            "name": "openai/whisper-base",
            "description": "Small model (74M parameters)",
            "language": "Multilingual"
        },
        "3": {
            "name": "openai/whisper-small",
            "description": "Medium model (244M parameters)",
            "language": "Multilingual"
        },
        "4": {
            "name": "openai/whisper-medium",
            "description": "Large model (769M parameters) - Default",
            "language": "Multilingual"
        },
        "5": {
            "name": "openai/whisper-large",
            "description": "Very large model (1550M parameters)",
            "language": "Multilingual"
        },
        "6": {
            "name": "openai/whisper-large-v3",
            "description": "Latest large model with improved performance",
            "language": "Multilingual"
        },
        "7": {
            "name": "openai/whisper-large-v3-turbo",
            "description": "Latest large model with improved performance",
            "language": "Multilingual"
        }
    }
    return available_asr_models

def get_available_mainllm_models():
    mainllm_models = {
        "1": {"provider": "ollama", "model": "llama3.2:1b"},
        "2": {"provider": "ollama", "model": "deepseek-r1:1.5b"},
        "3": {"provider": "ollama", "model": "gemma3:1b"},
        "4": {"provider": "ollama", "model": "qwen3:0.6b"},
        "5": {"provider": "ollama", "model": "gemma3:latest"},
        "6": {"provider":"gemini", "model": "gemini-2.5-flash"}
    }
    return mainllm_models

def get_available_llm_models():
    """Alias for get_available_mainllm_models for compatibility."""
    return get_available_mainllm_models()
