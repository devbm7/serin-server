#!/usr/bin/env python3
"""
Script to download Whisper model locally for faster startup
"""

import os
import sys
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_whisper_model(model_name="openai/whisper-medium", local_dir="whisper-medium"):
    """Download Whisper model to local directory."""
    try:
        local_path = Path(local_dir)
        
        if local_path.exists():
            logger.info(f"Model directory {local_dir} already exists")
            return True
        
        logger.info(f"Downloading {model_name} to {local_dir}...")
        
        # Create directory
        local_path.mkdir(exist_ok=True)
        
        # Download processor
        logger.info("Downloading processor...")
        processor = WhisperProcessor.from_pretrained(model_name)
        processor.save_pretrained(local_dir)
        
        # Download model
        logger.info("Downloading model...")
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        model.save_pretrained(local_dir)
        
        logger.info(f"Model downloaded successfully to {local_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return False

def main():
    """Main function."""
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "openai/whisper-medium"
    
    if len(sys.argv) > 2:
        local_dir = sys.argv[2]
    else:
        local_dir = "whisper-medium"
    
    logger.info(f"Downloading {model_name} to {local_dir}")
    
    success = download_whisper_model(model_name, local_dir)
    
    if success:
        logger.info("✅ Model download completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ Model download failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
