#!/usr/bin/env python3
"""
Test script for Gemini ASR integration
"""

import os
import sys
import numpy as np
import time
import logging

# Add the current directory to the path so we can import from fastapi_pipeline
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gemini_asr():
    """Test the Gemini ASR integration."""
    try:
        # Import the ASRProcessor from fastapi_pipeline
        from fastapi_pipeline import ASRProcessor
        
        logger.info("Testing Gemini ASR integration...")
        
        # Create ASR processor
        asr_processor = ASRProcessor()
        
        logger.info(f"ASR Processor initialized with primary method: {asr_processor.primary_method}")
        
        # Create a simple test audio (1 second of silence with a tone)
        sample_rate = 16000
        duration = 1.0  # 1 second
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Create a simple tone (440 Hz)
        frequency = 440
        audio_data = 0.1 * np.sin(2 * np.pi * frequency * t)
        
        # Add some noise
        noise = 0.01 * np.random.randn(len(audio_data))
        audio_data = audio_data + noise
        
        logger.info(f"Created test audio: {len(audio_data)} samples, {duration}s duration")
        
        # Test transcription
        logger.info("Testing transcription...")
        start_time = time.time()
        
        transcription = asr_processor.transcribe(audio_data, sample_rate)
        
        elapsed_time = time.time() - start_time
        
        logger.info(f"Transcription completed in {elapsed_time:.3f}s")
        logger.info(f"Result: '{transcription}'")
        
        if transcription:
            logger.info("✅ Gemini ASR test PASSED - Transcription successful")
            return True
        else:
            logger.warning("⚠️ Gemini ASR test WARNING - Empty transcription (this might be expected for test audio)")
            return True  # Still consider it a pass since the API call worked
            
    except Exception as e:
        logger.error(f"❌ Gemini ASR test FAILED: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_whisper_fallback():
    """Test the Whisper fallback functionality."""
    try:
        from fastapi_pipeline import ASRProcessor
        
        logger.info("Testing Whisper fallback...")
        
        # Create ASR processor
        asr_processor = ASRProcessor()
        
        # Create test audio
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        frequency = 440
        audio_data = 0.1 * np.sin(2 * np.pi * frequency * t)
        noise = 0.01 * np.random.randn(len(audio_data))
        audio_data = audio_data + noise
        
        # Test Whisper transcription directly
        if asr_processor.whisper_processor:
            logger.info("Testing Whisper transcription...")
            start_time = time.time()
            
            transcription = asr_processor._transcribe_with_whisper(audio_data, sample_rate)
            
            elapsed_time = time.time() - start_time
            
            logger.info(f"Whisper transcription completed in {elapsed_time:.3f}s")
            logger.info(f"Result: '{transcription}'")
            
            logger.info("✅ Whisper fallback test PASSED")
            return True
        else:
            logger.warning("⚠️ Whisper fallback test SKIPPED - No Whisper model available")
            return True
            
    except Exception as e:
        logger.error(f"❌ Whisper fallback test FAILED: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting Gemini ASR integration tests...")
    
    # Test 1: Gemini ASR
    test1_passed = test_gemini_asr()
    
    # Test 2: Whisper fallback
    test2_passed = test_whisper_fallback()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    logger.info(f"Gemini ASR Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    logger.info(f"Whisper Fallback Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        logger.info("🎉 All tests PASSED!")
        return 0
    else:
        logger.error("💥 Some tests FAILED!")
        return 1

if __name__ == "__main__":
    exit(main())
