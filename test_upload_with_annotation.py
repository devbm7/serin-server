#!/usr/bin/env python3
"""
Test script to test the actual upload functionality with video annotation.
This script creates a test video and uploads it to the FastAPI server.
"""

import os
import sys
import tempfile
import requests
import time
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_video(output_path: str, duration_seconds: int = 5):
    """Create a simple test video."""
    try:
        import cv2
        import numpy as np
        
        # Video settings
        fps = 20
        width, height = 640, 480
        total_frames = fps * duration_seconds
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Create frames
        for i in range(total_frames):
            # Create a frame with a moving rectangle
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add a moving green rectangle
            x = (i * 10) % (width - 40)
            cv2.rectangle(frame, (x, 200), (x + 40, 240), (0, 255, 0), -1)
            
            # Add some text
            cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print(f"✅ Created test video: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create test video: {e}")
        return False

def test_upload_with_annotation():
    """Test uploading a video with annotation."""
    print("\n🧪 Testing Upload with Video Annotation...")
    
    # FastAPI server URL (adjust if needed)
    base_url = "http://localhost:8000"
    
    # Create a test session first
    session_data = {
        "job_role": "Software Engineer",
        "user_id": "test-user-123",
        "resume_url": "test/resume.pdf",
        "asr_model": "openai/whisper-medium",
        "llm_provider": "gemini",
        "llm_model": "gemini-2.5-flash"
    }
    
    try:
        # Create session
        print("📝 Creating test session...")
        response = requests.post(f"{base_url}/sessions/create", json=session_data)
        if response.status_code != 200:
            print(f"❌ Failed to create session: {response.status_code} - {response.text}")
            return False
        
        session_info = response.json()
        session_id = session_info.get("session_id")
        print(f"✅ Session created: {session_id}")
        
        # Create test video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
            temp_video_path = temp_video.name
        
        if not create_test_video(temp_video_path, duration_seconds=3):
            return False
        
        # Upload video
        print("📤 Uploading video with annotation...")
        with open(temp_video_path, 'rb') as video_file:
            files = {'recording': ('test_video.mp4', video_file, 'video/mp4')}
            data = {'recording_type': 'session'}
            
            response = requests.post(
                f"{base_url}/recordings/upload/{session_id}",
                files=files,
                data=data
            )
        
        if response.status_code != 200:
            print(f"❌ Upload failed: {response.status_code} - {response.text}")
            return False
        
        upload_result = response.json()
        print(f"✅ Upload successful: {upload_result}")
        
        # Check if detection data is available
        print("📊 Checking detection data...")
        time.sleep(2)  # Wait a bit for processing
        
        response = requests.get(f"{base_url}/recordings/{session_id}/detection-data")
        if response.status_code == 200:
            detection_data = response.json()
            print(f"✅ Detection data: {detection_data}")
        else:
            print(f"⚠️  No detection data available: {response.status_code}")
        
        # List recordings
        print("📋 Listing recordings...")
        response = requests.get(f"{base_url}/recordings/{session_id}/list")
        if response.status_code == 200:
            recordings = response.json()
            print(f"✅ Recordings: {recordings}")
        else:
            print(f"❌ Failed to list recordings: {response.status_code}")
        
        # Clean up
        os.unlink(temp_video_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_server_status():
    """Test if the server is running."""
    print("🔍 Checking server status...")
    
    try:
        response = requests.get("http://localhost:8000/docs")
        if response.status_code == 200:
            print("✅ Server is running")
            return True
        else:
            print(f"❌ Server responded with status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running. Please start the FastAPI server first.")
        return False

if __name__ == "__main__":
    print("🚀 Starting Upload with Annotation Test...")
    
    # Check server status
    if not test_server_status():
        print("\n💡 To start the server, run:")
        print("   cd poc")
        print("   python fastapi_pipeline.py")
        sys.exit(1)
    
    # Run the test
    success = test_upload_with_annotation()
    
    if success:
        print("\n🎉 Upload with annotation test completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Upload with annotation test failed!")
        sys.exit(1)
