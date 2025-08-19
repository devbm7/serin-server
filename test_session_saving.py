#!/usr/bin/env python3
"""
Test script for session saving functionality
"""

import requests
import json
import time
import uuid
from datetime import datetime

# FastAPI server URL
BASE_URL = "http://localhost:8000"

def test_session_saving():
    """Test the session saving functionality."""
    
    print("Testing session saving functionality...")
    
    # 1. Create a session
    print("\n1. Creating a test session...")
    session_data = {
        "job_role": "Software Engineer",
        "user_id": "test-user-123",
        "resume_url": "test-user-123/resume.pdf",
        "asr_model": "openai/whisper-medium",
        "llm_provider": "gemini",
        "llm_model": "gemini-2.5-flash"
    }
    
    response = requests.post(f"{BASE_URL}/sessions/create", json=session_data)
    if response.status_code != 200:
        print(f"❌ Failed to create session: {response.status_code}")
        print(response.text)
        return
    
    session_info = response.json()
    session_id = session_info["session_id"]
    print(f"✅ Session created: {session_id}")
    
    # Validate UUID format
    try:
        uuid.UUID(session_id)
        print(f"✅ Session ID is valid UUID: {session_id}")
    except ValueError:
        print(f"❌ Session ID is not a valid UUID: {session_id}")
        return
    
    # 2. Get session information
    print("\n2. Getting session information...")
    response = requests.get(f"{BASE_URL}/sessions/{session_id}/info")
    if response.status_code != 200:
        print(f"❌ Failed to get session info: {response.status_code}")
        print(response.text)
        return
    
    session_info_data = response.json()
    print(f"✅ Session info retrieved:")
    print(f"   - Job Role: {session_info_data['session_information']['job_role']}")
    print(f"   - Conversation History: {len(session_info_data['session_information']['conversation_history'])} messages")
    print(f"   - Anomalies: {len(session_info_data['session_information']['anomalies_detected'])} detected")
    
    # 3. Save session information
    print("\n3. Saving session information...")
    response = requests.post(f"{BASE_URL}/sessions/{session_id}/save")
    if response.status_code != 200:
        print(f"❌ Failed to save session: {response.status_code}")
        print(response.text)
        return
    
    save_result = response.json()
    print(f"✅ Session saved: {save_result['message']}")
    
    # 4. End session gracefully
    print("\n4. Ending session gracefully...")
    response = requests.post(f"{BASE_URL}/sessions/{session_id}/end")
    if response.status_code != 200:
        print(f"❌ Failed to end session: {response.status_code}")
        print(response.text)
        return
    
    end_result = response.json()
    print(f"✅ Session ended: {end_result['message']}")
    
    # 5. Verify session is deleted
    print("\n5. Verifying session is deleted...")
    response = requests.get(f"{BASE_URL}/sessions/{session_id}")
    if response.status_code == 404:
        print("✅ Session successfully deleted")
    else:
        print(f"❌ Session still exists: {response.status_code}")
    
    print("\n🎉 All tests completed successfully!")

def test_session_with_anomalies():
    """Test session saving with simulated anomalies."""
    
    print("\n" + "="*50)
    print("Testing session saving with anomalies...")
    
    # Create a session
    session_data = {
        "job_role": "Data Scientist",
        "user_id": "test-user-456",
        "resume_url": "test-user-456/resume.pdf",
        "asr_model": "openai/whisper-medium",
        "llm_provider": "gemini",
        "llm_model": "gemini-2.5-flash"
    }
    
    response = requests.post(f"{BASE_URL}/sessions/create", json=session_data)
    if response.status_code != 200:
        print(f"❌ Failed to create session: {response.status_code}")
        return
    
    session_id = response.json()["session_id"]
    print(f"✅ Session created: {session_id}")
    
    # Validate UUID format
    try:
        uuid.UUID(session_id)
        print(f"✅ Session ID is valid UUID: {session_id}")
    except ValueError:
        print(f"❌ Session ID is not a valid UUID: {session_id}")
        return
    
    # Simulate some detection data (anomalies)
    detection_data = {
        "session_id": session_id,
        "capture_data": {
            "2025-01-01 10:00:00": ["person 0.95", "person 0.87", "laptop 0.72"],
            "2025-01-01 10:01:00": ["person 0.92", "phone 0.68"],
            "2025-01-01 10:02:00": ["person 0.89", "tv 0.75"]
        },
        "statistics": {
            "total_frames": 100,
            "processed_frames": 100,
            "detection_frames": 3,
            "person_detections": 3,
            "device_detections": 3
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Save detection data
    import os
    sessions_dir = os.path.join("sessions")
    os.makedirs(sessions_dir, exist_ok=True)
    
    detection_file = os.path.join(sessions_dir, f"capture_data_{session_id}.json")
    with open(detection_file, 'w') as f:
        json.dump(detection_data, f, indent=2)
    
    print(f"✅ Detection data saved: {detection_file}")
    
    # Get session information (should include anomalies)
    response = requests.get(f"{BASE_URL}/sessions/{session_id}/info")
    if response.status_code == 200:
        session_info = response.json()
        anomalies = session_info['session_information']['anomalies_detected']
        print(f"✅ Anomalies detected: {len(anomalies)}")
        for anomaly in anomalies:
            print(f"   - {anomaly['type']}: {anomaly['count']} at {anomaly['timestamp']}")
    
    # End session
    response = requests.post(f"{BASE_URL}/sessions/{session_id}/end")
    if response.status_code == 200:
        print("✅ Session ended with anomalies saved")
    
    # Clean up detection file
    if os.path.exists(detection_file):
        os.remove(detection_file)
        print("✅ Detection file cleaned up")

if __name__ == "__main__":
    try:
        test_session_saving()
        test_session_with_anomalies()
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to FastAPI server. Make sure it's running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
