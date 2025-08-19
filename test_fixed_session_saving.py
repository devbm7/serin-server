#!/usr/bin/env python3
"""
Test script to verify the fixed session saving functionality
"""

import requests
import json
import uuid
from datetime import datetime

# FastAPI server URL
BASE_URL = "http://localhost:8000"

def test_fixed_session_saving():
    """Test the fixed session saving functionality."""
    
    print("Testing fixed session saving functionality...")
    
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
    print(f"   - Start Time: {session_info_data['start_time']}")
    print(f"   - Job Role: {session_info_data['session_information']['job_role']}")
    print(f"   - Resume URL: {session_info_data['resume_url']}")
    print(f"   - Recording URL: {session_info_data['recording_url']}")
    print(f"   - Conversation History: {len(session_info_data['session_information']['conversation_history'])} messages")
    print(f"   - Anomalies: {len(session_info_data['session_information']['anomalies_detected'])} detected")
    
    # Verify the structure is correct
    session_information = session_info_data['session_information']
    if 'date_of_interview' in session_information:
        print("❌ ERROR: date_of_interview should NOT be in session_information")
        return
    else:
        print("✅ date_of_interview correctly removed from session_information")
    
    if 'resume_url' in session_information:
        print("❌ ERROR: resume_url should NOT be in session_information")
        return
    else:
        print("✅ resume_url correctly moved to separate column")
    
    if 'recording_url' in session_information:
        print("❌ ERROR: recording_url should NOT be in session_information")
        return
    else:
        print("✅ recording_url correctly moved to separate column")
    
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
    print("\n✅ FIXES VERIFIED:")
    print("   - Date moved to start_time column")
    print("   - resume_url moved to separate column")
    print("   - recording_url moved to separate column")
    print("   - session_information JSON structure cleaned up")

if __name__ == "__main__":
    try:
        test_fixed_session_saving()
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to FastAPI server. Make sure it's running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
