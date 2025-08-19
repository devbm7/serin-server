#!/usr/bin/env python3
"""
Test script to verify session completion flow and identify issues with session updates.
"""

import requests
import json
import uuid
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"

def test_session_completion_flow():
    """Test the session completion flow to identify issues."""
    
    print("Testing session completion flow...")
    
    # Step 1: Create a test session ID (simulate frontend creation)
    session_id = str(uuid.uuid4())
    print(f"\n1. Using test session ID: {session_id}")
    
    # Step 2: Test session initialization
    print("\n2. Testing session initialization...")
    
    init_request = {
        "session_id": session_id,
        "job_role": "Test Engineer",
        "user_id": "test-user-id",
        "resume_url": "test-user-id/resume.pdf",
        "llm_provider": "gemini",
        "llm_model": "gemini-2.5-flash",
        "asr_model": "openai/whisper-medium"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/sessions/initialize",
            json=init_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Session initialized successfully")
            print(f"   Status: {result.get('status')}")
        else:
            print(f"   ❌ Session initialization failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error initializing session: {e}")
        return False
    
    # Step 3: Test session save (simulate interview completion)
    print("\n3. Testing session save...")
    
    try:
        response = requests.post(f"{BASE_URL}/sessions/{session_id}/save")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Session saved successfully")
            print(f"   Message: {result.get('message')}")
        else:
            print(f"   ❌ Session save failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error saving session: {e}")
        return False
    
    # Step 4: Test session info retrieval
    print("\n4. Testing session info retrieval...")
    
    try:
        response = requests.get(f"{BASE_URL}/sessions/{session_id}/info")
        
        if response.status_code == 200:
            session_info = response.json()
            print(f"   ✅ Session info retrieved successfully")
            print(f"   Session ID: {session_info.get('session_id')}")
            print(f"   Start Time: {session_info.get('start_time')}")
            print(f"   Resume URL: {session_info.get('resume_url')}")
            print(f"   Recording URL: {session_info.get('recording_url')}")
            
            # Check session information
            session_information = session_info.get('session_information', {})
            print(f"   Job Role: {session_information.get('job_role')}")
            print(f"   Conversation History Length: {len(session_information.get('conversation_history', []))}")
            print(f"   Anomalies Detected: {len(session_information.get('anomalies_detected', []))}")
        else:
            print(f"   ❌ Session info retrieval failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error retrieving session info: {e}")
        return False
    
    # Step 5: Test session completion (simulate interview ending)
    print("\n5. Testing session completion (interview ending)...")
    
    try:
        response = requests.post(f"{BASE_URL}/sessions/{session_id}/complete")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Session completed successfully")
            print(f"   Status: {result.get('status')}")
            print(f"   Message: {result.get('message')}")
        else:
            print(f"   ❌ Session completion failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error completing session: {e}")
        return False
    
    # Step 6: Test session info retrieval after completion
    print("\n6. Testing session info retrieval after completion...")
    
    try:
        response = requests.get(f"{BASE_URL}/sessions/{session_id}/info")
        
        if response.status_code == 200:
            session_info = response.json()
            print(f"   ✅ Session info retrieved successfully")
            print(f"   Session ID: {session_info.get('session_id')}")
            print(f"   Start Time: {session_info.get('start_time')}")
            print(f"   Resume URL: {session_info.get('resume_url')}")
            print(f"   Recording URL: {session_info.get('recording_url')}")
            
            # Check session information
            session_information = session_info.get('session_information', {})
            print(f"   Job Role: {session_information.get('job_role')}")
            print(f"   Conversation History Length: {len(session_information.get('conversation_history', []))}")
            print(f"   Anomalies Detected: {len(session_information.get('anomalies_detected', []))}")
        else:
            print(f"   ❌ Session info retrieval failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error retrieving session info: {e}")
        return False
    
    # Step 7: Test recording upload (simulate recording being available after interview ends)
    print("\n7. Testing recording upload after session completion...")
    
    # Create a dummy recording file
    dummy_recording_data = b"dummy recording data for testing"
    
    try:
        response = requests.post(
            f"{BASE_URL}/recordings/upload/{session_id}",
            files={"recording": ("test_recording.webm", dummy_recording_data, "video/webm")},
            data={"recording_type": "session"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Recording uploaded successfully")
            print(f"   File Path: {result.get('file_path')}")
            print(f"   File Size: {result.get('file_size')}")
            print(f"   Message: {result.get('message')}")
        else:
            print(f"   ❌ Recording upload failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error uploading recording: {e}")
        return False
    
    # Step 8: Test session info retrieval after recording upload
    print("\n8. Testing session info retrieval after recording upload...")
    
    try:
        response = requests.get(f"{BASE_URL}/sessions/{session_id}/info")
        
        if response.status_code == 200:
            session_info = response.json()
            print(f"   ✅ Session info retrieved successfully")
            print(f"   Recording URL: {session_info.get('recording_url')}")
            
            # Check if recording URL was updated
            if session_info.get('recording_url'):
                print(f"   ✅ Recording URL was updated successfully")
            else:
                print(f"   ⚠️  Recording URL was not updated")
                
            # Check session information is still intact
            session_information = session_info.get('session_information', {})
            print(f"   Job Role: {session_information.get('job_role')}")
            print(f"   Conversation History Length: {len(session_information.get('conversation_history', []))}")
            print(f"   Anomalies Detected: {len(session_information.get('anomalies_detected', []))}")
        else:
            print(f"   ❌ Session info retrieval failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error retrieving session info: {e}")
        return False
    
    # Step 9: Clean up
    print("\n9. Cleaning up...")
    
    try:
        response = requests.delete(f"{BASE_URL}/sessions/{session_id}")
        
        if response.status_code == 200:
            print(f"   ✅ Session deleted successfully")
        else:
            print(f"   ⚠️  Session deletion failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"   ⚠️  Error deleting session: {e}")
    
    print("\n✅ Session completion flow test completed!")
    return True

def test_database_session_lookup():
    """Test database session lookup functionality."""
    
    print("\n=== Testing Database Session Lookup ===")
    
    # This would require direct database access to test
    # For now, we'll just test the API endpoints that interact with the database
    
    print("Database session lookup test requires direct database access.")
    print("The issue might be in the session lookup logic in supabase_config.py")
    
    return True

if __name__ == "__main__":
    print("=== Session Completion Flow Test ===")
    
    # Test the session completion flow
    if test_session_completion_flow():
        print("\n🎉 Session completion flow test passed!")
    else:
        print("\n❌ Session completion flow test failed!")
        exit(1)
    
    # Test database session lookup
    test_database_session_lookup()
