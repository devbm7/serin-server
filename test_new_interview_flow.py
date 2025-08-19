#!/usr/bin/env python3
"""
Test script for the new interview flow:
1. Frontend creates initial session entry in database
2. Server initializes session with the database session ID
3. Server updates the session with completion data
"""

import requests
import json
import uuid
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"

def test_new_interview_flow():
    """Test the complete new interview flow."""
    
    print("Testing new interview flow...")
    
    # Step 1: Simulate frontend creating initial session entry
    print("\n1. Creating initial session entry in database...")
    
    # This would normally be done by the frontend using Supabase client
    # For testing, we'll simulate the data that would be created
    session_id = str(uuid.uuid4())
    initial_session_data = {
        "session_id": session_id,
        "interviewee_id": "test-user-id",
        "template_id": "test-template-id", 
        "start_time": datetime.now().isoformat(),
        "status": "initializing",
        "resume_url": "test-user-id/resume.pdf",
        "session_information": {
            "job_role": "Software Engineer",
            "conversation_history": [],
            "anomalies_detected": [],
            "asr_model": "gemini-2.5-flash",
            "llm_provider": "gemini",
            "llm_model": "gemini-2.5-flash"
        }
    }
    
    print(f"   Session ID: {session_id}")
    print(f"   Job Role: {initial_session_data['session_information']['job_role']}")
    
    # Step 2: Initialize session on server
    print("\n2. Initializing session on server...")
    
    init_request = {
        "session_id": session_id,
        "job_role": "Software Engineer",
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
    
    # Step 3: Test session retrieval
    print("\n3. Testing session retrieval...")
    
    try:
        response = requests.get(f"{BASE_URL}/sessions/{session_id}")
        
        if response.status_code == 200:
            session_info = response.json()
            print(f"   ✅ Session retrieved successfully")
            print(f"   Status: {session_info.get('status')}")
            print(f"   Conversation length: {session_info.get('conversation_length')}")
        else:
            print(f"   ❌ Session retrieval failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error retrieving session: {e}")
        return False
    
    # Step 4: Test session completion (simulate interview end)
    print("\n4. Testing session completion...")
    
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
    
    # Step 5: Clean up
    print("\n5. Cleaning up...")
    
    try:
        response = requests.delete(f"{BASE_URL}/sessions/{session_id}")
        
        if response.status_code == 200:
            print(f"   ✅ Session deleted successfully")
        else:
            print(f"   ⚠️  Session deletion failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"   ⚠️  Error deleting session: {e}")
    
    print("\n✅ New interview flow test completed successfully!")
    return True

def test_health_check():
    """Test the health check endpoint."""
    print("Testing health check...")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Health check passed")
            print(f"   Status: {health_data.get('status')}")
            print(f"   Models ready: {health_data.get('all_models_ready')}")
            return True
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error checking health: {e}")
        return False

if __name__ == "__main__":
    print("=== New Interview Flow Test ===")
    
    # First check if server is healthy
    if not test_health_check():
        print("Server is not healthy, exiting...")
        exit(1)
    
    # Test the new flow
    if test_new_interview_flow():
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        exit(1)
