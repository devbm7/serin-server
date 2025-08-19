#!/usr/bin/env python3
"""
Test script to verify resume ID integration with FastAPI server
"""

import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_resume_id_integration():
    """Test the resume ID integration with the FastAPI server"""
    
    base_url = "http://localhost:8000"
    
    print("=== Testing Resume ID Integration ===")
    
    # Test 1: Check if server is running
    try:
        response = requests.get(f"{base_url}/models")
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print(f"❌ Server returned status code: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure the FastAPI server is running.")
        return
    
    # Test 2: Create session with resume_file (user ID)
    session_data = {
        "interview_topic": "Software Engineering",
        "resume_file": "test_user_123",  # Test user ID
        "asr_model": "openai/whisper-medium",
        "llm_provider": "ollama",
        "llm_model": "llama3.2:1b"
    }
    
    try:
        print("\n🔄 Creating session with resume_id...")
        response = requests.post(f"{base_url}/sessions/create", json=session_data)
        
        if response.status_code == 200:
            result = response.json()
            session_id = result.get("session_id")
            print(f"✅ Session created successfully: {session_id}")
            
            # Test 3: Get session info
            print(f"\n🔄 Getting session info for {session_id}...")
            response = requests.get(f"{base_url}/sessions/{session_id}")
            
            if response.status_code == 200:
                session_info = response.json()
                print("✅ Session info retrieved:")
                print(f"   - Session ID: {session_info.get('session_id')}")
                print(f"   - Status: {session_info.get('status')}")
                print(f"   - Conversation length: {session_info.get('conversation_length')}")
                
                # Test 4: Check if resume content was loaded
                if session_info.get('conversation_length', 0) > 0:
                    print("✅ Resume content appears to have been loaded (conversation history exists)")
                else:
                    print("⚠️  No conversation history found - resume may not have loaded")
                
            else:
                print(f"❌ Failed to get session info: {response.status_code}")
                print(f"   Response: {response.text}")
            
            # Test 5: Clean up - delete session
            print(f"\n🔄 Cleaning up session {session_id}...")
            response = requests.delete(f"{base_url}/sessions/{session_id}")
            
            if response.status_code == 200:
                print("✅ Session deleted successfully")
            else:
                print(f"⚠️  Failed to delete session: {response.status_code}")
                
        else:
            print(f"❌ Failed to create session: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
    
    print("\n=== Test Complete ===")

def test_with_real_user_id():
    """Test with a real user ID that might have a resume in the database"""
    
    base_url = "http://localhost:8000"
    
    print("\n=== Testing with Real User ID ===")
    
    # You can replace this with a real user ID from your database
    real_user_id = input("Enter a real user ID to test with (or press Enter to skip): ").strip()
    
    if not real_user_id:
        print("Skipping real user ID test")
        return
    
    session_data = {
        "interview_topic": "Technical Interview",
        "resume_file": real_user_id,
        "asr_model": "openai/whisper-medium",
        "llm_provider": "ollama",
        "llm_model": "llama3.2:1b"
    }
    
    try:
        print(f"\n🔄 Creating session with real user ID: {real_user_id}")
        response = requests.post(f"{base_url}/sessions/create", json=session_data)
        
        if response.status_code == 200:
            result = response.json()
            session_id = result.get("session_id")
            print(f"✅ Session created successfully: {session_id}")
            
            # Get session info to check resume loading
            response = requests.get(f"{base_url}/sessions/{session_id}")
            if response.status_code == 200:
                session_info = response.json()
                print(f"✅ Session info: {session_info}")
            
            # Clean up
            requests.delete(f"{base_url}/sessions/{session_id}")
            print("✅ Session cleaned up")
            
        else:
            print(f"❌ Failed to create session: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error during real user test: {e}")

if __name__ == "__main__":
    test_resume_id_integration()
    test_with_real_user_id()
