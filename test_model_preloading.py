#!/usr/bin/env python3
"""
Test script to verify model preloading functionality
"""

import requests
import time
import json

def test_health_endpoint():
    """Test the health endpoint to check preloaded models status"""
    print("Testing health endpoint...")
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            data = response.json()
            print("✓ Health endpoint working")
            print(f"Status: {data['status']}")
            print(f"Active sessions: {data['active_sessions']}")
            print("Preloaded models status:")
            for model_type, status in data['preloaded_models'].items():
                if isinstance(status, dict):
                    print(f"  {model_type}: {len(status)} models loaded")
                else:
                    print(f"  {model_type}: {'✓' if status else '✗'}")
            return True
        else:
            print(f"✗ Health endpoint failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health endpoint error: {e}")
        return False

def test_preloaded_models_endpoint():
    """Test the preloaded models endpoint for detailed information"""
    print("\nTesting preloaded models endpoint...")
    try:
        response = requests.get("http://localhost:8000/models/preloaded")
        if response.status_code == 200:
            data = response.json()
            print("✓ Preloaded models endpoint working")
            print("Detailed model information:")
            for model_type, info in data.items():
                if info is None:
                    print(f"  {model_type}: Not loaded")
                elif isinstance(info, dict):
                    print(f"  {model_type}: {info}")
                else:
                    print(f"  {model_type}: {info}")
            return True
        else:
            print(f"✗ Preloaded models endpoint failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Preloaded models endpoint error: {e}")
        return False

def test_session_creation():
    """Test creating a session to verify preloaded models are used"""
    print("\nTesting session creation...")
    try:
        # Create a test session
        session_data = {
            "job_role": "Software Engineer",
            "user_id": "test-user-123",
            "resume_url": "test/resume.pdf",
            "asr_model": "openai/whisper-medium",
            "llm_provider": "gemini",
            "llm_model": "gemini-2.5-flash"
        }
        
        response = requests.post("http://localhost:8000/sessions/create", json=session_data)
        if response.status_code == 200:
            data = response.json()
            session_id = data['session_id']
            print(f"✓ Session created successfully: {session_id}")
            
            # Get session info
            info_response = requests.get(f"http://localhost:8000/sessions/{session_id}")
            if info_response.status_code == 200:
                print("✓ Session info retrieved successfully")
                
                # Clean up session
                delete_response = requests.delete(f"http://localhost:8000/sessions/{session_id}")
                if delete_response.status_code == 200:
                    print("✓ Session cleaned up successfully")
                else:
                    print(f"✗ Session cleanup failed: {delete_response.status_code}")
            else:
                print(f"✗ Session info retrieval failed: {info_response.status_code}")
            
            return True
        else:
            print(f"✗ Session creation failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Session creation error: {e}")
        return False

def test_available_models():
    """Test the available models endpoint"""
    print("\nTesting available models endpoint...")
    try:
        response = requests.get("http://localhost:8000/models")
        if response.status_code == 200:
            data = response.json()
            print("✓ Available models endpoint working")
            print(f"ASR models: {len(data['asr_models'])} available")
            print(f"LLM models: {len(data['llm_models'])} available")
            return True
        else:
            print(f"✗ Available models endpoint failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Available models endpoint error: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Model Preloading Test Suite ===")
    print("Make sure the FastAPI server is running on localhost:8000")
    print()
    
    tests = [
        test_health_endpoint,
        test_preloaded_models_endpoint,
        test_available_models,
        test_session_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
        print()
    
    print(f"=== Test Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("✓ All tests passed! Model preloading is working correctly.")
    else:
        print("✗ Some tests failed. Check the server logs for more details.")

if __name__ == "__main__":
    main()
