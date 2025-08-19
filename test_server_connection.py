#!/usr/bin/env python3
"""
Test script to verify server connection and WebSocket functionality
"""

import requests
import websocket
import json
import time
import sys

def test_server_health():
    """Test if the server is running and healthy"""
    print("Testing server health...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✓ Server is healthy")
            print(f"Status: {data['status']}")
            print(f"Active sessions: {data['active_sessions']}")
            print("Preloaded models:")
            for model_type, status in data['preloaded_models'].items():
                if isinstance(status, dict):
                    print(f"  {model_type}: {len(status)} models loaded")
                else:
                    print(f"  {model_type}: {'✓' if status else '✗'}")
            return True
        else:
            print(f"✗ Server health check failed with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server. Is the FastAPI server running on localhost:8000?")
        return False
    except requests.exceptions.Timeout:
        print("✗ Server health check timed out")
        return False
    except Exception as e:
        print(f"✗ Server health check error: {e}")
        return False

def test_session_creation():
    """Test creating a session"""
    print("\nTesting session creation...")
    try:
        session_data = {
            "job_role": "Software Engineer",
            "user_id": "test-user-123",
            "resume_url": "test/resume.pdf",
            "asr_model": "openai/whisper-medium",
            "llm_provider": "gemini",
            "llm_model": "gemini-2.5-flash"
        }
        
        response = requests.post("http://localhost:8000/sessions/create", json=session_data, timeout=30)
        if response.status_code == 200:
            data = response.json()
            session_id = data['session_id']
            print(f"✓ Session created successfully: {session_id}")
            return session_id
        else:
            print(f"✗ Session creation failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"✗ Session creation error: {e}")
        return None

def test_websocket_connection(session_id):
    """Test WebSocket connection"""
    print(f"\nTesting WebSocket connection for session {session_id}...")
    
    def on_message(ws, message):
        print(f"✓ Received WebSocket message: {message}")
        ws.close()
    
    def on_error(ws, error):
        print(f"✗ WebSocket error: {error}")
    
    def on_close(ws, close_status_code, close_msg):
        print(f"WebSocket closed: {close_status_code} - {close_msg}")
    
    def on_open(ws):
        print("✓ WebSocket connected successfully")
        # Send a test message
        test_message = {
            "type": "audio_chunk",
            "audio_data": "dGVzdA==",  # base64 encoded "test"
            "sample_rate": 16000,
            "chunk_size": 512
        }
        ws.send(json.dumps(test_message))
    
    try:
        # Create WebSocket connection
        ws = websocket.WebSocketApp(
            f"ws://localhost:8000/ws/{session_id}",
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        # Set timeout for connection
        ws.run_forever(timeout=20)
        return True
        
    except Exception as e:
        print(f"✗ WebSocket connection error: {e}")
        return False

def test_session_cleanup(session_id):
    """Clean up test session"""
    if session_id:
        print(f"\nCleaning up test session {session_id}...")
        try:
            response = requests.delete(f"http://localhost:8000/sessions/{session_id}")
            if response.status_code == 200:
                print("✓ Session cleaned up successfully")
            else:
                print(f"✗ Session cleanup failed: {response.status_code}")
        except Exception as e:
            print(f"✗ Session cleanup error: {e}")

def main():
    """Run all tests"""
    print("=== Server Connection Test Suite ===")
    print("This script tests the FastAPI server connection and WebSocket functionality.")
    print("Make sure the FastAPI server is running on localhost:8000")
    print()
    
    # Test server health
    if not test_server_health():
        print("\n❌ Server is not available. Please start the FastAPI server first:")
        print("   cd poc")
        print("   python fastapi_pipeline.py")
        sys.exit(1)
    
    # Test session creation
    session_id = test_session_creation()
    if not session_id:
        print("\n❌ Session creation failed. Check server logs for details.")
        sys.exit(1)
    
    # Test WebSocket connection
    websocket_success = test_websocket_connection(session_id)
    
    # Clean up
    test_session_cleanup(session_id)
    
    # Summary
    print("\n=== Test Results ===")
    if websocket_success:
        print("✓ All tests passed! Server and WebSocket are working correctly.")
        print("\nIf you're still experiencing WebSocket timeout issues in the frontend:")
        print("1. Check browser console for CORS errors")
        print("2. Verify the frontend is using the correct WebSocket URL")
        print("3. Check if there are any firewall or network restrictions")
        print("4. Try refreshing the browser page")
    else:
        print("✗ WebSocket test failed. Check server logs for WebSocket-related errors.")
        print("\nCommon WebSocket issues:")
        print("1. CORS configuration in FastAPI server")
        print("2. WebSocket endpoint not properly configured")
        print("3. Session not found or expired")
        print("4. Network connectivity issues")

if __name__ == "__main__":
    main()
