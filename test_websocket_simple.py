#!/usr/bin/env python3
"""
Simple WebSocket test for debugging connection issues
"""

import asyncio
import json
import requests
import websockets
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
BASE_URL = "http://34.142.208.17:8000"
WS_URL = "ws://34.142.208.17:8000"

async def test_basic_websocket():
    """Test basic WebSocket connection without session."""
    logger.info("Testing basic WebSocket connection...")
    
    try:
        # Try to connect to a non-existent session
        ws_url = f"{WS_URL}/ws/test-session-123"
        logger.info(f"Connecting to: {ws_url}")
        
        websocket = await websockets.connect(ws_url)
        logger.info("✅ WebSocket connection established")
        
        # Wait for error message
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            response_data = json.loads(response)
            logger.info(f"✅ Received response: {response_data}")
        except asyncio.TimeoutError:
            logger.warning("⚠️ No response received (timeout)")
        
        await websocket.close()
        logger.info("✅ WebSocket connection closed")
        return True
        
    except Exception as e:
        logger.error(f"❌ WebSocket connection failed: {e}")
        return False

def test_session_creation():
    """Test session creation."""
    logger.info("Testing session creation...")
    
    session_data = {
        "job_role": "Software Engineer",
        "user_id": "test-user-123",
        "resume_url": "test-resume.pdf",
        "asr_model": "openai/whisper-medium",
        "llm_provider": "gemini",
        "llm_model": "gemini-2.5-flash"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/sessions/create", json=session_data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        session_id = result.get("session_id")
        
        if session_id:
            logger.info(f"✅ Session created successfully: {session_id}")
            return session_id
        else:
            logger.error("❌ Failed to get session ID")
            logger.error(f"Response: {result}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Failed to create session: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Status code: {e.response.status_code}")
            logger.error(f"Response: {e.response.text}")
        return None

async def test_websocket_with_session(session_id: str):
    """Test WebSocket connection with a valid session."""
    logger.info(f"Testing WebSocket connection with session: {session_id}")
    
    try:
        ws_url = f"{WS_URL}/ws/{session_id}"
        logger.info(f"Connecting to: {ws_url}")
        
        websocket = await websockets.connect(ws_url)
        logger.info("✅ WebSocket connection established")
        
        # Send a ping message
        ping_message = json.dumps({"type": "ping", "timestamp": 1234567890})
        await websocket.send(ping_message)
        logger.info("✅ Ping message sent")
        
        # Wait for response
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            response_data = json.loads(response)
            logger.info(f"✅ Response received: {response_data}")
            
            if response_data.get("type") == "pong":
                logger.info("✅ Pong response received!")
            elif "error" in response_data:
                logger.warning(f"⚠️ Error response: {response_data['error']}")
            else:
                logger.info(f"✅ Other response: {response_data}")
                
        except asyncio.TimeoutError:
            logger.warning("⚠️ No response received (timeout)")
        
        await websocket.close()
        logger.info("✅ WebSocket connection closed")
        return True
        
    except Exception as e:
        logger.error(f"❌ WebSocket connection failed: {e}")
        return False

def test_http_endpoints():
    """Test basic HTTP endpoints."""
    logger.info("Testing HTTP endpoints...")
    
    endpoints = [
        ("/", "GET"),
        ("/docs", "GET"),
        ("/job-templates", "GET"),
        ("/models", "GET"),
    ]
    
    for endpoint, method in endpoints:
        try:
            if method == "GET":
                response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
            else:
                response = requests.post(f"{BASE_URL}{endpoint}", timeout=10)
            
            logger.info(f"{method} {endpoint}: {response.status_code}")
            
            if response.status_code == 200:
                logger.info(f"✅ {method} {endpoint} - OK")
            elif response.status_code == 404:
                logger.warning(f"⚠️ {method} {endpoint} - Not Found (expected for some endpoints)")
            else:
                logger.warning(f"⚠️ {method} {endpoint} - Status {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ {method} {endpoint} - Error: {e}")

async def main():
    """Run all tests."""
    logger.info("="*50)
    logger.info("SIMPLE WEBSOCKET TEST")
    logger.info("="*50)
    
    # Test 1: HTTP endpoints
    test_http_endpoints()
    
    # Test 2: Basic WebSocket (should fail with session error)
    await test_basic_websocket()
    
    # Test 3: Create session
    session_id = test_session_creation()
    
    if session_id:
        # Test 4: WebSocket with valid session
        await test_websocket_with_session(session_id)
        
        # Clean up session
        try:
            response = requests.post(f"{BASE_URL}/sessions/{session_id}/end", timeout=10)
            if response.status_code == 200:
                logger.info("✅ Session ended successfully")
            else:
                logger.warning(f"⚠️ Failed to end session: {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ Error ending session: {e}")
    
    logger.info("="*50)
    logger.info("TEST COMPLETED")
    logger.info("="*50)

if __name__ == "__main__":
    asyncio.run(main())
