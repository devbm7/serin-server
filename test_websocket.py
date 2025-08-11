#!/usr/bin/env python3
"""
Comprehensive WebSocket test for the FastAPI Interview Agent
Tests session creation, WebSocket connection, audio streaming, and response handling
"""

import asyncio
import base64
import json
import time
import websockets
import requests
import numpy as np
import threading
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
BASE_URL = "http://34.142.208.17:8000"  # Change this to your server URL
WS_URL = "ws://34.142.208.17:8000"      # WebSocket URL

class WebSocketTester:
    def __init__(self, base_url: str, ws_url: str):
        self.base_url = base_url
        self.ws_url = ws_url
        self.session_id = None
        self.websocket = None
        self.test_results = {}
        
    def create_test_session(self) -> Optional[str]:
        """Create a test session for WebSocket testing."""
        logger.info("Creating test session...")
        
        session_data = {
            "job_role": "Software Engineer",
            "user_id": "test-user-123",
            "resume_url": "test-resume.pdf",
            "asr_model": "openai/whisper-medium",
            "llm_provider": "gemini",
            "llm_model": "gemini-2.5-flash"
        }
        
        try:
            response = requests.post(f"{self.base_url}/sessions/create", json=session_data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            session_id = result.get("session_id")
            
            if session_id:
                logger.info(f"✅ Session created successfully: {session_id}")
                self.session_id = session_id
                return session_id
            else:
                logger.error("❌ Failed to get session ID from response")
                logger.error(f"Response: {result}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to create session: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                logger.error(f"Response: {e.response.text}")
            return None
    
    def generate_test_audio(self, duration_ms: int = 1000, sample_rate: int = 16000) -> bytes:
        """Generate test audio data (sine wave)."""
        # Generate a 440Hz sine wave
        frequency = 440  # Hz
        duration = duration_ms / 1000.0  # seconds
        samples = int(duration * sample_rate)
        
        # Generate sine wave
        t = np.linspace(0, duration, samples, False)
        audio_data = np.sin(2 * np.pi * frequency * t)
        
        # Convert to 16-bit PCM
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        return audio_int16.tobytes()
    
    def create_audio_chunk_message(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """Create an audio chunk message for WebSocket."""
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        message = {
            "type": "audio_chunk",
            "audio_data": audio_base64,
            "sample_rate": sample_rate,
            "chunk_size": len(audio_data)
        }
        
        return json.dumps(message)
    
    async def test_websocket_connection(self) -> bool:
        """Test basic WebSocket connection."""
        if not self.session_id:
            logger.error("❌ No session ID available for WebSocket test")
            return False
        
        logger.info(f"Testing WebSocket connection for session: {self.session_id}")
        
        try:
            # Connect to WebSocket
            ws_url = f"{self.ws_url}/ws/{self.session_id}"
            logger.info(f"Connecting to: {ws_url}")
            
            self.websocket = await websockets.connect(ws_url)
            logger.info("✅ WebSocket connection established")
            
            # Test ping/pong
            ping_message = json.dumps({"type": "ping", "timestamp": time.time()})
            await self.websocket.send(ping_message)
            logger.info("✅ Ping message sent")
            
            # Wait for pong response
            try:
                response = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
                response_data = json.loads(response)
                
                if response_data.get("type") == "pong":
                    logger.info("✅ Pong response received")
                    self.test_results["ping_pong"] = True
                else:
                    logger.info(f"Received response: {response_data}")
                    self.test_results["ping_pong"] = True
                    
            except asyncio.TimeoutError:
                logger.warning("⚠️ No pong response received (timeout)")
                self.test_results["ping_pong"] = False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ WebSocket connection failed: {e}")
            self.test_results["websocket_connection"] = False
            return False
    
    async def test_audio_streaming(self, num_chunks: int = 5) -> bool:
        """Test audio streaming through WebSocket."""
        if not self.websocket:
            logger.error("❌ No WebSocket connection available")
            return False
        
        logger.info(f"Testing audio streaming with {num_chunks} chunks...")
        
        try:
            for i in range(num_chunks):
                # Generate test audio
                audio_data = self.generate_test_audio(duration_ms=500)  # 500ms chunks
                
                # Create audio chunk message
                message = self.create_audio_chunk_message(audio_data)
                
                # Send audio chunk
                await self.websocket.send(message)
                logger.info(f"✅ Audio chunk {i+1}/{num_chunks} sent ({len(audio_data)} bytes)")
                
                # Wait a bit between chunks
                await asyncio.sleep(0.1)
            
            # Wait for any responses
            logger.info("Waiting for responses...")
            responses_received = 0
            
            for _ in range(10):  # Check for responses for up to 10 iterations
                try:
                    response = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
                    response_data = json.loads(response)
                    
                    if "error" in response_data:
                        logger.error(f"❌ Error response: {response_data['error']}")
                        self.test_results["audio_streaming"] = False
                        return False
                    
                    logger.info(f"✅ Response received: {response_data.get('type', 'unknown')}")
                    responses_received += 1
                    
                    # Check if it's an interview response
                    if "transcription" in response_data or "response" in response_data:
                        logger.info("✅ Interview response received!")
                        self.test_results["interview_response"] = True
                    
                except asyncio.TimeoutError:
                    break  # No more responses
            
            logger.info(f"Audio streaming test completed. Responses received: {responses_received}")
            self.test_results["audio_streaming"] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Audio streaming test failed: {e}")
            self.test_results["audio_streaming"] = False
            return False
    
    async def test_large_audio_chunk(self) -> bool:
        """Test handling of large audio chunks."""
        if not self.websocket:
            logger.error("❌ No WebSocket connection available")
            return False
        
        logger.info("Testing large audio chunk handling...")
        
        try:
            # Generate a large audio chunk (>1MB)
            large_audio = self.generate_test_audio(duration_ms=30000)  # 30 seconds
            large_audio = large_audio * 10  # Make it even larger
            
            message = self.create_audio_chunk_message(large_audio)
            
            # Send large chunk
            await self.websocket.send(message)
            logger.info(f"✅ Large audio chunk sent ({len(large_audio)} bytes)")
            
            # Wait for response
            try:
                response = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
                response_data = json.loads(response)
                
                if "error" in response_data and "too large" in response_data["error"].lower():
                    logger.info("✅ Large audio chunk properly rejected")
                    self.test_results["large_audio_handling"] = True
                    return True
                else:
                    logger.warning(f"⚠️ Unexpected response to large audio: {response_data}")
                    self.test_results["large_audio_handling"] = False
                    return False
                    
            except asyncio.TimeoutError:
                logger.warning("⚠️ No response to large audio chunk (timeout)")
                self.test_results["large_audio_handling"] = False
                return False
                
        except Exception as e:
            logger.error(f"❌ Large audio chunk test failed: {e}")
            self.test_results["large_audio_handling"] = False
            return False
    
    async def test_invalid_messages(self) -> bool:
        """Test handling of invalid messages."""
        if not self.websocket:
            logger.error("❌ No WebSocket connection available")
            return False
        
        logger.info("Testing invalid message handling...")
        
        invalid_tests = [
            ("Invalid JSON", "invalid json string"),
            ("Missing type", '{"audio_data": "test"}'),
            ("Invalid audio data", '{"type": "audio_chunk", "audio_data": "invalid_base64", "sample_rate": 16000}'),
            ("Empty message", ""),
        ]
        
        results = []
        
        for test_name, message in invalid_tests:
            try:
                await self.websocket.send(message)
                logger.info(f"✅ {test_name} sent")
                
                # Wait for error response
                try:
                    response = await asyncio.wait_for(self.websocket.recv(), timeout=3.0)
                    response_data = json.loads(response)
                    
                    if "error" in response_data:
                        logger.info(f"✅ {test_name} properly handled with error: {response_data['error']}")
                        results.append(True)
                    else:
                        logger.warning(f"⚠️ {test_name} unexpected response: {response_data}")
                        results.append(False)
                        
                except asyncio.TimeoutError:
                    logger.warning(f"⚠️ {test_name} no response received")
                    results.append(False)
                    
            except Exception as e:
                logger.error(f"❌ {test_name} failed: {e}")
                results.append(False)
        
        success_rate = sum(results) / len(results) if results else 0
        logger.info(f"Invalid message handling success rate: {success_rate:.2%}")
        self.test_results["invalid_message_handling"] = success_rate > 0.5
        return success_rate > 0.5
    
    async def test_connection_stability(self) -> bool:
        """Test WebSocket connection stability over time."""
        if not self.websocket:
            logger.error("❌ No WebSocket connection available")
            return False
        
        logger.info("Testing connection stability...")
        
        try:
            # Send periodic pings for 30 seconds
            start_time = time.time()
            pings_sent = 0
            pongs_received = 0
            
            while time.time() - start_time < 30:
                ping_message = json.dumps({"type": "ping", "timestamp": time.time()})
                await self.websocket.send(ping_message)
                pings_sent += 1
                
                try:
                    response = await asyncio.wait_for(self.websocket.recv(), timeout=2.0)
                    response_data = json.loads(response)
                    
                    if response_data.get("type") == "pong":
                        pongs_received += 1
                    
                except asyncio.TimeoutError:
                    pass  # Expected for some pings
                
                await asyncio.sleep(2)  # Send ping every 2 seconds
            
            success_rate = pongs_received / pings_sent if pings_sent > 0 else 0
            logger.info(f"Connection stability: {pings_sent} pings sent, {pongs_received} pongs received ({success_rate:.2%})")
            
            self.test_results["connection_stability"] = success_rate > 0.7
            return success_rate > 0.7
            
        except Exception as e:
            logger.error(f"❌ Connection stability test failed: {e}")
            self.test_results["connection_stability"] = False
            return False
    
    async def cleanup(self):
        """Clean up resources."""
        if self.websocket:
            await self.websocket.close()
            logger.info("✅ WebSocket connection closed")
        
        if self.session_id:
            try:
                # End the session
                response = requests.post(f"{self.base_url}/sessions/{self.session_id}/end", timeout=10)
                if response.status_code == 200:
                    logger.info("✅ Session ended successfully")
                else:
                    logger.warning(f"⚠️ Failed to end session: {response.status_code}")
            except Exception as e:
                logger.warning(f"⚠️ Error ending session: {e}")
    
    def print_test_summary(self):
        """Print a summary of all test results."""
        logger.info("\n" + "="*50)
        logger.info("WEBSOCKET TEST SUMMARY")
        logger.info("="*50)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name:25} : {status}")
        
        passed_tests = sum(1 for result in self.test_results.values() if result)
        total_tests = len(self.test_results)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        logger.info("-"*50)
        logger.info(f"Overall Success Rate: {success_rate:.2%} ({passed_tests}/{total_tests})")
        
        if success_rate >= 0.8:
            logger.info("🎉 WebSocket tests PASSED!")
        elif success_rate >= 0.6:
            logger.info("⚠️ WebSocket tests PARTIALLY PASSED")
        else:
            logger.info("💥 WebSocket tests FAILED")

async def run_websocket_tests():
    """Run all WebSocket tests."""
    tester = WebSocketTester(BASE_URL, WS_URL)
    
    try:
        # Step 1: Create session
        session_id = tester.create_test_session()
        if not session_id:
            logger.error("❌ Cannot proceed without session")
            return
        
        # Step 2: Test WebSocket connection
        if not await tester.test_websocket_connection():
            logger.error("❌ Cannot proceed without WebSocket connection")
            return
        
        # Step 3: Test audio streaming
        await tester.test_audio_streaming()
        
        # Step 4: Test large audio handling
        await tester.test_large_audio_chunk()
        
        # Step 5: Test invalid message handling
        await tester.test_invalid_messages()
        
        # Step 6: Test connection stability
        await tester.test_connection_stability()
        
        # Step 7: Print summary
        tester.print_test_summary()
        
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
    finally:
        await tester.cleanup()

def test_http_endpoints():
    """Test HTTP endpoints before WebSocket tests."""
    logger.info("Testing HTTP endpoints...")
    
    endpoints = [
        ("/job-templates", "GET"),
        ("/models", "GET"),
        ("/docs", "GET"),
    ]
    
    for endpoint, method in endpoints:
        try:
            if method == "GET":
                response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
            else:
                response = requests.post(f"{BASE_URL}{endpoint}", timeout=10)
            
            if response.status_code == 200:
                logger.info(f"✅ {method} {endpoint} - OK")
            else:
                logger.warning(f"⚠️ {method} {endpoint} - Status {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ {method} {endpoint} - Error: {e}")

if __name__ == "__main__":
    logger.info("Starting WebSocket tests...")
    
    # Test HTTP endpoints first
    test_http_endpoints()
    
    # Run WebSocket tests
    asyncio.run(run_websocket_tests())
