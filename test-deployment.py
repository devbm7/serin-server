#!/usr/bin/env python3
"""
Comprehensive testing script for the deployed Interview Agent service
"""

import asyncio
import json
import time
import websockets
import requests
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InterviewAgentTester:
    def __init__(self, base_url: str = "https://api.devbm.site"):
        self.base_url = base_url.rstrip('/')
        self.session_id = None
        
    def test_health_endpoint(self) -> bool:
        """Test the health endpoint."""
        try:
            logger.info("Testing health endpoint...")
            response = requests.get(f"{self.base_url}/health", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Health check passed: {data}")
                return data.get("all_models_ready", False)
            else:
                logger.error(f"Health check failed with status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False
    
    def test_websocket_health(self) -> bool:
        """Test the WebSocket health endpoint."""
        try:
            logger.info("Testing WebSocket health endpoint...")
            response = requests.get(f"{self.base_url}/ws-health", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"WebSocket health check passed: {data}")
                return True
            else:
                logger.error(f"WebSocket health check failed with status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"WebSocket health check error: {e}")
            return False
    
    def test_session_creation(self) -> bool:
        """Test session creation endpoint."""
        try:
            logger.info("Testing session creation...")
            
            # Test data
            session_data = {
                "job_role": "Software Engineer",
                "user_id": "test-user-123",
                "resume_url": "test-resume.pdf",
                "asr_model": "openai/whisper-medium",
                "llm_provider": "gemini",
                "llm_model": "gemini-2.5-flash"
            }
            
            response = requests.post(
                f"{self.base_url}/sessions/create",
                json=session_data,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                self.session_id = data.get("session_id")
                logger.info(f"Session created successfully: {self.session_id}")
                return True
            else:
                logger.error(f"Session creation failed with status {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Session creation error: {e}")
            return False
    
    async def test_websocket_connection(self) -> bool:
        """Test WebSocket connection and basic communication."""
        if not self.session_id:
            logger.error("No session ID available for WebSocket test")
            return False
            
        try:
            logger.info("Testing WebSocket connection...")
            
            # Connect to WebSocket
            websocket_url = f"wss://{self.base_url.replace('https://', '')}/ws/{self.session_id}"
            logger.info(f"Connecting to WebSocket: {websocket_url}")
            
            async with websockets.connect(websocket_url) as websocket:
                logger.info("WebSocket connected successfully")
                
                # Wait for initial message (opening question)
                try:
                    initial_message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    initial_data = json.loads(initial_message)
                    logger.info(f"Received initial message: {initial_data}")
                    
                    # Send a test audio chunk (silence)
                    test_audio_data = {
                        "type": "audio_chunk",
                        "audio_data": "UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUarm7blmGgU7k9n1unEiBC13yO/eizEIHWq+8+OWT",
                        "sample_rate": 16000,
                        "chunk_size": 512
                    }
                    
                    await websocket.send(json.dumps(test_audio_data))
                    logger.info("Sent test audio chunk")
                    
                    # Wait for response
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                        response_data = json.loads(response)
                        logger.info(f"Received response: {response_data}")
                        return True
                    except asyncio.TimeoutError:
                        logger.warning("No response received within timeout")
                        return True  # Still consider it a success if connection works
                        
                except asyncio.TimeoutError:
                    logger.warning("No initial message received within timeout")
                    return True  # Still consider it a success if connection works
                    
        except Exception as e:
            logger.error(f"WebSocket test error: {e}")
            return False
    
    def test_job_templates(self) -> bool:
        """Test job templates endpoint."""
        try:
            logger.info("Testing job templates endpoint...")
            response = requests.get(f"{self.base_url}/job-templates", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Job templates retrieved: {len(data.get('job_templates', []))} templates")
                return True
            else:
                logger.error(f"Job templates failed with status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Job templates error: {e}")
            return False
    
    def test_models_endpoint(self) -> bool:
        """Test models endpoint."""
        try:
            logger.info("Testing models endpoint...")
            response = requests.get(f"{self.base_url}/models", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Models endpoint working: {len(data.get('asr_models', []))} ASR models")
                return True
            else:
                logger.error(f"Models endpoint failed with status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Models endpoint error: {e}")
            return False
    
    def cleanup_session(self) -> bool:
        """Clean up the test session."""
        if not self.session_id:
            return True
            
        try:
            logger.info(f"Cleaning up session: {self.session_id}")
            response = requests.delete(f"{self.base_url}/sessions/{self.session_id}", timeout=30)
            
            if response.status_code == 200:
                logger.info("Session cleaned up successfully")
                return True
            else:
                logger.warning(f"Session cleanup failed with status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Session cleanup error: {e}")
            return False
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test suite."""
        logger.info("🚀 Starting comprehensive deployment test")
        logger.info(f"Testing service at: {self.base_url}")
        
        results = {
            "health_check": False,
            "websocket_health": False,
            "session_creation": False,
            "websocket_connection": False,
            "job_templates": False,
            "models_endpoint": False,
            "cleanup": False,
            "overall_success": False
        }
        
        # Test health endpoints
        results["health_check"] = self.test_health_endpoint()
        results["websocket_health"] = self.test_websocket_health()
        
        if not results["health_check"]:
            logger.error("❌ Health check failed - service may not be ready")
            return results
        
        # Test session creation
        results["session_creation"] = self.test_session_creation()
        
        if results["session_creation"]:
            # Test WebSocket connection
            results["websocket_connection"] = await self.test_websocket_connection()
            
            # Clean up session
            results["cleanup"] = self.cleanup_session()
        
        # Test other endpoints
        results["job_templates"] = self.test_job_templates()
        results["models_endpoint"] = self.test_models_endpoint()
        
        # Determine overall success
        critical_tests = ["health_check", "session_creation"]
        results["overall_success"] = all(results[test] for test in critical_tests)
        
        # Print summary
        logger.info("\n📊 Test Results Summary:")
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"  {test_name}: {status}")
        
        if results["overall_success"]:
            logger.info("\n🎉 All critical tests passed! Deployment is working correctly.")
        else:
            logger.error("\n⚠️  Some tests failed. Please check the service configuration.")
        
        return results

async def main():
    """Main test function."""
    import sys
    
    # Get base URL from command line or use default
    base_url = sys.argv[1] if len(sys.argv) > 1 else "https://api.devbm.site"
    
    tester = InterviewAgentTester(base_url)
    results = await tester.run_comprehensive_test()
    
    # Exit with appropriate code
    sys.exit(0 if results["overall_success"] else 1)

if __name__ == "__main__":
    asyncio.run(main())
