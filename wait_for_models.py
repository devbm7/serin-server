#!/usr/bin/env python3
"""
Script to wait for models to be loaded before considering the server ready
"""

import time
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def wait_for_models_ready(url="http://localhost:8000", max_wait_time=300):
    """Wait for models to be loaded."""
    logger.info(f"Waiting for models to be ready at {url}")
    
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        try:
            response = requests.get(f"{url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("all_models_ready", False):
                    logger.info("All models are ready!")
                    return True
                else:
                    models_status = data.get("models_ready", {})
                    logger.info(f"Models status: {models_status}")
            else:
                logger.warning(f"Health check returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Health check failed: {e}")
        
        logger.info("Waiting 10 seconds before next check...")
        time.sleep(10)
    
    logger.error(f"Timeout waiting for models after {max_wait_time} seconds")
    return False

if __name__ == "__main__":
    import sys
    
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    max_wait = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    
    success = wait_for_models_ready(url, max_wait)
    sys.exit(0 if success else 1)
