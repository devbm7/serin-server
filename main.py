"""
Startup script for the Interview Agent FastAPI Server
"""

import os
import sys
import uvicorn
import logging
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('fastapi_server.log')
        ]
    )

def check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []
    
    try:
        import fastapi
    except ImportError:
        missing_deps.append("fastapi")
    
    try:
        import uvicorn
    except ImportError:
        missing_deps.append("uvicorn")
    
    try:
        import pydantic
    except ImportError:
        missing_deps.append("pydantic")
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import transformers
    except ImportError:
        missing_deps.append("transformers")
    
    try:
        import pyaudio
    except ImportError:
        missing_deps.append("pyaudio")
    
    if missing_deps:
        print(f"Missing dependencies: {', '.join(missing_deps)}")
        print("Please install them using: pip install -r requirements_fastapi.txt")
        return False
    
    return True

def check_config_file():
    """Check if configuration file exists"""
    config_file = Path("pipeline_config.yaml")
    if not config_file.exists():
        print("Warning: pipeline_config.yaml not found. Using default configuration.")
        return False
    return True

def check_supabase_config():
    """Check Supabase configuration"""
    try:
        from supabase_config import supabase_config
        
        if supabase_config.client:
            print("✅ Supabase storage is available for recordings and resumes")
            return True
        else:
            print("⚠️  Supabase storage is not available - recordings will be saved locally")
            print("   To enable cloud storage, set up your Supabase credentials in .env file")
            return False
    except ImportError:
        print("⚠️  Supabase configuration not found - recordings will be saved locally")
        print("   Install supabase package: pip install supabase")
        return False
    except Exception as e:
        print(f"⚠️  Supabase configuration error: {e}")
        return False

def main():
    """Main startup function"""
    print("=== Interview Agent FastAPI Server ===")
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Check configuration
    print("Checking configuration...")
    check_config_file()
    check_supabase_config()
    
    # Get server configuration
    host = os.getenv("FASTAPI_HOST", "0.0.0.0")
    port = int(os.getenv("FASTAPI_PORT", "8000"))
    reload = os.getenv("FASTAPI_RELOAD", "false").lower() == "true"
    
    print(f"Starting server on {host}:{port}")
    print(f"Auto-reload: {reload}")
    print("Press Ctrl+C to stop the server")
    print()
    
    try:
        # Import and run the FastAPI app
        from fastapi_pipeline import app
        
        uvicorn.run(
            "fastapi_pipeline:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 