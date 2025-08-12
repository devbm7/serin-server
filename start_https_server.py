#!/usr/bin/env python3
"""
Convenience script to start the FastAPI server with HTTPS
"""

import os
import sys
from pathlib import Path

def setup_https_environment():
    """Setup environment variables for HTTPS"""
    
    # Check if certificates exist
    cert_dir = Path("certificates")
    key_file = cert_dir / "server.key"
    cert_file = cert_dir / "server.crt"
    
    if not key_file.exists() or not cert_file.exists():
        print("❌ SSL certificates not found!")
        print("Generating self-signed certificates...")
        
        # Try to generate certificates
        try:
            from generate_ssl_cert import generate_self_signed_cert
            if not generate_self_signed_cert():
                print("❌ Failed to generate SSL certificates")
                return False
        except ImportError:
            print("❌ Could not import generate_ssl_cert.py")
            print("Please run: python generate_ssl_cert.py")
            return False
    
    # Set environment variables
    os.environ["SSL_KEYFILE"] = str(key_file)
    os.environ["SSL_CERTFILE"] = str(cert_file)
    
    print("✅ SSL environment configured")
    print(f"   Key: {key_file}")
    print(f"   Cert: {cert_file}")
    
    return True

def main():
    """Main function"""
    print("=== FastAPI HTTPS Server Startup ===")
    
    # Setup HTTPS environment
    if not setup_https_environment():
        sys.exit(1)
    
    # Import and run the main startup script
    try:
        from start_fastapi_server import main as start_server
        start_server()
    except ImportError:
        print("❌ Could not import start_fastapi_server.py")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
