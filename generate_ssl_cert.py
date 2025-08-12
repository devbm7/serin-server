#!/usr/bin/env python3
"""
Generate a self-signed SSL certificate for testing HTTPS on the FastAPI server
"""

import os
import subprocess
import sys
from pathlib import Path

def generate_self_signed_cert():
    """Generate a self-signed SSL certificate"""
    
    # Create certificates directory if it doesn't exist
    cert_dir = Path("certificates")
    cert_dir.mkdir(exist_ok=True)
    
    key_file = cert_dir / "server.key"
    cert_file = cert_dir / "server.crt"
    
    print("Generating self-signed SSL certificate...")
    print(f"Key file: {key_file}")
    print(f"Cert file: {cert_file}")
    
    # Check if OpenSSL is available
    try:
        subprocess.run(["openssl", "version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ OpenSSL is not installed or not available in PATH")
        print("Please install OpenSSL and try again")
        return False
    
    # Generate private key
    print("Generating private key...")
    try:
        subprocess.run([
            "openssl", "genrsa", "-out", str(key_file), "2048"
        ], check=True, capture_output=True)
        print("✅ Private key generated")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to generate private key: {e}")
        return False
    
    # Generate certificate
    print("Generating certificate...")
    try:
        subprocess.run([
            "openssl", "req", "-new", "-x509", "-key", str(key_file),
            "-out", str(cert_file), "-days", "365", "-subj",
            "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        ], check=True, capture_output=True)
        print("✅ Certificate generated")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to generate certificate: {e}")
        return False
    
    # Set proper permissions
    try:
        os.chmod(key_file, 0o600)
        os.chmod(cert_file, 0o644)
        print("✅ Set proper file permissions")
    except Exception as e:
        print(f"⚠️  Warning: Could not set file permissions: {e}")
    
    print("\n🎉 SSL certificate generated successfully!")
    print("\nTo use HTTPS with your FastAPI server:")
    print("1. Set environment variables:")
    print(f"   export SSL_KEYFILE={key_file}")
    print(f"   export SSL_CERTFILE={cert_file}")
    print("2. Start your server:")
    print("   python start_fastapi_server.py")
    print("\n⚠️  Note: This is a self-signed certificate for testing only.")
    print("   For production, use a proper SSL certificate from a trusted CA.")
    
    return True

if __name__ == "__main__":
    if generate_self_signed_cert():
        sys.exit(0)
    else:
        sys.exit(1)
