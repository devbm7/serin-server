#!/usr/bin/env python3
"""
Test script to verify signed URL approach for resume downloading
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from supabase_config import supabase_config

def test_signed_url():
    """Test the signed URL approach for downloading resumes"""
    
    print("=== Testing Signed URL Approach ===")
    
    # Test with a known resume path
    test_path = "23a0b603-e437-42d6-b1e0-6e0a1b983150/resume_1754526365292.pdf"
    
    print(f"Testing with path: {test_path}")
    
    try:
        # Test creating a signed URL
        print("\n1. Testing signed URL creation...")
        signed_url_response = supabase_config.client.storage.from_("resumes").create_signed_url(test_path, 60)
        
        if signed_url_response:
            # Extract the actual URL from the response
            signed_url = signed_url_response.get('signedURL') or signed_url_response.get('signedUrl')
            
            if signed_url:
                print(f"✅ Signed URL created successfully")
                print(f"URL: {signed_url}")
                
                # Test downloading using the signed URL
                print("\n2. Testing download using signed URL...")
                import requests
                response = requests.get(signed_url)
            else:
                print(f"❌ Could not extract signed URL from response: {signed_url_response}")
                return
            
            if response.status_code == 200:
                print(f"✅ Download successful")
                print(f"File size: {len(response.content)} bytes")
                print(f"Content type: {response.headers.get('content-type', 'unknown')}")
            else:
                print(f"❌ Download failed: {response.status_code}")
                print(f"Response: {response.text}")
        else:
            print("❌ Failed to create signed URL")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_signed_url()
