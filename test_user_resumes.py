#!/usr/bin/env python3
"""
Test script to verify the updated user resumes endpoint
"""

import requests
import json

def test_user_resumes_endpoint():
    """Test the updated user resumes endpoint"""
    
    base_url = "http://localhost:8000"
    test_user_id = "23a0b603-e437-42d6-b1e0-6e0a1b983150"
    
    print("=== Testing Updated User Resumes Endpoint ===")
    
    try:
        print(f"🔄 Fetching resumes for user: {test_user_id}")
        response = requests.get(f"{base_url}/user-resumes/{test_user_id}")
        
        if response.status_code == 200:
            data = response.json()
            user_resumes = data.get("user_resumes", [])
            print(f"✅ User resumes fetched successfully: {len(user_resumes)} resumes found")
            
            for resume in user_resumes:
                print(f"   - {resume.get('first_name')} {resume.get('last_name')} ({resume.get('resume_filename')})")
                print(f"     User ID: {resume.get('user_id')}")
                print(f"     Resume URL: {resume.get('resume_url')}")
        else:
            print(f"❌ Failed to fetch user resumes: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_user_resumes_endpoint()
