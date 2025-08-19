#!/usr/bin/env python3
"""
Test script to verify the new API endpoints for job templates and user resumes
"""

import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_new_api_endpoints():
    """Test the new API endpoints for job templates and user resumes"""
    
    base_url = "http://localhost:8000"
    
    print("=== Testing New API Endpoints ===")
    
    # Test 1: Check if server is running
    try:
        response = requests.get(f"{base_url}/models")
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print(f"❌ Server returned status code: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure the FastAPI server is running.")
        return
    
    # Test 2: Get job templates
    try:
        print("\n🔄 Fetching job templates...")
        response = requests.get(f"{base_url}/job-templates")
        
        if response.status_code == 200:
            data = response.json()
            job_templates = data.get("job_templates", [])
            print(f"✅ Job templates fetched successfully: {len(job_templates)} templates found")
            
            for template in job_templates:
                print(f"   - {template.get('template_name')} ({template.get('job_role')})")
        else:
            print(f"❌ Failed to fetch job templates: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error fetching job templates: {e}")
    
    # Test 3: Get user resumes for specific user
    try:
        print("\n🔄 Fetching user resumes for specific user...")
        test_user_id = "23a0b603-e437-42d6-b1e0-6e0a1b983150"
        response = requests.get(f"{base_url}/user-resumes/{test_user_id}")
        
        if response.status_code == 200:
            data = response.json()
            user_resumes = data.get("user_resumes", [])
            print(f"✅ User resumes fetched successfully: {len(user_resumes)} resumes found")
            
            for resume in user_resumes:
                print(f"   - {resume.get('first_name')} {resume.get('last_name')} ({resume.get('resume_filename')})")
        else:
            print(f"❌ Failed to fetch user resumes: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error fetching user resumes: {e}")
    
    # Test 4: Create session with new format
    job_templates = job_templates if 'job_templates' in locals() else []
    user_resumes = user_resumes if 'user_resumes' in locals() else []
    
    if job_templates and user_resumes:
        try:
            print("\n🔄 Testing session creation with new format...")
            
            # Use the first job template and user resume
            job_template = job_templates[0]
            user_resume = user_resumes[0]
            
            session_data = {
                "job_role": job_template.get("job_role"),
                "user_id": user_resume.get("user_id"),
                "resume_url": user_resume.get("resume_url"),
                "llm_provider": "gemini",
                "llm_model": "gemini-2.5-flash"
            }
            
            print(f"Creating session with:")
            print(f"   - Job Role: {session_data['job_role']}")
            print(f"   - User ID: {session_data['user_id']}")
            print(f"   - Resume URL: {session_data['resume_url']}")
            
            response = requests.post(f"{base_url}/sessions/create", json=session_data)
            
            if response.status_code == 200:
                result = response.json()
                session_id = result.get("session_id")
                print(f"✅ Session created successfully: {session_id}")
                
                # Get session info
                response = requests.get(f"{base_url}/sessions/{session_id}")
                if response.status_code == 200:
                    session_info = response.json()
                    print(f"✅ Session info: {session_info}")
                
                # Clean up
                requests.delete(f"{base_url}/sessions/{session_id}")
                print("✅ Session cleaned up")
                
            else:
                print(f"❌ Failed to create session: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error during session creation test: {e}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_new_api_endpoints()
