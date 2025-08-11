import requests
import json

# The public IP address of your VM instance
# We assume the Docker container is mapped to port 80 on the VM.
BASE_URL = "http://35.240.151.5:8000"

def test_job_templates():
    """Test the job-templates endpoint."""
    print(f"--- Testing job-templates endpoint on {BASE_URL} ---")  
    
    create_url = f"{BASE_URL}/job-templates"
    
    try:
        response = requests.get(create_url, timeout=20)  # Changed to GET request for job-templates
        response.raise_for_status()
        
        response_data = response.json()
        
        print("\n✅ Job templates retrieved successfully from VM!")
        print(f"   Response: {response_data}")
        return response_data
            
    except requests.exceptions.Timeout:
        print("\n❌ Error: The request timed out.")
        print("   This usually means a firewall is blocking the connection.")
        print("   Ensure you have a GCP firewall rule to allow TCP traffic on port 80.")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"\n❌ Error: Could not connect to the server: {e}")
        print("   This could mean the Docker container isn't running, or it's not mapped to port 80.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"\n❌ An application error occurred: {e}")
        if e.response is not None:
            print(f"   Status Code: {e.response.status_code}")
            print(f"   Server Response: {e.response.text}")
        return None

def create_session_on_vm():
    """Creates a new interview session on the VM."""
    print(f"--- Creating session on {BASE_URL} ---")  
    
    create_url = f"{BASE_URL}/sessions/create"
    
    # Sample session creation data
    session_data = {
        "job_role": "Software Engineer",
        "user_id": "test-user-123",
        "resume_url": "test-resume.pdf",
        "asr_model": "openai/whisper-medium",
        "llm_provider": "gemini",
        "llm_model": "gemini-2.5-flash"
    }
    
    try:
        response = requests.post(create_url, json=session_data, timeout=20)
        response.raise_for_status()
        
        response_data = response.json()
        session_id = response_data.get("session_id")
        
        if session_id:
            print("\n✅ Session created successfully on VM!")
            print(f"   Session ID: {session_id}")
            return session_id
        else:
            print(f"\n❌ Failed to get session ID from response.")
            print(f"   Response: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("\n❌ Error: The request timed out.")
        print("   This usually means a firewall is blocking the connection.")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"\n❌ Error: Could not connect to the server: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"\n❌ An application error occurred: {e}")
        if e.response is not None:
            print(f"   Status Code: {e.response.status_code}")
            print(f"   Server Response: {e.response.text}")
        return None

if __name__ == "__main__":
    # Test job templates first
    test_job_templates()
    
    # Then test session creation
    create_session_on_vm()
