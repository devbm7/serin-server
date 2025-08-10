import requests

# The public IP address of your VM instance
# We assume the Docker container is mapped to port 80 on the VM.
BASE_URL = "http://34.142.232.217"

def create_session_on_vm():
    """Sends a POST request to the server running on the VM."""
    print(f"--- Attempting to create a session on {BASE_URL} ---")  
    
    create_url = f"{BASE_URL}/sessions/create"
    
    session_payload = {
        "job_role": "Senior Python Developer",
        "user_id": "vm-test-user-002",
        "resume_url": "vm-test-user-002/resume.pdf",
        "asr_model": "openai/whisper-medium",
        "llm_provider": "gemini",
        "llm_model": "gemini-1.5-flash"
    }
    
    try:
        response = requests.post(create_url, json=session_payload, timeout=20)
        response.raise_for_status()
        
        response_data = response.json()
        session_id = response_data.get("session_id")
        
        if session_id:
            print("\n✅ Session created successfully on VM!")
            print(f"   Session ID: {session_id}")
            return session_id
        else:
            print(f"\n❌ Failed to get session_id from response.")
            print(f"   Response: {response.text}")
            return None
            
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

if __name__ == "__main__":
    create_session_on_vm()
