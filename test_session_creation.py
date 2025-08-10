import requests

# The base URL of your deployed Cloud Run service
BASE_URL = "http://35.240.151.5:8000"

def create_session():
    """Sends a POST request to create a new interview session."""
    print("--- Attempting to create a new session ---")
    
    create_url = f"{BASE_URL}/sessions/create"
    
    # IMPORTANT: This data should match a real user and resume in your Supabase storage
    # for the server to process it correctly.
    session_payload = {
        "job_role": "Senior Python Developer",
        "user_id": "test-user-001",
        "resume_url": "test-user-001/resume.pdf", # This must exist in your Supabase bucket
        "asr_model": "openai/whisper-medium",
        "llm_provider": "gemini",
        "llm_model": "gemini-1.5-flash"
    }
    
    try:
        # Set a reasonable timeout for the request
        response = requests.post(create_url, json=session_payload, timeout=30)
        
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()
        
        response_data = response.json()
        session_id = response_data.get("session_id")
        
        if session_id:
            print("\n✅ Session created successfully!")
            print(f"   Session ID: {session_id}")
            return session_id
        else:
            print("\n❌ Failed to get session_id from response.")
            print(f"   Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ An error occurred while creating the session: {e}")
        # If the request completed with an error, show the response from the server
        if e.response is not None:
            print(f"   Status Code: {e.response.status_code}")
            print(f"   Server Response: {e.response.text}")
        return None

# --- Main execution ---
if __name__ == "__main__":
    create_session()
