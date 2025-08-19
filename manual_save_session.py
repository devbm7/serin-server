#!/usr/bin/env python3
"""
Manual script to save a specific missed session using local recording file
"""

import requests
import json
import os
from datetime import datetime
from pathlib import Path

# FastAPI server URL
BASE_URL = "http://localhost:8000"

def manual_save_session():
    """Manually save the specific missed session."""
    
    session_id = "62b0266a-6498-4657-88aa-40366d7a8738"
    
    print(f"Manually saving session: {session_id}")
    
    # Check if recording file exists locally
    recording_dir = Path("recordings") / session_id
    if not recording_dir.exists():
        print(f"❌ Recording directory not found: {recording_dir}")
        return False
    
    # Find the recording file
    recording_files = list(recording_dir.glob("*.webm"))
    if not recording_files:
        print(f"❌ No recording files found in {recording_dir}")
        return False
    
    recording_file = recording_files[0]
    print(f"✅ Found recording file: {recording_file}")
    
    # Get file info
    file_size = recording_file.stat().st_size
    file_time = datetime.fromtimestamp(recording_file.stat().st_mtime)
    
    print(f"   - File size: {file_size} bytes")
    print(f"   - File time: {file_time}")
    
    # Create the session data manually
    session_data = {
        "interviewee_id": None,
        "template_id": None,
        "start_time": file_time,  # Use file creation time as start time
        "status": "completed",
        "session_information": {
            "job_role": "Public Relations Manager",  # From your example
            "conversation_history": [
                {
                    "role": "assistant",
                    "content": "Hello, I'm Serin. Given your background in artificial intelligence and machine learning, how do you envision those skills contributing to the role of a Public Relations Manager?"
                }
            ],
            "anomalies_detected": [],
            "asr_model": "openai/whisper-medium",
            "llm_provider": "gemini",
            "llm_model": "gemini-2.5-flash"
        },
        "resume_url": "https://ibnsjeoemngngkqnnjdz.supabase.co/storage/v1/object/public/resumes/23a0b603-e437-42d6-b1e0-6e0a1b983150/resume_1754526365292.pdf",
        "recording_url": f"https://ibnsjeoemngngkqnnjdz.supabase.co/storage/v1/object/public/interview-recordings/{session_id}/{recording_file.name}"
    }
    
    print(f"\nPrepared session data:")
    print(f"   - Start time: {session_data['start_time']}")
    print(f"   - Job role: {session_data['session_information']['job_role']}")
    print(f"   - Resume URL: {session_data['resume_url']}")
    print(f"   - Recording URL: {session_data['recording_url']}")
    print(f"   - Conversation messages: {len(session_data['session_information']['conversation_history'])}")
    
    # Save to database using direct Supabase call
    try:
        from supabase_config import supabase_config
        
        if not supabase_config.client:
            print("❌ Supabase client not available")
            return False
        
        # Insert the session data
        response = supabase_config.client.table("interview_sessions").insert(session_data).execute()
        
        if response.data and len(response.data) > 0:
            saved_session_id = response.data[0].get("session_id")
            print(f"✅ Session saved successfully with ID: {saved_session_id}")
            return True
        else:
            print("❌ Failed to save session: no data returned")
            return False
            
    except Exception as e:
        print(f"❌ Error saving session: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("MANUAL SESSION SAVE")
    print("="*60)
    
    success = manual_save_session()
    
    if success:
        print("\n🎉 Session saved successfully!")
    else:
        print("\n❌ Failed to save session")
    
    print("\n" + "="*60)
