#!/usr/bin/env python3
"""
Script to manually update a session with its recording URL
"""

import requests
import json
from pathlib import Path

# FastAPI server URL
BASE_URL = "http://localhost:8000"

def update_session_with_recording():
    """Update the specific session with its recording URL."""
    
    session_id = "5055e335-c758-4a89-bfb3-13717a9ad43e"
    
    print(f"Updating session with recording URL: {session_id}")
    
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
    
    # Construct the Supabase URL
    recording_url = f"https://ibnsjeoemngngkqnnjdz.supabase.co/storage/v1/object/public/interview-recordings/{session_id}/{recording_file.name}"
    
    print(f"Recording URL: {recording_url}")
    
    # Update the session using direct Supabase call
    try:
        from supabase_config import supabase_config
        
        if not supabase_config.client:
            print("❌ Supabase client not available")
            return False
        
        # Update the session with recording URL
        success = supabase_config.update_session_recording_url(session_id, recording_url)
        
        if success:
            print(f"✅ Session {session_id} updated with recording URL successfully")
            return True
        else:
            print(f"❌ Failed to update session {session_id} with recording URL")
            return False
            
    except Exception as e:
        print(f"❌ Error updating session: {e}")
        return False

def check_session_in_database():
    """Check if the session exists in the database."""
    
    session_id = "5055e335-c758-4a89-bfb3-13717a9ad43e"
    
    try:
        from supabase_config import supabase_config
        
        if not supabase_config.client:
            print("❌ Supabase client not available")
            return False
        
        # Query the session
        response = supabase_config.client.table("interview_sessions").select("*").eq("session_id", session_id).execute()
        
        if response.data and len(response.data) > 0:
            session_data = response.data[0]
            print(f"✅ Session found in database:")
            print(f"   - Session ID: {session_data.get('session_id')}")
            print(f"   - Start Time: {session_data.get('start_time')}")
            print(f"   - Status: {session_data.get('status')}")
            print(f"   - Resume URL: {session_data.get('resume_url')}")
            print(f"   - Recording URL: {session_data.get('recording_url')}")
            return True
        else:
            print(f"❌ Session {session_id} not found in database")
            return False
            
    except Exception as e:
        print(f"❌ Error checking session: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("UPDATE SESSION WITH RECORDING URL")
    print("="*60)
    
    # First check if session exists
    print("\n1. Checking if session exists in database...")
    session_exists = check_session_in_database()
    
    if session_exists:
        print("\n2. Updating session with recording URL...")
        success = update_session_with_recording()
        
        if success:
            print("\n🎉 Session updated successfully!")
            
            # Check the updated session
            print("\n3. Verifying update...")
            check_session_in_database()
        else:
            print("\n❌ Failed to update session")
    else:
        print("\n❌ Session not found in database, cannot update")
    
    print("\n" + "="*60)
