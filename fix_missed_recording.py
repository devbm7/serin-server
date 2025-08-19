#!/usr/bin/env python3
"""
Script to manually fix the missed recording URL for a specific session
"""

import requests
import json
from pathlib import Path
from datetime import datetime, timedelta

def fix_missed_recording():
    """Fix the missed recording URL for the specific session."""
    
    session_id = "f8cff030-65b0-48d0-972c-62be70a8bc63"
    
    print(f"Fixing missed recording URL for session: {session_id}")
    
    # Check if recording file exists locally (use poc/recordings path)
    recording_dir = Path("poc/recordings") / session_id
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
    
    # Update the session using the new method
    try:
        from supabase_config import supabase_config
        
        if not supabase_config.client:
            print("❌ Supabase client not available")
            return False
        
        # Use the new method to find and update the session
        success = supabase_config.update_session_recording_url_by_original_id(session_id, recording_url)
        
        if success:
            print(f"✅ Session updated with recording URL successfully")
            return True
        else:
            print(f"❌ Failed to update session with recording URL")
            return False
            
    except Exception as e:
        print(f"❌ Error updating session: {e}")
        return False

def check_recent_sessions():
    """Check recent sessions in the database."""
    
    try:
        from supabase_config import supabase_config
        
        if not supabase_config.client:
            print("❌ Supabase client not available")
            return False
        
        # Get sessions created in the last hour
        one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
        
        response = supabase_config.client.table("interview_sessions").select("*").gte("start_time", one_hour_ago).execute()
        
        if response.data:
            print(f"✅ Found {len(response.data)} recent sessions:")
            for session in response.data:
                print(f"   - Session ID: {session.get('session_id')}")
                print(f"     Start Time: {session.get('start_time')}")
                print(f"     Recording URL: {session.get('recording_url')}")
                print(f"     Resume URL: {session.get('resume_url')}")
                print()
            return True
        else:
            print("❌ No recent sessions found")
            return False
            
    except Exception as e:
        print(f"❌ Error checking sessions: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("FIX MISSED RECORDING URL")
    print("="*60)
    
    # First check recent sessions
    print("\n1. Checking recent sessions in database...")
    check_recent_sessions()
    
    print("\n2. Fixing missed recording URL...")
    success = fix_missed_recording()
    
    if success:
        print("\n🎉 Recording URL fixed successfully!")
        
        # Check the updated sessions
        print("\n3. Verifying fix...")
        check_recent_sessions()
    else:
        print("\n❌ Failed to fix recording URL")
    
    print("\n" + "="*60)
