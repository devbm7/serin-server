#!/usr/bin/env python3
"""
Script to manually save a missed interview session
"""

import requests
import json
from datetime import datetime

# FastAPI server URL
BASE_URL = "http://localhost:8000"

def save_missed_session(session_id: str):
    """Save a missed session to the database."""
    
    print(f"Attempting to save missed session: {session_id}")
    
    # Try the regular save endpoint first
    print("\n1. Trying regular save endpoint...")
    response = requests.post(f"{BASE_URL}/sessions/{session_id}/save")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Session saved successfully: {result['message']}")
        return True
    elif response.status_code == 404:
        print("❌ Session not found in memory, trying missed session endpoint...")
        
        # Try the missed session endpoint
        response = requests.post(f"{BASE_URL}/sessions/{session_id}/save-missed")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Session saved (partial): {result['message']}")
            return True
        else:
            print(f"❌ Failed to save missed session: {response.status_code}")
            print(response.text)
            return False
    else:
        print(f"❌ Failed to save session: {response.status_code}")
        print(response.text)
        return False

def check_session_status(session_id: str):
    """Check if session exists and get its status."""
    
    print(f"\nChecking session status: {session_id}")
    
    # Check if session exists
    response = requests.get(f"{BASE_URL}/sessions/{session_id}")
    
    if response.status_code == 200:
        session_info = response.json()
        print(f"✅ Session exists: {session_info}")
        return True
    elif response.status_code == 404:
        print("❌ Session not found in memory")
        return False
    else:
        print(f"❌ Error checking session: {response.status_code}")
        print(response.text)
        return False

def get_session_info(session_id: str):
    """Get session information."""
    
    print(f"\nGetting session information: {session_id}")
    
    response = requests.get(f"{BASE_URL}/sessions/{session_id}/info")
    
    if response.status_code == 200:
        session_info = response.json()
        print(f"✅ Session info retrieved:")
        print(f"   - Job Role: {session_info['session_information']['job_role']}")
        print(f"   - Conversation History: {len(session_info['session_information']['conversation_history'])} messages")
        print(f"   - Anomalies: {len(session_info['session_information']['anomalies_detected'])} detected")
        return session_info
    else:
        print(f"❌ Failed to get session info: {response.status_code}")
        print(response.text)
        return None

if __name__ == "__main__":
    # The session ID from your logs
    session_id = "62b0266a-6498-4657-88aa-40366d7a8738"
    
    print("="*60)
    print("SAVING MISSED INTERVIEW SESSION")
    print("="*60)
    
    # Check session status
    session_exists = check_session_status(session_id)
    
    if session_exists:
        # Get session info
        session_info = get_session_info(session_id)
        
        if session_info:
            # Save the session
            success = save_missed_session(session_id)
            
            if success:
                print("\n🎉 Session saved successfully!")
            else:
                print("\n❌ Failed to save session")
        else:
            print("\n❌ Could not retrieve session information")
    else:
        # Try to save anyway (might have recording data)
        print("\nSession not in memory, attempting to save from available data...")
        success = save_missed_session(session_id)
        
        if success:
            print("\n🎉 Session saved successfully!")
        else:
            print("\n❌ Failed to save session")
    
    print("\n" + "="*60)
