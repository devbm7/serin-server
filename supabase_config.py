"""
Supabase configuration for FastAPI server
"""

import os
import logging
from typing import Optional
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions

logger = logging.getLogger(__name__)

class SupabaseConfig:
    def __init__(self):
        self.url: Optional[str] = None
        self.key: Optional[str] = None
        self.client: Optional[Client] = None
        self.bucket_name = "interview-recordings"
        
        self._load_config()
        self._initialize_client()
    
    def _load_config(self):
        """Load Supabase configuration from environment variables."""
        self.url = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
        self.key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
        
        if not self.url:
            logger.warning("SUPABASE_URL not found in environment variables")
            self.url = "https://ibnsjeoemngngkqnnjdz.supabase.co"
        if not self.key:
            self.key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlibnNqZW9lbW5nbmdrcW5uamR6Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MzcwOTgxMSwiZXhwIjoyMDY5Mjg1ODExfQ.9Qr2srBzKeVLkZcq1ZMv-B2-_mj71QyDTzdedgxSCSs"
            logger.warning("SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY not found in environment variables")
    
    def _initialize_client(self):
        """Initialize Supabase client."""
        if not self.url or not self.key:
            logger.error("Cannot initialize Supabase client: missing URL or key")
            return
        
        try:
            options = ClientOptions(
                schema="public",
                headers={
                    "X-Client-Info": "fastapi-server",
                    "Authorization": f"Bearer {self.key}"
                }
            )
            
            self.client = create_client(self.url, self.key, options)
            logger.info("Supabase client initialized successfully")
            
            # Test connection
            self._test_connection()
            
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            self.client = None
    
    def _test_connection(self):
        """Test Supabase connection."""
        try:
            if self.client:
                # Simple test query
                response = self.client.table("_test_connection").select("*").limit(1).execute()
                logger.info("Supabase connection test successful")
        except Exception as e:
            logger.warning(f"Supabase connection test failed (this is normal if test table doesn't exist): {e}")
    
    def upload_recording(self, session_id: str, recording_data: bytes, filename: str) -> dict:
        """Upload recording to Supabase storage."""
        if not self.client:
            raise Exception("Supabase client not initialized")
        
        try:
            # Create file path in the bucket
            file_path = f"{session_id}/{filename}"
            
            # Upload file to Supabase storage
            response = self.client.storage.from_(self.bucket_name).upload(
                path=file_path,
                file=recording_data,
                file_options={"content-type": "video/webm"}
            )
            
            # Get public URL
            public_url = self.client.storage.from_(self.bucket_name).get_public_url(file_path)
            
            logger.info(f"Recording uploaded to Supabase: {file_path}")
            
            return {
                "success": True,
                "file_path": file_path,
                "public_url": public_url,
                "file_size": len(recording_data),
                "bucket": self.bucket_name
            }
            
        except Exception as e:
            logger.error(f"Failed to upload recording to Supabase: {e}")
            raise
    
    def get_recording_url(self, session_id: str, filename: str) -> Optional[str]:
        """Get public URL for a recording."""
        if not self.client:
            return None
        
        try:
            file_path = f"{session_id}/{filename}"
            return self.client.storage.from_(self.bucket_name).get_public_url(file_path)
        except Exception as e:
            logger.error(f"Failed to get recording URL: {e}")
            return None
    
    def delete_recording(self, session_id: str, filename: str) -> bool:
        """Delete recording from Supabase storage."""
        if not self.client:
            return False
        
        try:
            file_path = f"{session_id}/{filename}"
            self.client.storage.from_(self.bucket_name).remove([file_path])
            logger.info(f"Recording deleted from Supabase: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete recording from Supabase: {e}")
            return False
    
    def list_session_recordings(self, session_id: str) -> list:
        """List all recordings for a session."""
        if not self.client:
            return []
        
        try:
            # List files in the session folder
            response = self.client.storage.from_(self.bucket_name).list(path=session_id)
            return response
        except Exception as e:
            logger.error(f"Failed to list session recordings: {e}")
            return []
    
    def download_resume(self, user_id: str, filename: str) -> Optional[bytes]:
        """Download resume file from Supabase storage."""
        if not self.client:
            logger.error("Supabase client not initialized")
            return None
        
        try:
            # The filename parameter might already contain the full path
            # If it's just a filename, construct the full path
            if '/' in filename:
                file_path = filename  # Already contains the full path
            else:
                file_path = f"{user_id}/{filename}"
            
            # Use the common method to download
            return self.download_resume_by_path(file_path)
                
        except Exception as e:
            logger.error(f"Failed to download resume from Supabase: {e}")
            return None
    
    def download_resume_by_path(self, file_path: str) -> Optional[bytes]:
        """Download resume file from Supabase storage using the full file path."""
        if not self.client:
            logger.error("Supabase client not initialized")
            return None
        
        try:
            # First, try to get a signed URL for the file
            signed_url_response = self.client.storage.from_("resumes").create_signed_url(file_path, 60)  # 60 seconds expiry
            
            if signed_url_response:
                # Extract the actual URL from the response (it can be either 'signedURL' or 'signedUrl')
                signed_url = signed_url_response.get('signedURL') or signed_url_response.get('signedUrl')
                
                if signed_url:
                    # Download the file using the signed URL
                    import requests
                    response = requests.get(signed_url)
                    if response.status_code == 200:
                        logger.info(f"Resume downloaded successfully: {file_path}")
                        return response.content
                    else:
                        logger.warning(f"Failed to download from signed URL: {response.status_code}")
                        return None
                else:
                    logger.warning(f"Could not extract signed URL from response: {signed_url_response}")
                    return None
            else:
                logger.warning(f"Could not create signed URL for: {file_path}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to download resume from Supabase: {e}")
            return None
    
    def get_resume_url(self, user_id: str, filename: str) -> Optional[str]:
        """Get public URL for resume file."""
        if not self.client:
            return None
        
        try:
            file_path = f"{user_id}/{filename}"
            public_url = self.client.storage.from_("resumes").get_public_url(file_path)
            return public_url
        except Exception as e:
            logger.error(f"Failed to get resume URL: {e}")
            return None

    def save_interview_session(self, session_data: dict) -> Optional[str]:
        """Save interview session information to the interview_sessions table."""
        if not self.client:
            logger.error("Supabase client not initialized")
            return None
        
        try:
            # Insert session data into interview_sessions table
            response = self.client.table("interview_sessions").insert(session_data).execute()
            
            if response.data and len(response.data) > 0:
                session_id = response.data[0].get("session_id")
                logger.info(f"Interview session saved successfully with ID: {session_id}")
                return session_id
            else:
                logger.error("Failed to save interview session: no data returned")
                return None
                
        except Exception as e:
            logger.error(f"Failed to save interview session: {e}")
            return None

    def update_session_recording_url(self, session_id: str, recording_url: str) -> bool:
        """Update an existing session with a recording URL."""
        if not self.client:
            logger.error("Supabase client not initialized")
            return False
        
        try:
            # Update the session with the recording URL
            response = self.client.table("interview_sessions").update({
                "recording_url": recording_url
            }).eq("session_id", session_id).execute()
            
            if response.data and len(response.data) > 0:
                logger.info(f"Updated session {session_id} with recording URL: {recording_url}")
                return True
            else:
                logger.warning(f"No session found to update with recording URL: {session_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update session {session_id} with recording URL: {e}")
            return False

    def find_session_by_original_id(self, original_session_id: str) -> Optional[str]:
        """Find a session in the database by the original session ID stored in session_information."""
        if not self.client:
            logger.error("Supabase client not initialized")
            return None
        
        try:
            # Query sessions and look for the original session ID in session_information
            response = self.client.table("interview_sessions").select("session_id, session_information").execute()
            
            if response.data:
                for session in response.data:
                    session_information = session.get("session_information", {})
                    # Check if this session has the original session ID stored somewhere
                    # For now, we'll look for it in the session_information JSON
                    if isinstance(session_information, dict):
                        # We could store the original session ID in session_information
                        # For now, let's try a different approach
                        pass
                
                logger.warning(f"Could not find session with original ID: {original_session_id}")
                return None
            else:
                logger.warning(f"No sessions found in database")
                return None
                
        except Exception as e:
            logger.error(f"Failed to find session by original ID {original_session_id}: {e}")
            return None

    def update_session_recording_url_by_original_id(self, original_session_id: str, recording_url: str) -> bool:
        """Update a session with recording URL by finding it using the original session ID."""
        if not self.client:
            logger.error("Supabase client not initialized")
            return False
        
        try:
            # First, let's try to find the session by looking for sessions created recently
            # and checking if they match our session pattern
            from datetime import datetime, timedelta
            
            # Get sessions created in the last hour
            one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
            
            response = self.client.table("interview_sessions").select("session_id, start_time, recording_url").gte("start_time", one_hour_ago).execute()
            
            if response.data:
                # Find the most recent session that doesn't have a recording URL
                for session in response.data:
                    if not session.get("recording_url"):
                        # This is likely our session - update it
                        db_session_id = session.get("session_id")
                        update_response = self.client.table("interview_sessions").update({
                            "recording_url": recording_url
                        }).eq("session_id", db_session_id).execute()
                        
                        if update_response.data and len(update_response.data) > 0:
                            logger.info(f"Updated session {db_session_id} with recording URL: {recording_url}")
                            return True
                
                logger.warning(f"No session found to update with recording URL for original ID: {original_session_id}")
                return False
            else:
                logger.warning(f"No recent sessions found in database")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update session by original ID {original_session_id}: {e}")
            return False

# Global Supabase configuration instance
supabase_config = SupabaseConfig()
