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
            self.key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlibnNqZW9lbW5nbmdrcW5uamR6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM3MDk4MTEsImV4cCI6MjA2OTI4NTgxMX0.iR8d0XxR-UOPPrK74IIV6Z7gVPP2rHS2b1ZCKwGOSqQ"
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
                    "X-Client-Info": "fastapi-server"
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
            files = self.client.storage.from_(self.bucket_name).list(path=session_id)
            return [file["name"] for file in files if file["name"].endswith(('.webm', '.mp4', '.wav'))]
        except Exception as e:
            logger.error(f"Failed to list session recordings: {e}")
            return []

# Global Supabase configuration instance
supabase_config = SupabaseConfig()
