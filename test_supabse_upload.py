from supabase import create_client

# Replace with your Supabase project details
SUPABASE_URL = "https://ibnsjeoemngngkqnnjdz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlibnNqZW9lbW5nbmdrcW5uamR6Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MzcwOTgxMSwiZXhwIjoyMDY5Mjg1ODExfQ.9Qr2srBzKeVLkZcq1ZMv-B2-_mj71QyDTzdedgxSCSs"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

bucket_name = "interview-recordings"
file_path = "recordings/session_20250807_183201/session_recording_session_20250807_183201_20250807_183324.webm"
destination_path = "test.webm"
mime_type = "video/webm"

with open(file_path, "rb") as f:
    file_data = f.read()

response = supabase.storage.from_(bucket_name).upload(destination_path, file_data, {"content-type": mime_type})

print("Upload response:", response)