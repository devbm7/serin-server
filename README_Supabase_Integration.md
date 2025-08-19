# Supabase Integration for FastAPI Server

This document describes the integration of Supabase storage for interview recordings in the FastAPI server.

## Overview

The FastAPI server now supports uploading interview recordings to Supabase storage instead of saving them locally. This provides:

- **Cloud Storage**: Recordings are stored securely in the cloud
- **Scalability**: No local storage limitations
- **Accessibility**: Recordings can be accessed from anywhere
- **Fallback**: Local storage is still available if Supabase is unavailable

## Setup

### 1. Install Dependencies

The Supabase Python client has been added to `requirements_fastapi.txt`. Install it with:

```bash
pip install -r requirements_fastapi.txt
```

### 2. Configure Environment Variables

Copy `env.example` to `.env` and configure your Supabase credentials:

```bash
cp env.example .env
```

Edit `.env` with your actual Supabase values:

```env
# Supabase URL
SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co

# Supabase Service Role Key (preferred)
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key-here

# Supabase Anon Key (fallback)
SUPABASE_ANON_KEY=your-anon-key-here
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key-here
```

### 3. Create Supabase Storage Bucket

In your Supabase dashboard:

1. Go to Storage
2. Create a new bucket called `interview-recordings`
3. Set the bucket to public (if you want public URLs) or private
4. Configure RLS (Row Level Security) policies as needed

## API Endpoints

### Upload Recording

**POST** `/recordings/upload/{session_id}`

Uploads a recording file to Supabase storage.

```bash
curl -X POST "http://localhost:8000/recordings/upload/session_123" \
  -F "recording=@interview.webm" \
  -F "recording_type=session"
```

**Response:**
```json
{
  "success": true,
  "file_path": "session_123/session_recording_session_123_20241201_143022.webm",
  "file_size": 1024000,
  "message": "Recording uploaded to Supabase successfully: session_recording_session_123_20241201_143022.webm"
}
```

### Get Recording Info

**GET** `/recordings/{session_id}`

Returns information about the recording for a session.

**Response:**
```json
{
  "session_id": "session_123",
  "file_path": "session_123/session_recording_session_123_20241201_143022.webm",
  "file_size": 1024000,
  "upload_time": "2024-12-01T14:30:22",
  "public_url": "https://your-project.supabase.co/storage/v1/object/public/interview-recordings/session_123/session_recording_session_123_20241201_143022.webm",
  "bucket": "interview-recordings",
  "storage_type": "supabase"
}
```

### Download Recording

**GET** `/recordings/{session_id}/download`

For Supabase recordings, returns the public URL. For local recordings, serves the file directly.

**Response for Supabase:**
```json
{
  "storage_type": "supabase",
  "public_url": "https://your-project.supabase.co/storage/v1/object/public/interview-recordings/session_123/session_recording_session_123_20241201_143022.webm",
  "file_path": "session_123/session_recording_session_123_20241201_143022.webm",
  "file_size": 1024000
}
```

### List Session Recordings

**GET** `/recordings/{session_id}/list`

Lists all recordings for a session (both Supabase and local).

**Response:**
```json
{
  "session_id": "session_123",
  "recordings": [
    {
      "filename": "session_recording_session_123_20241201_143022.webm",
      "file_path": "session_123/session_recording_session_123_20241201_143022.webm",
      "public_url": "https://your-project.supabase.co/storage/v1/object/public/interview-recordings/session_123/session_recording_session_123_20241201_143022.webm",
      "storage_type": "supabase",
      "bucket": "interview-recordings"
    }
  ],
  "total_count": 1
}
```

### Delete Recording

**DELETE** `/recordings/{session_id}/{filename}`

Deletes a recording from both Supabase and local storage.

```bash
curl -X DELETE "http://localhost:8000/recordings/session_123/session_recording_session_123_20241201_143022.webm"
```

**Response:**
```json
{
  "success": true,
  "message": "Recording deleted from Supabase: session_recording_session_123_20241201_143022.webm Recording deleted from local storage: session_recording_session_123_20241201_143022.webm",
  "filename": "session_recording_session_123_20241201_143022.webm"
}
```

## Storage Behavior

### Primary Storage: Supabase
- Recordings are uploaded to the `interview-recordings` bucket
- Files are organized by session ID: `{session_id}/{filename}`
- Public URLs are generated for easy access

### Fallback Storage: Local
- If Supabase is unavailable, recordings are saved locally
- Local files are stored in `recordings/{session_id}/` directory
- The system automatically falls back to local storage

### Hybrid Mode
- The system can handle recordings stored in both locations
- List endpoints show recordings from both sources
- Download endpoints handle both storage types appropriately

## Error Handling

The system includes comprehensive error handling:

- **Supabase Unavailable**: Automatically falls back to local storage
- **Upload Failures**: Logs errors and continues with fallback
- **Missing Credentials**: Warns but continues with local storage
- **Network Issues**: Graceful degradation to local storage

## Security Considerations

1. **Service Role Key**: Use the service role key for server-side operations
2. **RLS Policies**: Configure appropriate Row Level Security policies in Supabase
3. **Bucket Permissions**: Set bucket permissions according to your security requirements
4. **Environment Variables**: Keep credentials secure and never commit them to version control

## Troubleshooting

### Common Issues

1. **"Supabase client not initialized"**
   - Check that environment variables are set correctly
   - Verify Supabase URL and key are valid

2. **"Failed to upload recording to Supabase"**
   - Check bucket exists and is accessible
   - Verify RLS policies allow uploads
   - Check network connectivity

3. **"Recording saved locally (Supabase unavailable)"**
   - This is normal fallback behavior
   - Check Supabase configuration if you want cloud storage

### Debug Mode

Enable debug logging by setting the log level:

```python
logging.basicConfig(level=logging.DEBUG)
```

## Migration from Local Storage

If you have existing local recordings and want to migrate them to Supabase:

1. The system will continue to serve local recordings
2. New recordings will be uploaded to Supabase
3. You can manually upload existing recordings using the upload endpoint
4. Consider implementing a migration script for bulk uploads

## Performance Considerations

- **Large Files**: Supabase has file size limits (typically 50MB for free tier)
- **Upload Speed**: Depends on network connection and file size
- **Concurrent Uploads**: Supabase handles concurrent uploads well
- **Caching**: Consider implementing caching for frequently accessed recordings
