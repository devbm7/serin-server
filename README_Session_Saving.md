# Session Saving Functionality

This document describes the session saving functionality that has been added to the FastAPI server to save interview session information to the `interview_sessions` table in Supabase.

## Overview

When an interview session ends, the system automatically saves comprehensive session information to the database, including:

- **Start Time** - Date and time when the interview started (stored in `start_time` column)
- **Job Role/Interview Topic** - Stored in `session_information` JSON
- **Conversation History** - Stored in `session_information` JSON (excluding system prompt and initial "Hello")
- **Anomalies detected** - Stored in `session_information` JSON (multiple persons, devices detected)
- **Resume URL** - Stored in separate `resume_url` column
- **Recording URL** - Stored in separate `recording_url` column
- **Model information** - Stored in `session_information` JSON (ASR, LLM provider, LLM model)

## Database Schema

The session information is saved to the `interview_sessions` table with the following structure:

```sql
create table public.interview_sessions (
  session_id uuid not null default extensions.uuid_generate_v4 (),
  interviewee_id uuid null,
  template_id uuid null,
  start_time timestamp with time zone null,
  end_time timestamp with time zone null,
  status text not null,
  session_information jsonb null,
  "Interview_report" jsonb null,
  recording_url text null,
  resume_url text null,
  constraint interview_sessions_pkey primary key (session_id),
  constraint interview_sessions_interviewee_id_fkey foreign KEY (interviewee_id) references interviewee_profiles (interviewee_id) on delete CASCADE,
  constraint interview_sessions_template_id_fkey foreign KEY (template_id) references job_templates (template_id) on delete set null
);
```

## Session Information JSON Structure

The `session_information` field contains a JSON object with the following structure:

```json
{
  "job_role": "Software Engineer",
  "conversation_history": [
    {
      "role": "assistant",
      "content": "Hello, I'm Serin. Could you tell me more about your experience with machine learning?"
    },
    {
      "role": "user", 
      "content": "I have worked on several ML projects..."
    }
  ],
  "anomalies_detected": [
    {
      "timestamp": "2025-01-01 10:00:00",
      "type": "multiple_persons",
      "count": 2,
      "detections": ["person 0.95", "person 0.87"]
    },
    {
      "timestamp": "2025-01-01 10:01:00", 
      "type": "devices_detected",
      "count": 1,
      "detections": ["laptop 0.72"]
    }
  ],
  "asr_model": "openai/whisper-medium",
  "llm_provider": "gemini",
  "llm_model": "gemini-2.5-flash"
}
```

## Database Record Structure

Each record in the `interview_sessions` table will have:

- **session_id**: UUID (auto-generated)
- **start_time**: Interview start timestamp
- **status**: "completed"
- **session_information**: JSON object (see above)
- **resume_url**: Direct URL to resume file
- **recording_url**: Direct URL to recording file

## API Endpoints

### 1. Save Session Information
```
POST /sessions/{session_id}/save
```
Manually save session information to the database without deleting the session.

**Response:**
```json
{
  "status": "saved",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Session information saved to database successfully"
}
```

### 2. End Session Gracefully
```
POST /sessions/{session_id}/end
```
Save session information and then delete the session from memory.

**Response:**
```json
{
  "status": "ended", 
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Session ended and information saved to database"
}
```

### 3. Get Session Information
```
GET /sessions/{session_id}/info
```
Get the session information that would be saved to the database.

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "session_information": {
    "date_of_interview": "2025-01-01T10:00:00",
    "job_role": "Software Engineer",
    "conversation_history": [...],
    "anomalies_detected": [...],
    "resume_url": "user_id/resume.pdf",
    "recording_url": "https://...",
    "asr_model": "openai/whisper-medium",
    "llm_provider": "gemini",
    "llm_model": "gemini-2.5-flash"
  },
  "recording_url": "https://..."
}
```

## Session ID Format

Session IDs are now generated as UUIDs (e.g., `550e8400-e29b-41d4-a716-446655440000`) instead of timestamp-based strings. This ensures compatibility with the database schema which expects UUID primary keys.

## Automatic Saving

Session information is automatically saved in the following scenarios:

1. **Session Deletion**: When a session is deleted via `DELETE /sessions/{session_id}`, the system automatically saves the session information before cleanup.

2. **Graceful Ending**: When using `POST /sessions/{session_id}/end`, the session is saved and then deleted.

## Anomaly Detection

The system automatically detects and records anomalies during the interview:

### Multiple Persons Detected
- When more than one person is detected in the video frame
- Indicates potential cheating or unauthorized assistance

### Devices Detected  
- When laptops, phones, TVs, or other devices are detected
- May indicate use of external resources or distractions

### Anomaly Data Structure
```json
{
  "timestamp": "2025-01-01 10:00:00",
  "type": "multiple_persons|devices_detected",
  "count": 2,
  "detections": ["person 0.95", "person 0.87", "laptop 0.72"]
}
```

## Testing

Use the provided test script to verify the functionality:

```bash
python test_session_saving.py
```

The test script will:
1. Create a test session with UUID
2. Validate UUID format
3. Get session information
4. Save session information
5. End session gracefully
6. Verify session deletion
7. Test anomaly detection

## Error Handling

- If Supabase is not available, session saving is skipped with a warning log
- If session saving fails, the error is logged but doesn't prevent session deletion
- All database operations are wrapped in try-catch blocks for robustness
- UUID validation is performed to ensure proper session ID format

## Configuration

The session saving functionality requires:
- Supabase connection (URL and API key)
- `interview_sessions` table in the database
- Proper permissions for the service role

Environment variables:
- `SUPABASE_URL` or `NEXT_PUBLIC_SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY` or `SUPABASE_ANON_KEY`

## Files Modified

1. **`supabase_config.py`**: Added `save_interview_session()` method
2. **`fastapi_pipeline.py`**: 
   - Added `save_session_to_database()` method to SessionManager
   - Updated `delete_session()` to save before deletion
   - Added new API endpoints for session management
   - Changed session ID generation to use UUIDs
3. **`test_session_saving.py`**: Test script for verification with UUID validation
4. **`README_Session_Saving.md`**: This documentation
