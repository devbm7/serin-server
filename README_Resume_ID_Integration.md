# Resume ID and Resume URL Integration for FastAPI Server

## Overview

The FastAPI server has been updated to support both user IDs (resume IDs) and direct resume URLs for accessing resumes. This dual approach provides flexibility - you can either provide a user ID to fetch the resume from their profile, or provide a direct path to the resume in Supabase storage.

## Changes Made

### 1. FastAPI Server (`fastapi_pipeline.py`)

#### Updated Classes:
- **ModelSelectionRequest**: Added both `resume_id` and `resume_url` as optional fields
- **SessionManager**: Added methods to handle both resume loading approaches

#### New Methods:
- `load_resume_content_for_session(resume_id, resume_url)`: Orchestrates resume loading
- `_load_resume_from_path(resume_file_path)`: Loads resume directly from Supabase path
- `_load_resume_from_profile(user_id)`: Loads resume using user ID from profile
- `get_user_profile(user_id: str)`: Fetches user profile from database

#### Updated Methods:
- `create_session()`: Now uses both resume_id and resume_url
- `load_resume_content_from_supabase()`: Kept for backward compatibility

### 2. Supabase Configuration (`supabase_config.py`)

#### New Methods:
- `download_resume(user_id: str, filename: str)`: Downloads resume by user ID and filename
- `download_resume_by_path(file_path: str)`: Downloads resume using full file path
- `get_resume_url(user_id: str, filename: str)`: Gets public URL for resume

### 3. Client Libraries

#### FastAPI Client (`fastapi_client.py`):
- Updated `SessionConfig` class to include both `resume_id` and `resume_url`
- Updated `create_session()` method to send both fields in request

#### Interactive Client (`interactive_client.py`):
- Updated to prompt for both user ID and resume URL
- Provides flexibility to use either or both fields

### 4. Startup Script (`start_fastapi_server.py`)

#### Removed:
- `check_resume_file()` function (no longer needed)
- Resume file existence checks

## How It Works

The system supports multiple approaches for loading resumes:

### Approach 1: Using User ID (resume_file as UUID)
1. **User Profile Lookup**: When creating a session with a UUID as `resume_file`, the system looks up the user's profile in the `user_profiles` table.
2. **Resume Path Extraction**: The user's profile contains a `resume_url` field that stores the full file path in Supabase storage (e.g., "user_id/filename.pdf").
3. **Resume Download**: The system creates a signed URL and downloads the resume file from Supabase storage using the file path.
4. **Content Processing**: The PDF content is extracted and used for the interview session.

### Approach 2: Using Direct Resume Path (resume_file as path)
1. **Direct Access**: When creating a session with a file path as `resume_file`, the system directly downloads the resume from the provided Supabase storage path.
2. **Signed URL**: The system creates a signed URL for secure access to the file.
3. **Content Processing**: The PDF content is extracted and used for the interview session.

### Approach 3: Using Local File (resume_file as local path)
1. **Local File Access**: When creating a session with a local file path as `resume_file`, the system reads the file directly from the local filesystem.
2. **Content Processing**: The PDF content is extracted and used for the interview session.

### URL Structure
The system uses signed URLs for secure access to Supabase storage:
- **Correct Format**: `https://[project].supabase.co/storage/v1/object/sign/resumes/[path]?token=[token]`
- **Secure Access**: Uses temporary signed URLs with 60-second expiry for security

## Database Schema

The system uses the existing `user_profiles` table with these fields:
- `user_id`: UUID (primary key)
- `resume_url`: TEXT (stores the full file path in Supabase storage)
- `resume_filename`: TEXT (stores the original filename)

## API Changes

### Before:
```json
{
  "interview_topic": "Software Engineering",
  "resume_file": "path/to/resume.pdf",
  "asr_model": "openai/whisper-medium",
  "llm_provider": "ollama",
  "llm_model": "llama3.2:1b"
}
```

### After:
```json
{
  "interview_topic": "Software Engineering",
  "resume_file": "user-uuid-here",  // Can be user ID, resume path, or local file path
  "asr_model": "openai/whisper-medium",
  "llm_provider": "ollama",
  "llm_model": "llama3.2:1b"
}
```

### Examples:

#### Using User ID (UUID):
```json
{
  "interview_topic": "Software Engineering",
  "resume_file": "23a0b603-e437-42d6-b1e0-6e0a1b983150",
  "asr_model": "openai/whisper-medium",
  "llm_provider": "ollama",
  "llm_model": "llama3.2:1b"
}
```

#### Using Direct Resume Path in Supabase:
```json
{
  "interview_topic": "Software Engineering",
  "resume_file": "23a0b603-e437-42d6-b1e0-6e0a1b983150/resume_1754526365292.pdf",
  "asr_model": "openai/whisper-medium",
  "llm_provider": "ollama",
  "llm_model": "llama3.2:1b"
}
```

#### Using Local File Path:
```json
{
  "interview_topic": "Software Engineering",
  "resume_file": "/path/to/local/resume.pdf",
  "asr_model": "openai/whisper-medium",
  "llm_provider": "ollama",
  "llm_model": "llama3.2:1b"
}
```

## Testing

A test script has been created (`test_resume_id_integration.py`) to verify the integration:

```bash
python test_resume_id_integration.py
```

The test script:
1. Checks if the server is running
2. Creates a session with a test resume_id
3. Verifies session creation and resume loading
4. Cleans up the test session

## Prerequisites

1. **Supabase Setup**: Ensure Supabase is properly configured with:
   - `resumes` storage bucket
   - `user_profiles` table with resume fields
   - Proper RLS policies

2. **Environment Variables**: Set up the following environment variables:
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_ROLE_KEY` or `SUPABASE_ANON_KEY`

3. **Dependencies**: Ensure all required packages are installed:
   ```bash
   pip install -r requirements_fastapi.txt
   ```

## Error Handling

The system includes comprehensive error handling for:
- Missing user profiles
- Resume files not found in storage
- Supabase connection issues
- PDF processing errors

## Benefits

1. **Flexibility**: Support for both user ID lookup and direct URL access
2. **Scalability**: No longer dependent on local file system
3. **Security**: Resumes are stored securely in Supabase
4. **Performance**: Direct URL access can be faster than profile lookup
5. **Multi-user Support**: Each user can have their own resume
6. **Cloud-based**: Works across different deployment environments
7. **Database Integration**: Seamless integration with user profiles
8. **Backward Compatibility**: Existing code using user IDs continues to work
9. **Future-Proof**: Can easily add more resume loading methods

## Migration Notes

If you have existing code that uses `resume_file`, you'll need to:
1. Update API calls to use either `resume_id` or `resume_url` instead of `resume_file`
2. Ensure users have resumes uploaded to Supabase storage
3. Update any client code to pass user IDs or resume URLs instead of file paths
4. Choose the appropriate approach based on your use case:
   - Use `resume_id` if you want to fetch from user profiles
   - Use `resume_url` if you have direct access to resume paths
   - Use both if you want maximum flexibility

## Troubleshooting

### Common Issues:

1. **"Resume not found for this user"**
   - Check if the user has uploaded a resume
   - Verify the user profile exists in the database
   - Check if the resume_url field is populated

2. **"Supabase storage is not available"**
   - Verify Supabase credentials are set correctly
   - Check if the `resumes` bucket exists
   - Ensure RLS policies are configured

3. **"Failed to download resume from Supabase"**
   - Check if the file path in resume_url is correct
   - Verify the file exists in Supabase storage
   - Check storage permissions

### Debug Mode:

Enable debug logging by setting the log level to DEBUG in the FastAPI server configuration.
