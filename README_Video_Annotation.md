# Video Annotation Feature

This document describes the video annotation functionality that has been added to the FastAPI server.

## Overview

The video annotation feature automatically processes uploaded video recordings by:

1. **Temporarily saving** the raw footage locally
2. **Annotating the video** with object detection using RFDETR (Real-time Detection Transformer)
3. **Saving the annotated footage** to Supabase instead of the raw footage
4. **Generating detection statistics** and saving them as JSON files

## Features

### Object Detection
- **Person Detection**: Identifies and counts people in the video
- **Device Detection**: Detects laptops, phones, remotes, and TVs
- **Real-time Processing**: Uses RFDETR Nano model for efficient processing
- **Visual Annotations**: Adds bounding boxes and labels to the video

### Detection Statistics
- Total frames processed
- Number of frames with detections
- Count of person detections
- Count of device detections
- Timestamped capture data for frames with multiple people or devices

## API Endpoints

### Upload Recording with Annotation
```
POST /recordings/upload/{session_id}
```

**Parameters:**
- `session_id`: The session identifier
- `recording`: Video file (supports .mp4, .avi, .mov, .webm, .mkv)
- `recording_type`: Type of recording (default: "session")

**Response:**
```json
{
  "success": true,
  "file_path": "session_id/filename_annotated.mp4",
  "file_size": 1234567,
  "message": "Recording uploaded to Supabase successfully: filename_annotated.mp4"
}
```

### Get Detection Data
```
GET /recordings/{session_id}/detection-data
```

**Response:**
```json
{
  "session_id": "session_123",
  "detection_data_available": true,
  "detection_data": {
    "session_id": "session_123",
    "capture_data": {
      "2024-01-15 10:30:45": ["person 0.95", "laptop 0.87"],
      "2024-01-15 10:31:12": ["person 0.92", "person 0.88"]
    },
    "statistics": {
      "total_frames": 1800,
      "processed_frames": 1800,
      "detection_frames": 450,
      "person_detections": 1200,
      "device_detections": 50
    },
    "timestamp": "2024-01-15T10:35:00"
  }
}
```

## File Structure

```
poc/
├── recordings/
│   └── {session_id}/
│       ├── raw_video.mp4          # Temporary raw video (deleted after processing)
│       └── raw_video_annotated.mp4 # Final annotated video (uploaded to Supabase)
├── sessions/
│   └── capture_data_{session_id}.json  # Detection statistics and data
└── fastapi_pipeline.py            # Updated server with video annotation
```

## Dependencies

The video annotation feature requires the following Python packages:

```bash
pip install opencv-python==4.8.1.78
pip install supervision==0.16.0
pip install rfdetr==0.1.0
```

## Processing Flow

1. **Upload**: Client uploads video file to `/recordings/upload/{session_id}`
2. **Local Save**: Server temporarily saves raw video to `recordings/{session_id}/`
3. **Annotation**: VideoAnnotationProcessor processes the video:
   - Loads RFDETR Nano model
   - Processes each frame with object detection
   - Adds bounding boxes and labels
   - Counts detections and generates statistics
4. **Data Save**: Detection statistics saved to `sessions/capture_data_{session_id}.json`
5. **Upload**: Annotated video uploaded to Supabase storage
6. **Cleanup**: Raw video file deleted from local storage

## Error Handling

- **Video Processing Unavailable**: If video libraries are not installed, the system falls back to uploading the raw video without annotation
- **Annotation Failure**: If annotation fails, the system falls back to uploading the raw video
- **Supabase Unavailable**: If Supabase is not available, videos are stored locally

## Testing

Run the test script to verify the video annotation functionality:

```bash
cd poc
python test_video_annotation.py
```

The test script will:
- Create a test video
- Process it with annotation
- Verify the output files
- Test SessionManager integration

## Configuration

The video annotation processor is automatically initialized when the FastAPI server starts. It will:

- Check for required video processing libraries
- Initialize the RFDETR Nano model
- Set up local directories for processing
- Log the initialization status

## Performance Considerations

- **Processing Time**: Video annotation adds processing time proportional to video length
- **Memory Usage**: The RFDETR model requires GPU memory if available
- **Storage**: Annotated videos may be larger than raw videos due to visual overlays
- **Concurrent Processing**: Multiple video uploads are processed sequentially to avoid memory conflicts

## Security Notes

- Raw video files are temporarily stored locally and automatically deleted after processing
- Detection data contains timestamps and object counts but no personal information
- Annotated videos are stored in Supabase with the same access controls as other recordings

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all video processing dependencies are installed
2. **Memory Errors**: Large videos may require more RAM/GPU memory
3. **Processing Failures**: Check logs for specific error messages
4. **File Permission Errors**: Ensure write permissions for `recordings/` and `sessions/` directories

### Log Messages

The system logs detailed information about:
- Video processing initialization
- Frame-by-frame processing progress
- Detection statistics
- File operations
- Error conditions

Check the FastAPI server logs for detailed information about video processing status.
