#!/usr/bin/env python3
"""
Test script for video annotation functionality in the FastAPI server.
This script tests the video annotation processor independently.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from fastapi_pipeline import VideoAnnotationProcessor
    print("✅ Successfully imported VideoAnnotationProcessor")
except ImportError as e:
    print(f"❌ Failed to import VideoAnnotationProcessor: {e}")
    sys.exit(1)

def test_video_annotation():
    """Test the video annotation functionality."""
    print("\n🧪 Testing Video Annotation Processor...")
    
    # Check if video processing is available
    try:
        import cv2
        import supervision as sv
        from rfdetr import RFDETRNano
        from rfdetr.util.coco_classes import COCO_CLASSES
        print("✅ Video processing libraries are available")
    except ImportError as e:
        print(f"❌ Video processing libraries not available: {e}")
        return False
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a simple test video (if no test video is available)
        test_video_path = temp_path / "test_video.mp4"
        
        # Try to create a simple test video using OpenCV
        try:
            # Create a simple video with some content
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(test_video_path), fourcc, 20.0, (640, 480))
            
            # Create 50 frames of a simple pattern
            for i in range(50):
                # Create a frame with a moving rectangle
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                x = (i * 10) % 600
                cv2.rectangle(frame, (x, 200), (x + 40, 240), (0, 255, 0), -1)
                out.write(frame)
            
            out.release()
            print(f"✅ Created test video: {test_video_path}")
            
        except Exception as e:
            print(f"❌ Failed to create test video: {e}")
            return False
        
        # Test video annotation processor
        try:
            processor = VideoAnnotationProcessor()
            print("✅ VideoAnnotationProcessor initialized successfully")
            
            # Test video annotation
            output_video_path = temp_path / "annotated_video.mp4"
            detection_stats = processor.annotate_video(
                str(test_video_path), 
                str(output_video_path)
            )
            
            print("✅ Video annotation completed successfully")
            print(f"📊 Detection statistics:")
            print(f"   - Total frames: {detection_stats['total_frames']}")
            print(f"   - Processed frames: {detection_stats['processed_frames']}")
            print(f"   - Detection frames: {detection_stats['detection_frames']}")
            print(f"   - Person detections: {detection_stats['person_detections']}")
            print(f"   - Device detections: {detection_stats['device_detections']}")
            
            # Test saving detection data
            detection_data_path = temp_path / "capture_data_test.json"
            processor.save_detection_data("test_session", detection_stats, str(detection_data_path))
            print(f"✅ Detection data saved to: {detection_data_path}")
            
            # Verify files were created
            if output_video_path.exists():
                print(f"✅ Annotated video created: {output_video_path}")
                print(f"   File size: {output_video_path.stat().st_size} bytes")
            else:
                print(f"❌ Annotated video not created")
                return False
            
            if detection_data_path.exists():
                print(f"✅ Detection data file created: {detection_data_path}")
                print(f"   File size: {detection_data_path.stat().st_size} bytes")
            else:
                print(f"❌ Detection data file not created")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Video annotation test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_session_manager_integration():
    """Test the SessionManager integration with video annotation."""
    print("\n🧪 Testing SessionManager Integration...")
    
    try:
        from fastapi_pipeline import SessionManager
        print("✅ Successfully imported SessionManager")
        
        # Create a session manager
        session_manager = SessionManager()
        print("✅ SessionManager initialized successfully")
        
        # Check if video processor is available
        if session_manager.video_processor:
            print("✅ Video annotation processor is available in SessionManager")
        else:
            print("⚠️  Video annotation processor is not available in SessionManager")
        
        return True
        
    except Exception as e:
        print(f"❌ SessionManager integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Video Annotation Tests...")
    
    # Import numpy for test video creation
    try:
        import numpy as np
    except ImportError:
        print("❌ NumPy not available, cannot create test video")
        sys.exit(1)
    
    # Run tests
    test1_passed = test_video_annotation()
    test2_passed = test_session_manager_integration()
    
    print("\n📋 Test Results:")
    print(f"   Video Annotation Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   SessionManager Integration: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Video annotation is working correctly.")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Please check the errors above.")
        sys.exit(1)
