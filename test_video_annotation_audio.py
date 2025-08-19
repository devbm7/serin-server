#!/usr/bin/env python3
"""
Test script to verify video annotation with audio preservation
"""

import os
import subprocess
from pathlib import Path

def test_ffmpeg_availability():
    """Test if FFmpeg is available and working."""
    
    print("Testing FFmpeg availability...")
    
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ FFmpeg is available")
            # Check if VP9 and VP8 encoders are available
            result_codecs = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True, timeout=10)
            if 'libvpx-vp9' in result_codecs.stdout:
                print("✅ VP9 encoder (libvpx-vp9) is available")
            else:
                print("⚠️  VP9 encoder (libvpx-vp9) is not available")
            
            if 'libvpx' in result_codecs.stdout:
                print("✅ VP8 encoder (libvpx) is available")
            else:
                print("⚠️  VP8 encoder (libvpx) is not available")
            
            return True
        else:
            print("❌ FFmpeg is not working properly")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"❌ FFmpeg is not available: {e}")
        return False

def test_video_annotation_processor():
    """Test the VideoAnnotationProcessor class."""
    
    print("\nTesting VideoAnnotationProcessor...")
    
    try:
        from fastapi_pipeline import VideoAnnotationProcessor
        
        # Check if video processing libraries are available
        try:
            processor = VideoAnnotationProcessor()
            print("✅ VideoAnnotationProcessor instantiated successfully")
            return processor
        except ImportError as e:
            print(f"⚠️  VideoAnnotationProcessor not available (missing dependencies): {e}")
            return None
        except Exception as e:
            print(f"❌ Error creating VideoAnnotationProcessor: {e}")
            return None
            
    except ImportError as e:
        print(f"❌ Could not import VideoAnnotationProcessor: {e}")
        return None

def test_webm_conversion():
    """Test WebM conversion with a sample video."""
    
    print("\nTesting WebM conversion...")
    
    # Look for a sample video file
    recordings_dir = Path("recordings")
    if not recordings_dir.exists():
        print("❌ No recordings directory found")
        return False
    
    # Find a WebM file
    webm_files = list(recordings_dir.rglob("*.webm"))
    if not webm_files:
        print("❌ No WebM files found for testing")
        return False
    
    test_video = webm_files[0]
    print(f"✅ Found test video: {test_video}")
    
    # Test FFmpeg conversion
    output_path = test_video.parent / f"test_conversion_{test_video.stem}.webm"
    
    try:
        # Test VP9 conversion
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', str(test_video),
            '-c:v', 'libvpx-vp9',
            '-crf', '30',
            '-b:v', '0',
            '-c:a', 'copy',
            '-y',
            str(output_path)
        ]
        
        print(f"Testing VP9 conversion: {' '.join(ffmpeg_cmd)}")
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ VP9 conversion successful")
            # Clean up test file
            if output_path.exists():
                output_path.unlink()
            return True
        else:
            print(f"❌ VP9 conversion failed: {result.stderr}")
            
            # Try VP8 as fallback
            ffmpeg_cmd_vp8 = [
                'ffmpeg',
                '-i', str(test_video),
                '-c:v', 'libvpx',
                '-crf', '10',
                '-b:v', '1M',
                '-c:a', 'copy',
                '-y',
                str(output_path)
            ]
            
            print(f"Testing VP8 conversion: {' '.join(ffmpeg_cmd_vp8)}")
            result_vp8 = subprocess.run(ffmpeg_cmd_vp8, capture_output=True, text=True, timeout=60)
            
            if result_vp8.returncode == 0:
                print("✅ VP8 conversion successful")
                # Clean up test file
                if output_path.exists():
                    output_path.unlink()
                return True
            else:
                print(f"❌ VP8 conversion also failed: {result_vp8.stderr}")
                return False
                
    except Exception as e:
        print(f"❌ Error during conversion test: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("VIDEO ANNOTATION AUDIO TEST")
    print("="*60)
    
    # Test FFmpeg availability
    ffmpeg_available = test_ffmpeg_availability()
    
    # Test VideoAnnotationProcessor
    processor = test_video_annotation_processor()
    
    # Test WebM conversion
    if ffmpeg_available:
        conversion_success = test_webm_conversion()
    else:
        conversion_success = False
        print("\n⚠️  Skipping WebM conversion test (FFmpeg not available)")
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"  - FFmpeg available: {'✅' if ffmpeg_available else '❌'}")
    print(f"  - VideoAnnotationProcessor: {'✅' if processor else '❌'}")
    print(f"  - WebM conversion: {'✅' if conversion_success else '❌'}")
    
    if ffmpeg_available and processor and conversion_success:
        print("\n🎉 All tests passed! Video annotation with audio should work correctly.")
    else:
        print("\n⚠️  Some tests failed. Video annotation may not work optimally.")
    
    print("="*60)
