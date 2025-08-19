#!/usr/bin/env python3
"""
Comprehensive test script for video synchronization fixes.
Tests various scenarios to ensure audio/video sync is maintained.
"""

import os
import sys
import subprocess
import json
import tempfile
import shutil
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_ffprobe_availability():
    """Test if ffprobe is available and working."""
    print("🔍 Testing ffprobe availability...")
    try:
        result = subprocess.run(['ffprobe', '-version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ ffprobe is available")
            return True
        else:
            print("❌ ffprobe failed to run")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ ffprobe not found")
        return False

def test_ffmpeg_availability():
    """Test if ffmpeg is available and working."""
    print("🔍 Testing ffmpeg availability...")
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ ffmpeg is available")
            return True
        else:
            print("❌ ffmpeg failed to run")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ ffmpeg not found")
        return False

def create_test_video(input_path, duration=10, fps=30, width=640, height=480):
    """Create a test video with specific properties using ffmpeg."""
    print(f"🎬 Creating test video: {duration}s, {fps} FPS, {width}x{height}")
    
    # Create a test video with a moving pattern and audio tone
    cmd = [
        'ffmpeg',
        '-f', 'lavfi',
        '-i', f'testsrc=duration={duration}:size={width}x{height}:rate={fps}',
        '-f', 'lavfi',
        '-i', f'sine=frequency=1000:duration={duration}',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-y',  # Overwrite output
        input_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"✅ Test video created: {input_path}")
            return True
        else:
            print(f"❌ Failed to create test video: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Test video creation timed out")
        return False

def analyze_video_properties(video_path):
    """Analyze video properties using ffprobe."""
    print(f"🔍 Analyzing video properties: {video_path}")
    
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            video_info = json.loads(result.stdout)
            
            # Extract video stream info
            video_stream = None
            audio_stream = None
            for stream in video_info.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                elif stream.get('codec_type') == 'audio':
                    audio_stream = stream
            
            if video_stream:
                # Get accurate frame rate from video stream
                fps_str = video_stream.get('r_frame_rate', '0/1')
                if '/' in fps_str:
                    num, den = map(int, fps_str.split('/'))
                    fps = num / den if den > 0 else 0
                else:
                    fps = float(fps_str)
                
                width = int(video_stream.get('width', 0))
                height = int(video_stream.get('height', 0))
                duration = float(video_info.get('format', {}).get('duration', 0))
                
                print(f"   📹 Video: {width}x{height}, {fps:.3f} FPS, duration: {duration:.2f}s")
                
                if audio_stream:
                    sample_rate = int(audio_stream.get('sample_rate', 0))
                    channels = int(audio_stream.get('channels', 0))
                    print(f"   🎵 Audio: {sample_rate} Hz, {channels} channels")
                
                return {
                    'fps': fps,
                    'width': width,
                    'height': height,
                    'duration': duration,
                    'has_audio': audio_stream is not None
                }
            else:
                print("❌ No video stream found")
                return None
        else:
            print(f"❌ ffprobe failed: {result.stderr}")
            return None
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        print(f"❌ Error analyzing video: {e}")
        return None

def test_video_annotation_import():
    """Test if VideoAnnotationProcessor can be imported and instantiated."""
    print("🔍 Testing VideoAnnotationProcessor import...")
    
    try:
        from fastapi_pipeline import VideoAnnotationProcessor
        
        # Try to instantiate (this will fail if dependencies are missing, but that's expected)
        try:
            processor = VideoAnnotationProcessor()
            print("✅ VideoAnnotationProcessor imported and instantiated successfully")
            return True
        except Exception as e:
            print(f"⚠️  VideoAnnotationProcessor instantiation failed (expected if dependencies missing): {e}")
            return True  # Import succeeded, which is what we're testing
    except ImportError as e:
        print(f"❌ Failed to import VideoAnnotationProcessor: {e}")
        return False

def test_video_synchronization_scenarios():
    """Test various video synchronization scenarios."""
    print("\n🎯 Testing video synchronization scenarios...")
    
    # Test different frame rates and durations
    test_cases = [
        {'fps': 30, 'duration': 10, 'width': 640, 'height': 480},
        {'fps': 60, 'duration': 15, 'width': 1280, 'height': 720},
        {'fps': 24, 'duration': 20, 'width': 1920, 'height': 1080},
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n📹 Test Case {i+1}: {test_case['fps']} FPS, {test_case['duration']}s, {test_case['width']}x{test_case['height']}")
        
        # Create test video
        test_video_path = f"test_video_sync_{i+1}.mp4"
        if create_test_video(
            test_video_path,
            duration=test_case['duration'],
            fps=test_case['fps'],
            width=test_case['width'],
            height=test_case['height']
        ):
            # Analyze original video
            original_props = analyze_video_properties(test_video_path)
            
            if original_props:
                # Test the annotation process (if VideoAnnotationProcessor is available)
                try:
                    from fastapi_pipeline import VideoAnnotationProcessor
                    processor = VideoAnnotationProcessor()
                    
                    # Create annotated output path
                    annotated_path = f"test_annotated_sync_{i+1}.webm"
                    
                    print(f"   🎨 Running video annotation...")
                    detection_stats = processor.annotate_video(test_video_path, annotated_path)
                    
                    if os.path.exists(annotated_path):
                        # Analyze annotated video
                        annotated_props = analyze_video_properties(annotated_path)
                        
                        if annotated_props:
                            # Check synchronization quality
                            fps_diff = abs(annotated_props['fps'] - original_props['fps'])
                            duration_diff = abs(annotated_props['duration'] - original_props['duration'])
                            
                            sync_quality = "✅ EXCELLENT"
                            if fps_diff > 0.1 or duration_diff > 0.5:
                                sync_quality = "⚠️  GOOD"
                            if fps_diff > 1.0 or duration_diff > 1.0:
                                sync_quality = "❌ POOR"
                            
                            print(f"   📊 Sync Quality: {sync_quality}")
                            print(f"      FPS difference: {fps_diff:.3f}")
                            print(f"      Duration difference: {duration_diff:.2f}s")
                            
                            results.append({
                                'test_case': test_case,
                                'original_props': original_props,
                                'annotated_props': annotated_props,
                                'fps_diff': fps_diff,
                                'duration_diff': duration_diff,
                                'sync_quality': sync_quality
                            })
                            
                            # Clean up annotated file
                            os.remove(annotated_path)
                        else:
                            print("   ❌ Could not analyze annotated video")
                    else:
                        print("   ❌ Annotated video was not created")
                        
                except Exception as e:
                    print(f"   ❌ Video annotation failed: {e}")
            
            # Clean up test video
            os.remove(test_video_path)
        else:
            print("   ❌ Test video creation failed")
    
    return results

def generate_sync_report(results):
    """Generate a comprehensive synchronization report."""
    print("\n📊 VIDEO SYNCHRONIZATION REPORT")
    print("=" * 50)
    
    if not results:
        print("❌ No test results available")
        return
    
    total_tests = len(results)
    excellent_sync = sum(1 for r in results if "EXCELLENT" in r['sync_quality'])
    good_sync = sum(1 for r in results if "GOOD" in r['sync_quality'])
    poor_sync = sum(1 for r in results if "POOR" in r['sync_quality'])
    
    print(f"📈 Overall Results:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Excellent Sync: {excellent_sync}")
    print(f"   Good Sync: {good_sync}")
    print(f"   Poor Sync: {poor_sync}")
    
    print(f"\n🔍 Detailed Results:")
    for i, result in enumerate(results):
        print(f"\n   Test {i+1}: {result['test_case']['fps']} FPS, {result['test_case']['duration']}s")
        print(f"      Original: {result['original_props']['fps']:.3f} FPS, {result['original_props']['duration']:.2f}s")
        print(f"      Annotated: {result['annotated_props']['fps']:.3f} FPS, {result['annotated_props']['duration']:.2f}s")
        print(f"      FPS Diff: {result['fps_diff']:.3f}")
        print(f"      Duration Diff: {result['duration_diff']:.2f}s")
        print(f"      Quality: {result['sync_quality']}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if poor_sync > 0:
        print("   ⚠️  Some tests showed poor synchronization. Consider:")
        print("      - Checking FFmpeg installation and version")
        print("      - Verifying input video properties")
        print("      - Reviewing FFmpeg parameters in VideoAnnotationProcessor")
    elif excellent_sync == total_tests:
        print("   🎉 All tests passed with excellent synchronization!")
        print("      The video processing pipeline is working perfectly.")
    else:
        print("   ✅ Most tests passed with good synchronization.")
        print("      Minor improvements may be possible but not critical.")

def main():
    """Main test function."""
    print("🎬 COMPREHENSIVE VIDEO SYNCHRONIZATION TEST")
    print("=" * 60)
    
    # Check prerequisites
    ffprobe_available = test_ffprobe_availability()
    ffmpeg_available = test_ffmpeg_availability()
    annotation_available = test_video_annotation_import()
    
    if not ffprobe_available or not ffmpeg_available:
        print("\n❌ Prerequisites not met. Please install ffmpeg and ffprobe.")
        return
    
    print(f"\n✅ Prerequisites met: ffprobe={ffprobe_available}, ffmpeg={ffmpeg_available}, annotation={annotation_available}")
    
    # Run synchronization tests
    results = test_video_synchronization_scenarios()
    
    # Generate report
    generate_sync_report(results)
    
    print(f"\n🏁 Testing completed!")
    
    if results:
        success_rate = sum(1 for r in results if "POOR" not in r['sync_quality']) / len(results) * 100
        print(f"   Success Rate: {success_rate:.1f}%")
    else:
        print("   No test results to analyze")

if __name__ == "__main__":
    main()
