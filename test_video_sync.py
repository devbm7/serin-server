#!/usr/bin/env python3
"""
Test script to verify video synchronization and timing
"""

import subprocess
import json
from pathlib import Path

def get_video_info(video_path):
    """Get video information using FFprobe."""
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            return None
    except Exception as e:
        print(f"Error getting video info: {e}")
        return None

def analyze_video_timing(video_path):
    """Analyze video timing and properties."""
    print(f"\nAnalyzing video: {video_path}")
    
    info = get_video_info(video_path)
    if not info:
        print("❌ Could not get video information")
        return None
    
    # Extract video and audio streams
    video_stream = None
    audio_stream = None
    
    for stream in info.get('streams', []):
        if stream.get('codec_type') == 'video':
            video_stream = stream
        elif stream.get('codec_type') == 'audio':
            audio_stream = stream
    
    if video_stream:
        print("📹 Video Stream:")
        print(f"  - Codec: {video_stream.get('codec_name', 'unknown')}")
        print(f"  - Resolution: {video_stream.get('width', 'unknown')}x{video_stream.get('height', 'unknown')}")
        print(f"  - Frame rate: {video_stream.get('r_frame_rate', 'unknown')}")
        print(f"  - Duration: {video_stream.get('duration', 'unknown')} seconds")
        print(f"  - Number of frames: {video_stream.get('nb_frames', 'unknown')}")
    
    if audio_stream:
        print("🔊 Audio Stream:")
        print(f"  - Codec: {audio_stream.get('codec_name', 'unknown')}")
        print(f"  - Sample rate: {audio_stream.get('sample_rate', 'unknown')} Hz")
        print(f"  - Duration: {audio_stream.get('duration', 'unknown')} seconds")
        print(f"  - Channels: {audio_stream.get('channels', 'unknown')}")
    
    format_info = info.get('format', {})
    if format_info:
        print("📋 Format Info:")
        print(f"  - Duration: {format_info.get('duration', 'unknown')} seconds")
        print(f"  - Bit rate: {format_info.get('bit_rate', 'unknown')} bps")
    
    return info

def test_video_synchronization():
    """Test video synchronization between original and annotated videos."""
    
    print("="*60)
    print("VIDEO SYNCHRONIZATION TEST")
    print("="*60)
    
    # Look for recent video files
    recordings_dir = Path("poc/recordings")
    if not recordings_dir.exists():
        print("❌ No recordings directory found")
        return False
    
    # Find the most recent session directory
    session_dirs = [d for d in recordings_dir.iterdir() if d.is_dir()]
    if not session_dirs:
        print("❌ No session directories found")
        return False
    
    # Get the most recent session
    latest_session = max(session_dirs, key=lambda d: d.stat().st_mtime)
    print(f"✅ Found latest session: {latest_session.name}")
    
    # Look for original and annotated videos
    original_videos = list(latest_session.glob("*.webm"))
    annotated_videos = list(latest_session.glob("*_annotated.webm"))
    
    if not original_videos:
        print("❌ No original videos found")
        return False
    
    if not annotated_videos:
        print("❌ No annotated videos found")
        return False
    
    original_video = original_videos[0]
    annotated_video = annotated_videos[0]
    
    print(f"✅ Original video: {original_video.name}")
    print(f"✅ Annotated video: {annotated_video.name}")
    
    # Analyze both videos
    print("\n" + "="*40)
    print("ORIGINAL VIDEO ANALYSIS")
    print("="*40)
    original_info = analyze_video_timing(original_video)
    
    print("\n" + "="*40)
    print("ANNOTATED VIDEO ANALYSIS")
    print("="*40)
    annotated_info = analyze_video_timing(annotated_video)
    
    if original_info and annotated_info:
        # Compare timing
        print("\n" + "="*40)
        print("TIMING COMPARISON")
        print("="*40)
        
        orig_duration = float(original_info.get('format', {}).get('duration', 0))
        anno_duration = float(annotated_info.get('format', {}).get('duration', 0))
        
        print(f"Original duration: {orig_duration:.2f} seconds")
        print(f"Annotated duration: {anno_duration:.2f} seconds")
        
        duration_diff = abs(orig_duration - anno_duration)
        print(f"Duration difference: {duration_diff:.2f} seconds")
        
        if duration_diff < 0.5:
            print("✅ Duration is well synchronized")
        elif duration_diff < 2.0:
            print("⚠️  Duration has minor differences")
        else:
            print("❌ Duration has significant differences")
        
        # Check frame rates
        orig_video_stream = None
        anno_video_stream = None
        
        for stream in original_info.get('streams', []):
            if stream.get('codec_type') == 'video':
                orig_video_stream = stream
                break
        
        for stream in annotated_info.get('streams', []):
            if stream.get('codec_type') == 'video':
                anno_video_stream = stream
                break
        
        if orig_video_stream and anno_video_stream:
            orig_fps = orig_video_stream.get('r_frame_rate', 'unknown')
            anno_fps = anno_video_stream.get('r_frame_rate', 'unknown')
            
            print(f"Original FPS: {orig_fps}")
            print(f"Annotated FPS: {anno_fps}")
            
            if orig_fps == anno_fps:
                print("✅ Frame rates match")
            else:
                print("❌ Frame rates don't match")
        
        return True
    
    return False

if __name__ == "__main__":
    success = test_video_synchronization()
    
    print("\n" + "="*60)
    if success:
        print("🎉 Video synchronization test completed!")
    else:
        print("⚠️  Video synchronization test failed!")
    print("="*60)
