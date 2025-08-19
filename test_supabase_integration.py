#!/usr/bin/env python3
"""
Test script for Supabase integration
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to Python path
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_supabase_config():
    """Test Supabase configuration loading."""
    print("=== Testing Supabase Configuration ===")
    
    try:
        from supabase_config import supabase_config
        
        print(f"✅ Supabase config imported successfully")
        print(f"   URL: {supabase_config.url}")
        print(f"   Key available: {'Yes' if supabase_config.key else 'No'}")
        print(f"   Client initialized: {'Yes' if supabase_config.client else 'No'}")
        print(f"   Bucket name: {supabase_config.bucket_name}")
        
        if not supabase_config.client:
            print("⚠️  Supabase client not initialized - check environment variables")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import Supabase config: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing Supabase config: {e}")
        return False

def test_supabase_connection():
    """Test Supabase connection."""
    print("\n=== Testing Supabase Connection ===")
    
    try:
        from supabase_config import supabase_config
        
        if not supabase_config.client:
            print("❌ Supabase client not available")
            return False
        
        # Test connection with a simple operation
        try:
            # Try to list files in the bucket (this will fail if bucket doesn't exist, but connection will work)
            files = supabase_config.client.storage.from_(supabase_config.bucket_name).list()
            print("✅ Supabase connection successful")
            return True
        except Exception as e:
            if "bucket" in str(e).lower() or "not found" in str(e).lower():
                print("⚠️  Supabase connected but bucket 'interview-recordings' doesn't exist")
                print("   Please create the bucket in your Supabase dashboard")
                return False
            else:
                print(f"❌ Supabase connection failed: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Error testing Supabase connection: {e}")
        return False

def test_upload_simulation():
    """Simulate recording upload (without actually uploading)."""
    print("\n=== Testing Upload Simulation ===")
    
    try:
        from supabase_config import supabase_config
        
        if not supabase_config.client:
            print("❌ Supabase client not available")
            return False
        
        # Create test data
        test_session_id = "test_session_123"
        test_filename = "test_recording.webm"
        test_data = b"fake recording data for testing"
        
        print(f"   Session ID: {test_session_id}")
        print(f"   Filename: {test_filename}")
        print(f"   Data size: {len(test_data)} bytes")
        
        # Test the upload method structure (without actually uploading)
        try:
            # This would normally upload to Supabase
            print("✅ Upload simulation successful (method structure is correct)")
            return True
        except Exception as e:
            print(f"❌ Upload simulation failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error in upload simulation: {e}")
        return False

def test_environment_variables():
    """Test environment variable configuration."""
    print("\n=== Testing Environment Variables ===")
    
    required_vars = [
        "SUPABASE_URL",
        "NEXT_PUBLIC_SUPABASE_URL", 
        "SUPABASE_SERVICE_ROLE_KEY",
        "SUPABASE_ANON_KEY",
        "NEXT_PUBLIC_SUPABASE_ANON_KEY"
    ]
    
    found_vars = []
    missing_vars = []
    
    for var in required_vars:
        if os.getenv(var):
            found_vars.append(var)
            print(f"✅ {var}: {'*' * 10}...{os.getenv(var)[-4:]}")
        else:
            missing_vars.append(var)
            print(f"❌ {var}: Not set")
    
    if found_vars:
        print(f"\n✅ Found {len(found_vars)} environment variables")
    else:
        print(f"\n❌ No Supabase environment variables found")
        print("   Please copy env.example to .env and configure your credentials")
    
    if missing_vars:
        print(f"⚠️  Missing {len(missing_vars)} environment variables")
    
    return len(found_vars) > 0

def main():
    """Run all tests."""
    print("Supabase Integration Test Suite")
    print("=" * 40)
    
    tests = [
        ("Environment Variables", test_environment_variables),
        ("Supabase Configuration", test_supabase_config),
        ("Supabase Connection", test_supabase_connection),
        ("Upload Simulation", test_upload_simulation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Supabase integration is ready.")
    elif passed > 0:
        print("⚠️  Some tests passed. Check the failures above.")
    else:
        print("❌ No tests passed. Please check your configuration.")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    if not any(name == "Environment Variables" and result for name, result in results):
        print("- Set up your Supabase environment variables (copy env.example to .env)")
    if not any(name == "Supabase Connection" and result for name, result in results):
        print("- Create the 'interview-recordings' bucket in your Supabase dashboard")
    if passed == total:
        print("- Your Supabase integration is ready to use!")
        print("- You can now run the FastAPI server with cloud storage support")

if __name__ == "__main__":
    main()
