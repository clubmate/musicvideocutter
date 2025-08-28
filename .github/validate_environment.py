#!/usr/bin/env python3
"""
Simple validation script for musicvideocutter environment.
This script tests that all core dependencies are working correctly.
"""

import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"Testing {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"  ✓ {description} - OK")
            return True
        else:
            print(f"  ✗ {description} - FAILED: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ✗ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"  ✗ {description} - ERROR: {e}")
        return False


def test_python_imports():
    """Test that Python libraries can be imported."""
    imports = [
        ("import sys; print('Python version:', sys.version.split()[0])", "Python"),
    ]
    
    # Only test these if they're available
    optional_imports = [
        ("import cv2; print('OpenCV version:', cv2.__version__)", "OpenCV"),
        ("import moviepy; print('MoviePy version:', moviepy.__version__)", "MoviePy"),
    ]
    
    success = True
    for cmd, desc in imports:
        if not run_command(f"python3 -c \"{cmd}\"", desc):
            success = False
    
    # Test optional imports but don't fail if they're missing
    for cmd, desc in optional_imports:
        run_command(f"python3 -c \"{cmd}\"", f"{desc} (optional)")
    
    return success


def test_ffmpeg():
    """Test FFmpeg functionality."""
    # Test FFmpeg is available
    if not run_command("ffmpeg -version | head -1", "FFmpeg installation"):
        return False
    
    # Create test directory
    test_dir = "/tmp/test_videos"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create test video
    create_cmd = f"cd {test_dir} && ffmpeg -f lavfi -i testsrc=duration=1:size=320x240:rate=30 -pix_fmt yuv420p validation_test.mp4 -y >/dev/null 2>&1"
    if not run_command(create_cmd, "Test video creation"):
        return False
    
    # Test video cutting
    cut_cmd = f"cd {test_dir} && ffmpeg -i validation_test.mp4 -ss 0.1 -t 0.5 validation_test_cut.mp4 -y >/dev/null 2>&1"
    if not run_command(cut_cmd, "Video cutting"):
        return False
    
    # Verify output file exists and has content
    check_cmd = f"[ -s {test_dir}/validation_test_cut.mp4 ]"
    if not run_command(check_cmd, "Output file verification"):
        return False
    
    print("  ✓ All FFmpeg tests passed")
    return True


def main():
    """Run all validation tests."""
    print("=" * 50)
    print("musicvideocutter Environment Validation")
    print("=" * 50)
    
    tests = [
        ("Python imports", test_python_imports),
        ("FFmpeg functionality", test_ffmpeg),
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if not test_func():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ ALL TESTS PASSED - Environment is ready for development")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED - Check the errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()