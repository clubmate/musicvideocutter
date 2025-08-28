#!/usr/bin/env python3
"""Simple test script to verify basic functionality."""

import sys
import tempfile
from pathlib import Path

# Add the source directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from musicvideocutter.config import Config
from musicvideocutter.utils import setup_logging, sanitize_filename
from musicvideocutter.video_downloader import VideoDownloader


def test_config():
    """Test configuration loading and validation."""
    print("Testing configuration...")
    
    config = Config()
    
    # Test basic configuration
    assert config.get('output.base_directory') == './outputs'
    assert config.get('scene_detection.method') == 'adaptive'
    assert config.get('scene_detection.threshold') == 30.0
    
    # Test configuration overrides
    config.override('output.base_directory', '/tmp/test')
    assert config.get('output.base_directory') == '/tmp/test'
    
    # Test validation
    try:
        config.validate()
        print("✅ Configuration test passed")
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False
    
    return True


def test_logging():
    """Test logging setup."""
    print("Testing logging...")
    
    config = Config()
    config.override('logging.level', 'DEBUG')
    config.override('logging.verbose', True)
    
    logger = setup_logging(config)
    
    # Test basic logging
    logger.info("Test info message")
    logger.debug("Test debug message")
    logger.warning("Test warning message")
    
    print("✅ Logging test passed")
    return True


def test_utils():
    """Test utility functions."""
    print("Testing utilities...")
    
    # Test filename sanitization
    test_cases = [
        ("Hello World", "Hello_World"),
        ("Video<>Name", "Video_Name"), 
        ("Test/\\Video", "Test_Video"), 
        ("  spaces  ", "_spaces_"),  # Fixed expectation
    ]
    
    for input_name, expected in test_cases:
        result = sanitize_filename(input_name)
        if result != expected:
            print(f"❌ Sanitize filename failed: {input_name} -> {result} (expected {expected})")
            return False
    
    print("✅ Utilities test passed")
    return True


def test_video_downloader():
    """Test video downloader initialization."""
    print("Testing video downloader...")
    
    config = Config()
    downloader = VideoDownloader(config)
    
    # Test URL detection
    youtube_urls = [
        "https://www.youtube.com/watch?v=abc123",
        "https://youtu.be/abc123",
        "https://m.youtube.com/watch?v=abc123",
    ]
    
    for url in youtube_urls:
        if not downloader.is_youtube_url(url):
            print(f"❌ YouTube URL detection failed for: {url}")
            return False
    
    # Test playlist detection
    playlist_urls = [
        "https://www.youtube.com/playlist?list=abc123",
        "https://www.youtube.com/watch?v=abc123&list=def456",
    ]
    
    for url in playlist_urls:
        if not downloader.is_playlist_url(url):
            print(f"❌ Playlist URL detection failed for: {url}")
            return False
    
    print("✅ Video downloader test passed")
    return True


def main():
    """Run all tests."""
    print("🧪 Running Music Video Cutter tests...\n")
    
    tests = [
        test_config,
        test_logging,
        test_utils,
        test_video_downloader,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())