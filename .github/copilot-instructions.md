# musicvideocutter

musicvideocutter is a Python-based application for cutting and editing music videos. The project uses FFmpeg, OpenCV, and MoviePy for video processing capabilities.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap, build, and test the repository:
- **System Dependencies**: Install required system packages:
  - `sudo apt-get update` -- takes 7 seconds. NEVER CANCEL. Set timeout to 15+ seconds.
  - `sudo apt-get install -y ffmpeg` -- takes 30 seconds. NEVER CANCEL. Set timeout to 60+ seconds.
- **Python Environment Setup**:
  - `python3 -m venv venv` -- takes 3-4 seconds. NEVER CANCEL. Set timeout to 15+ seconds.
  - `source venv/bin/activate` -- instantaneous
  - `pip install --upgrade pip` -- takes 2-3 seconds but may fail due to network timeouts. NEVER CANCEL. Set timeout to 60+ seconds.
- **Core Dependencies**: Install video processing libraries:
  - `pip install moviepy opencv-python` -- takes 15 seconds but may fail due to network timeouts. NEVER CANCEL. Set timeout to 300+ seconds.
  - **If pip install fails with timeout errors**: This is common in CI environments. Retry the command or use `pip install --default-timeout=300 moviepy opencv-python`
- **Development Dependencies**: Install development tools:
  - `pip install pytest black isort flake8` -- takes 12 seconds but may fail due to network timeouts. NEVER CANCEL. Set timeout to 300+ seconds.
  - **If pip install fails with timeout errors**: Retry with `pip install --default-timeout=300 pytest black isort flake8`

### Test the installation:
- **Verify core libraries**: 
  - `python3 -c "import cv2; print('OpenCV version:', cv2.__version__)"` -- should output OpenCV version 4.12.0+
  - `python3 -c "import moviepy; print('MoviePy version:', moviepy.__version__)"` -- should output MoviePy version 2.1.2+
  - `ffmpeg -version | head -1` -- should output FFmpeg version 6.1.1+
- **Run basic tests** (when tests exist):
  - `pytest` -- timing depends on test suite size. NEVER CANCEL. Set timeout to 300+ seconds.

### Linting and code quality:
- `black .` -- formats Python code, takes under 1 second for small projects
- `isort .` -- sorts imports, takes under 1 second for small projects  
- `flake8 .` -- checks code style, takes under 1 second for small projects
- **ALWAYS run these before committing** to avoid CI failures

## Validation

### Manual validation scenarios:
Since this is a video processing application, **ALWAYS** test these scenarios after making changes:

1. **Create test video**: 
   ```bash
   mkdir -p /tmp/test_videos
   cd /tmp/test_videos
   ffmpeg -f lavfi -i testsrc=duration=2:size=320x240:rate=30 -pix_fmt yuv420p test_video.mp4 -y
   ```
   -- takes under 1 second. Creates a 2-second test video.

2. **Basic video cutting**: 
   ```bash
   ffmpeg -i test_video.mp4 -ss 0.2 -t 1.0 test_video_cut.mp4 -y
   ```
   -- takes under 1 second. Cuts video from 0.2s to 1.2s (1 second duration).

3. **Verify video properties**: 
   ```bash
   ffprobe -v quiet -show_entries format=duration,size -show_entries stream=width,height,codec_name test_video.mp4
   ```
   -- should show duration=2.000000, width=320, height=240, codec_name=h264

4. **Python library verification** (when dependencies are installed):
   ```bash
   python3 -c "import cv2; print('OpenCV version:', cv2.__version__)"
   python3 -c "import moviepy; print('MoviePy version:', moviepy.__version__)"
   ```
   -- should output version numbers without errors

**Always run these validation steps after making changes to ensure core functionality works.**

### Quick validation script:
Use this script to validate your environment setup:
```bash
# Run the validation script (included in repository)
python3 .github/validate_environment.py
```

The script tests:
- Python installation and basic imports
- FFmpeg installation and video processing
- Optional libraries (OpenCV, MoviePy) if installed
- Video creation, cutting, and output verification

**Expected runtime**: Under 1 second. The script will exit with code 0 if all core tests pass.

### Testing approach:
- **Always create test videos** in a `/tmp/test_videos/` directory for validation
- **Never commit large video files** to the repository
- **Use small test clips** (under 10MB) for quick validation
- **Test common video formats**: MP4, AVI, MOV
- **Verify output quality** by manually inspecting generated video files

## Common Tasks

### Repository structure (current state):
```
.
..
.git/
.github/
  copilot-instructions.md
.gitignore
README.md
venv/ (after setup)
```

### Typical project structure (when implemented):
```
src/
  musicvideocutter/
    __init__.py
    core/
      video_processor.py
      audio_processor.py
    cli/
      main.py
    gui/ (if implemented)
      app.py
tests/
  test_video_processor.py
  test_audio_processor.py
requirements.txt
setup.py (or pyproject.toml)
```

### Essential commands:
- **Activate environment**: `source venv/bin/activate` (must run this first in any new terminal)
- **Run application** (when implemented): `python -m musicvideocutter.cli.main`
- **Run tests**: `pytest -v`
- **Install project in development mode**: `pip install -e .`

### Common video processing patterns:
When working with video files, always:
1. **Check file existence** before processing
2. **Handle common video formats**: MP4, AVI, MOV, MKV
3. **Implement progress callbacks** for long operations
4. **Use temporary files** for intermediate processing steps
5. **Clean up temporary files** after processing
6. **Validate video integrity** before and after operations

### Performance considerations:
- **Video processing is CPU-intensive** -- operations may take several minutes for large files
- **Memory usage scales with video resolution** -- monitor memory for 4K+ videos
- **Use FFmpeg for format conversions** -- it's faster than pure Python solutions
- **Implement chunked processing** for very large video files

### Debugging tips:
- **Enable verbose logging** for video processing operations
- **Use small test clips** during development
- **Check FFmpeg logs** for encoding/decoding issues
- **Verify video codecs** are supported before processing
- **Test with different video resolutions** and frame rates

### Common issues and solutions:
1. **pip install timeout errors**: Common in CI/development environments
   - Retry the command: `pip install --default-timeout=300 <package>`
   - Use `--retries 3` flag for automatic retries
   - Check network connectivity with `ping pypi.org`

2. **FFmpeg not found**: 
   - Verify installation: `ffmpeg -version`
   - Install if missing: `sudo apt-get install -y ffmpeg`

3. **Permission errors with virtual environment**:
   - Ensure you have write permissions to the directory
   - Use `python3 -m venv venv` instead of `virtualenv venv`

4. **Video processing fails**:
   - Check input file exists and is readable
   - Verify video format is supported: `ffprobe <file>`
   - Test with the provided validation video first

5. **Import errors for cv2 or moviepy**:
   - Ensure virtual environment is activated: `source venv/bin/activate`
   - Reinstall packages: `pip uninstall opencv-python && pip install opencv-python`

### Development workflow:
1. **Always activate virtual environment first**: `source venv/bin/activate`
2. **Write tests first** for new video processing features
3. **Test with small video files** before scaling to larger ones
4. **Run linting tools** before committing: `black . && isort . && flake8 .`
5. **Validate video output manually** -- automated tests can't check visual quality
6. **Use version control for configuration files** but exclude large video files

### Error handling patterns:
- **Always handle FFmpeg errors gracefully**
- **Provide clear error messages** for unsupported video formats
- **Implement retry logic** for network-dependent operations
- **Validate input parameters** before starting processing
- **Log processing progress** for long-running operations

### CI/CD considerations:
- **Exclude video files from repository** (use .gitignore)
- **Use small test videos** for automated testing
- **Mock heavy video operations** in unit tests
- **Test on multiple Python versions** (3.8+)
- **Verify FFmpeg availability** in CI environment

Remember: This is a multimedia application where processing times can vary dramatically based on input size and complexity. Always use appropriate timeouts and never cancel long-running video processing operations.