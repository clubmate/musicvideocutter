# Music Video Cutter

A powerful Python tool for cutting and merging music video segments based on scene detection and similarity analysis. Perfect for extracting and organizing music video sets that jump back and forth between different scenes.

## Features

- 🎬 **Video Input Support**: Load videos from YouTube URLs, playlists, or local files
- 🔍 **Advanced Scene Detection**: Multiple detection methods (adaptive, content-based, threshold)
- 🎭 **Scene Grouping**: Automatically group similar scenes using visual feature analysis
- 🎞️ **Video Processing**: Cut and merge scenes with customizable transition effects
- 🌐 **Cross-Video Analysis**: Optionally group scenes across multiple videos
- ⚙️ **Flexible Configuration**: YAML/JSON configuration with CLI overrides
- 📊 **Progress Tracking**: Real-time progress bars and comprehensive logging
- 🎨 **Transition Effects**: Support for fade, hard cuts, and dissolve transitions

## Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg (for video processing)

### Install FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from [FFmpeg official website](https://ffmpeg.org/download.html)

### Install Music Video Cutter

```bash
# Clone the repository
git clone https://github.com/clubmate/musicvideocutter.git
cd musicvideocutter

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### Basic Usage

```bash
# Process a single YouTube video
musicvideocutter process "https://www.youtube.com/watch?v=VIDEO_ID"

# Process a local video file
musicvideocutter process "/path/to/video.mp4"

# Process a YouTube playlist
musicvideocutter process "https://www.youtube.com/playlist?list=PLAYLIST_ID"
```

### Advanced Usage

```bash
# Custom output directory and settings
musicvideocutter process "VIDEO_URL" \
  --output-dir "./my_output" \
  --method adaptive \
  --threshold 25.0 \
  --similarity 0.8 \
  --transition fade

# Enable cross-video scene grouping
musicvideocutter process "PLAYLIST_URL" \
  --cross-video \
  --max-groups 8

# Extract scenes only (no grouping/merging)
musicvideocutter process "VIDEO_URL" --extract-only
```

## Configuration

### Default Configuration

The application uses a YAML configuration file. You can create a default config:

```bash
musicvideocutter create-config config.yaml
```

### Configuration Options

```yaml
# Output Settings
output:
  base_directory: "./outputs"
  video_format: "mp4"

# Scene Detection Settings
scene_detection:
  method: "adaptive"          # Options: "adaptive", "content", "threshold"
  threshold: 30.0             # Sensitivity (0-100)
  min_scene_length: 1.0       # Minimum scene length in seconds

# Scene Grouping Settings
scene_grouping:
  similarity_threshold: 0.7   # Similarity threshold (0-1)
  max_groups: 10             # Maximum number of groups
  cross_video_grouping: false # Group scenes across videos

# Video Processing Settings
video_processing:
  transition_effect: "fade"   # Options: "fade", "hard_cut", "dissolve"
  transition_duration: 0.5    # Transition duration in seconds
  output_quality: "high"      # Options: "low", "medium", "high"

# Download Settings
download:
  quality: "720p"            # Preferred video quality
  audio_quality: "128k"      # Audio quality
  format: "mp4"             # Output format

# Logging Settings
logging:
  level: "INFO"             # Log level
  log_file: "musicvideocutter.log"
  verbose: false

# Advanced Settings
advanced:
  max_workers: 4            # Parallel processing workers
  cache_scenes: true        # Cache scene detection results
  cleanup_temp: true        # Clean up temporary files
```

### CLI Configuration Override

```bash
# Override any config setting
musicvideocutter process "VIDEO_URL" \
  --config custom_config.yaml \
  --method content \
  --threshold 20 \
  --similarity 0.75 \
  --verbose
```

## How It Works

### 1. Video Input Processing
- Downloads videos from YouTube using yt-dlp
- Supports single videos, playlists, and local files
- Validates and preprocesses video files

### 2. Scene Detection
The tool detects scene cuts using multiple algorithms:

- **Adaptive Detection**: Automatically adjusts to video content
- **Content Detection**: Detects changes in visual content
- **Threshold Detection**: Simple threshold-based detection

### 3. Feature Extraction
For each detected scene:
- Extracts visual features (color, texture, edges)
- Generates color histograms
- Creates scene thumbnails

### 4. Scene Grouping
- Uses machine learning clustering (K-means)
- Calculates visual similarity between scenes
- Groups similar scenes together
- Optionally groups across multiple videos

### 5. Video Processing
- Cuts original videos into scene segments
- Merges similar scenes with transition effects
- Creates a montage video with representatives from each group
- Exports final videos in multiple quality settings

## Output Structure

```
outputs/
├── video_name/
│   ├── scenes/                 # Individual scene clips
│   │   ├── video_scene_001.mp4
│   │   ├── video_scene_002.mp4
│   │   └── ...
│   └── .scene_cache/          # Cached scene detection results
│       └── video_scenes.json
└── merged_videos/
    ├── group_01_5scenes.mp4   # Merged scene groups
    ├── group_02_3scenes.mp4
    ├── montage_all_groups.mp4 # Representative montage
    └── grouping_results.json  # Detailed grouping information
```

## Commands

### `process`
Main command for processing videos:
```bash
musicvideocutter process [OPTIONS] INPUT_SOURCE
```

**Options:**
- `--output-dir, -o`: Output directory
- `--method`: Scene detection method
- `--threshold`: Detection sensitivity
- `--similarity`: Scene similarity threshold
- `--max-groups`: Maximum number of groups
- `--transition`: Transition effect
- `--cross-video`: Enable cross-video grouping
- `--extract-only`: Only extract scenes

### `download`
Download videos without processing:
```bash
musicvideocutter download [OPTIONS] INPUT_SOURCE
```

### `create-config`
Create a default configuration file:
```bash
musicvideocutter create-config CONFIG_PATH
```

### `show-config`
Display current configuration:
```bash
musicvideocutter show-config
```

## Examples

### Music Video Processing
```bash
# Process a music video with custom settings
musicvideocutter process "https://www.youtube.com/watch?v=abc123" \
  --method adaptive \
  --threshold 25 \
  --similarity 0.8 \
  --transition fade \
  --output-dir "./my_music_videos"
```

### Playlist Processing
```bash
# Process entire playlist with cross-video grouping
musicvideocutter process "https://www.youtube.com/playlist?list=xyz789" \
  --cross-video \
  --max-groups 6 \
  --similarity 0.75
```

### Scene Extraction Only
```bash
# Just extract individual scenes without grouping
musicvideocutter process "./local_video.mp4" \
  --extract-only \
  --method content \
  --threshold 30
```

## Troubleshooting

### Common Issues

1. **FFmpeg not found**
   - Install FFmpeg and ensure it's in your PATH
   - On Windows, add FFmpeg to system environment variables

2. **YouTube download fails**
   - Check internet connection
   - Video might be private or restricted
   - Try updating yt-dlp: `pip install --upgrade yt-dlp`

3. **Scene detection takes too long**
   - Reduce video quality in config
   - Increase threshold value for faster but less accurate detection
   - Enable scene caching for repeated processing

4. **Memory issues with large videos**
   - Reduce `max_workers` in config
   - Process videos individually instead of playlists
   - Use "low" output quality setting

### Debugging

Enable verbose logging:
```bash
musicvideocutter process "VIDEO_URL" --verbose
```

Check log files:
```bash
tail -f musicvideocutter.log
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Dependencies

- **yt-dlp**: YouTube video downloading
- **scenedetect**: Scene change detection
- **moviepy**: Video editing and processing
- **opencv-python**: Computer vision and image processing
- **scikit-learn**: Machine learning for scene clustering
- **click**: Command-line interface
- **pyyaml**: Configuration file parsing
- **tqdm**: Progress bars
- **numpy**: Numerical computing

## Acknowledgments

- PySceneDetect for excellent scene detection algorithms
- MoviePy for powerful video editing capabilities
- yt-dlp for robust YouTube downloading
- The open source community for making this project possible