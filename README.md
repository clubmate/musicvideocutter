# Music Video Cutter

A Python script that automatically cuts music videos at scene changes and groups similar segments together, then merges them into cohesive videos for each "set" or scene type.

## Description

Music videos often feature multiple sets or locations that the artist jumps between. This tool detects scene cuts, groups visually similar segments, and recombines them to create separate videos for each distinct scene/set. Perfect for analyzing or reorganizing music videos by their visual components.

## Features

- **Multiple Input Sources**: Supports YouTube videos, playlists, or local video files
- **High-Quality Downloads**: Downloads best available quality (up to 1080p) from YouTube
- **Automatic Scene Detection**: Uses content-aware scene detection to identify cuts
- **Intelligent Grouping**: Groups similar scenes using perceptual hashing
- **Flexible Transitions**: Choose between hard cuts or fade transitions
- **Progress Tracking**: Real-time progress bars for all operations
- **Configurable**: YAML-based configuration for all parameters
- **Batch Processing**: Handles YouTube playlists automatically

## Installation

1. Clone or download this repository
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Requirements

- Python 3.7+
- FFmpeg (for video processing)
- The following Python packages:
  - yt-dlp
  - scenedetect
  - moviepy
  - pyyaml
  - tqdm
  - opencv-python
  - imagehash
  - Pillow
  - numpy

## Usage

### Basic Usage

```bash
python musicvideocutter.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

or for local files:

```bash
python musicvideocutter.py "path/to/your/video.mp4"
```

### Examples

Process a single YouTube video:
```bash
python musicvideocutter.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

Process a YouTube playlist:
```bash
python musicvideocutter.py "https://www.youtube.com/playlist?list=PLAYLIST_ID"
```

Process a local video file:
```bash
python musicvideocutter.py "my_music_video.mp4"
```

## Configuration

All settings are controlled via `config.yaml`:

```yaml
scene_detection:
  threshold: 30  # Lower values detect more cuts (more sensitive)
  min_scene_len: 15  # Minimum scene length in seconds

grouping:
  similarity_threshold: 10  # Hamming distance threshold for grouping similar scenes

transition:
  type: hard_cut  # Options: hard_cut, fade_in_out
  fade_duration: 1.0  # Duration of fade in seconds (only for fade_in_out)

output:
  temp_dir: temp_segments  # Directory for temporary segments
  merged_dir: merged_videos  # Directory for merged videos
```

### Configuration Options

- **scene_detection.threshold**: Sensitivity for detecting scene changes (lower = more sensitive)
- **scene_detection.min_scene_len**: Minimum length for a scene in seconds
- **grouping.similarity_threshold**: How similar scenes must be to be grouped (lower = stricter)
- **transition.type**: "hard_cut" for instant transitions, "fade_in_out" for smooth fades
- **transition.fade_duration**: Length of fade transitions in seconds
- **output.temp_dir**: Directory name for temporary segment files
- **output.merged_dir**: Directory name for final merged videos

## How It Works

1. **Download/Input**: Downloads YouTube video or uses local file
2. **Scene Detection**: Analyzes video for scene changes using content detection
3. **Segmentation**: Cuts video into individual scene segments
4. **Analysis**: Extracts perceptual hashes from middle frames of each segment
5. **Grouping**: Groups segments with similar visual content
6. **Merging**: Combines segments within each group with chosen transition
7. **Output**: Saves merged videos in organized directory structure

## Output Structure

For a video titled "Artist - Song Name":

```
Artist - Song Name/
├── temp_segments/
│   ├── segment_000.mp4
│   ├── segment_001.mp4
│   └── ...
└── merged_videos/
    ├── group_000.mp4
    ├── group_001.mp4
    └── ...
```

- `temp_segments/`: Individual scene segments (can be deleted after processing)
- `merged_videos/`: Final merged videos, one per group of similar scenes

## Tips

- For music videos with distinct sets/locations, lower the `similarity_threshold` for stricter grouping
- Increase `scene_detection.threshold` if too many false scene cuts are detected
- Use `fade_in_out` for smoother transitions between segments
- Processing time depends on video length and number of scenes

## Troubleshooting

- **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
- **FFmpeg not found**: Install FFmpeg and ensure it's in your system PATH
- **No groups found**: Try lowering the `similarity_threshold` in config.yaml
- **Too many groups**: Increase the `similarity_threshold`

## License

This project is open source. Feel free to use and modify as needed.
