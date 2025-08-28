# Music Video Cutter - Usage Examples

This document provides practical examples of how to use Music Video Cutter.

## Basic Examples

### 1. Process a YouTube Video
```bash
# Basic processing with default settings
musicvideocutter process "https://www.youtube.com/watch?v=abc123"

# With custom output directory
musicvideocutter process "https://www.youtube.com/watch?v=abc123" --output-dir "./my_videos"
```

### 2. Process a Local Video File
```bash
# Process a local MP4 file
musicvideocutter process "/path/to/video.mp4"

# Extract scenes only (no grouping/merging)
musicvideocutter process "/path/to/video.mp4" --extract-only
```

### 3. Process a YouTube Playlist
```bash
# Process entire playlist
musicvideocutter process "https://www.youtube.com/playlist?list=abc123"

# With cross-video scene grouping
musicvideocutter process "https://www.youtube.com/playlist?list=abc123" --cross-video
```

## Advanced Configuration

### 4. Custom Scene Detection
```bash
# Use content-based detection with high sensitivity
musicvideocutter process "VIDEO_URL" \
  --method content \
  --threshold 15.0 \
  --similarity 0.85

# Adaptive detection with more groups
musicvideocutter process "VIDEO_URL" \
  --method adaptive \
  --threshold 25.0 \
  --max-groups 15
```

### 5. Video Processing Options
```bash
# High quality output with fade transitions
musicvideocutter process "VIDEO_URL" \
  --transition fade \
  --similarity 0.8 \
  --max-groups 8

# Hard cuts (no transitions) for faster processing
musicvideocutter process "VIDEO_URL" \
  --transition hard_cut \
  --threshold 30.0
```

### 6. Using Configuration Files
```bash
# Create a custom config file
musicvideocutter create-config my_config.yaml

# Edit the config file as needed, then use it
musicvideocutter process "VIDEO_URL" --config my_config.yaml

# Override specific settings
musicvideocutter process "VIDEO_URL" \
  --config my_config.yaml \
  --threshold 20.0 \
  --verbose
```

## Workflow Examples

### 7. Music Video Analysis Workflow
```bash
# Step 1: Download and analyze a music video
musicvideocutter process "https://www.youtube.com/watch?v=MUSIC_VIDEO" \
  --method adaptive \
  --threshold 25.0 \
  --similarity 0.75 \
  --transition fade \
  --output-dir "./music_analysis"

# Step 2: Process multiple videos from an artist
musicvideocutter process "https://www.youtube.com/playlist?list=ARTIST_VIDEOS" \
  --cross-video \
  --max-groups 10 \
  --similarity 0.7 \
  --output-dir "./artist_collection"
```

### 8. Scene Collection Workflow
```bash
# Extract scenes only for manual review
musicvideocutter process "VIDEO_URL" \
  --extract-only \
  --method content \
  --threshold 20.0 \
  --output-dir "./scene_library"

# Then process with specific grouping settings
musicvideocutter process "VIDEO_URL" \
  --similarity 0.8 \
  --max-groups 6 \
  --transition dissolve
```

## Expected Output Structure

After processing, you'll find:

```
outputs/
├── video_name/
│   ├── scenes/                    # Individual scene clips
│   │   ├── video_scene_001.mp4
│   │   ├── video_scene_002.mp4
│   │   └── ...
│   └── .scene_cache/             # Cached scene detection
│       └── video_scenes.json
└── merged_videos/
    ├── group_01_5scenes.mp4      # Merged similar scenes
    ├── group_02_3scenes.mp4
    ├── montage_all_groups.mp4    # Overview of all groups
    └── grouping_results.json     # Detailed analysis results
```

## Performance Tips

### 9. Fast Processing
```bash
# Quick analysis with lower quality
musicvideocutter process "VIDEO_URL" \
  --threshold 35.0 \
  --similarity 0.6 \
  --max-groups 5 \
  --transition hard_cut
```

### 10. High Quality Analysis
```bash
# Detailed analysis for best results
musicvideocutter process "VIDEO_URL" \
  --method adaptive \
  --threshold 15.0 \
  --similarity 0.85 \
  --max-groups 12 \
  --transition fade \
  --verbose
```

## Troubleshooting

### Common Commands
```bash
# Check current configuration
musicvideocutter show-config

# Enable verbose logging
musicvideocutter process "VIDEO_URL" --verbose

# Download only (no processing)
musicvideocutter download "VIDEO_URL" --output-dir "./downloads"

# Get help
musicvideocutter --help
musicvideocutter process --help
```