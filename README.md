TODO:
- HEROSHOT EXPORT ALS PNG (VON EINZELSEGMENTEN UND GROUPSEGMENTEN)
- FRONTEND ZUM NACHTRÄGLICHEN EINGRUPPIEREN DER FALSCH ERKANNTEN VIDEOS
- FLORENCE2 CAPTIONS


# Music Video Cutter - Simplified Version

## Overview

The Music Video Cutter tool detects and cuts scenes from music videos and groups similar scenes based on **actual visual similarity**. 

**Simplified**: All old clustering methods have been removed - only the optimal similarity-based grouping is available.

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd musicvideocutter

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
# Scene detection and cutting only
python musicvideocutter.py "path/to/video.mp4" --no-group

# With YouTube URL
python musicvideocutter.py "https://www.youtube.com/watch?v=VIDEO_ID" --no-group

# Default behavior (with grouping)
python musicvideocutter.py "path/to/video.mp4"
```

### With Similarity Grouping (Default)
```bash
# Standard grouping (recommended)
python musicvideocutter.py "path/to/video.mp4"

# With custom similarity threshold
python musicvideocutter.py "path/to/video.mp4" --min-similarity 0.8

# All available parameters
python musicvideocutter.py "path/to/video.mp4" \
    --group-method cnn \
    --min-similarity 0.75 \
    --min-group-size 2 \
    --orphan-threshold 0.5 \
    --similarity-metric cosine
```

## Parameters

### Similarity-Based Grouping

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--group-method` | `cnn` | Feature extraction method (`histogram`, `cnn`, `audio`) |
| `--min-similarity` | `0.75` | Minimum similarity for grouping (0-1) |
| `--min-group-size` | `2` | Minimum number of videos per group |
| `--orphan-threshold` | `0.5` | Threshold for orphan videos |
| `--similarity-metric` | `cosine` | Similarity metric (`cosine`, `euclidean`) |
| `--no-group` | `false` | Skip grouping (scene detection only) |

## Configuration (config.yaml)

```yaml
scene_detection:
  method: adaptive                # Scene detection method
  min_scene_len: 7               # Minimum scene length in seconds

grouping:
  enabled: true                 # Enable automatic grouping
  method: cnn                   # Feature extraction method (recommended: cnn)
  min_similarity: 0.75          # Minimum similarity for grouping (0-1)
  min_group_size: 2             # Minimum number of videos per group
  orphan_threshold: 0.5         # Threshold for orphan videos
  similarity_metric: cosine     # Similarity metric (recommended: cosine)

output:
  download_dir: output          # Base directory for downloads
  temp_dir: temp_segments       # Target folder for cut scenes
  merged_dir: merged_videos     # Output folder for grouped videos
```

## How Similarity Grouping Works

### 1. Feature Extraction
- **CNN (recommended)**: Deep learning features with ResNet50
- **Histogram**: Color histograms
- **Audio**: Audio features (requires librosa)

### 2. Similarity Calculation
- **Cosine (recommended)**: Cosine similarity between feature vectors
- **Euclidean**: Euclidean distance

### 3. Grouping Algorithm
1. **Find qualifying pairs**: Only video pairs with similarity >= `min_similarity`
2. **Greedy expansion**: Start with best pair, expand only if **all** connections qualify
3. **No transitive inference**: A-B (0.8) + B-C (0.8) leads to A-B-C only if A-C >= 0.75
4. **Quality sorting**: Groups sorted by average similarity
5. **Orphan group**: Videos without sufficient similarities

### 4. Output
- **Quality-sorted groups**: `group_000_sim0.891`, `group_001_sim0.838`, etc.
- **Orphan group**: `group_XXX_orphans` for videos with low similarities
- **Detailed statistics**: JSON file with all metadata

## Example Results

For test video "Haiyti - Sweet" (57 segments) with `--min-similarity 0.75`:

```
Grouping completed:
  Rank 1: group_000_sim0.891 - 5 videos, Quality: 0.891
  Rank 2: group_001_sim0.838 - 8 videos, Quality: 0.838
  Rank 3: group_002_sim0.838 - 6 videos, Quality: 0.838
  ...
  Rank 14: group_013_orphans - 10 videos, Quality: 0.738 (Individual videos)
```

**Result**: 14 final videos - very similar scenes are combined, different ones remain separate.

## Recommended Settings

### For Music Videos (Default)
```bash
python musicvideocutter.py "video.mp4"
```

### For Stricter Grouping
```bash
python musicvideocutter.py "video.mp4" --min-similarity 0.85 --min-group-size 3
```

### For Looser Grouping
```bash
python musicvideocutter.py "video.mp4" --min-similarity 0.65 --orphan-threshold 0.4
```

## Troubleshooting

### TensorFlow not available
```bash
pip install tensorflow
```

### Librosa not available (for audio features)
```bash
pip install librosa
```

### Fallback for problems
```bash
# Use simpler histogram method
python musicvideocutter.py "video.mp4" --group-method histogram
```

## Technical Details

- **No cluster count**: System automatically determines optimal number of groups
- **Quality-based**: Groups are only created if similarity is sufficiently high
- **Robust grouping**: No "weak" groups through transitive connections
- **Scalable**: Works with 2-200+ video segments

## File Structure

```
output/
  video_name/
    video_name.mp4              # Original video
    temp_segments/              # Individual scene segments
      Scene-001.mp4
      Scene-002.mp4
      ...
    merged_videos/              # Grouped final videos
      similarity_group_000_sim0.891.mp4
      similarity_group_001_sim0.838.mp4
      ...
      similarity_grouping_info.json  # Detailed metadata
```
