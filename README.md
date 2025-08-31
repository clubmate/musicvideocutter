# Music Video Cutter

A Python tool that automatically detects scene cuts in music videos, extracts segments, groups visually similar ones (sets), and merges each group into its own compiled video. Grouping is enabled by default (disable with `--no-grouping`).

## Description

Music videos often feature multiple sets or locations that the artist jumps between. This tool detects scene cuts, groups visually similar segments, and recombines them to create separate videos for each distinct scene/set. Perfect for analyzing or reorganizing music videos by their visual components.

## Features

- YouTube videos & playlists or local files
- Multiple scene detection methods: adaptive | content | threshold | histogram | hash
- Lossless segment extraction via FFmpeg stream copy
- Robust set grouping (default enabled): 5 keyframes (0%,25%,50%,75%,100%), HSV histograms + perceptual hash combined distance
- Progress bars (tqdm) for feature extraction, clustering & merging
- Tunable weights & thresholds for grouping
- Modular architecture (`downloader`, `scene_detection`, `grouping`)
- Disable grouping with `--no-grouping`

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

Default (with grouping):
```bash
python musicvideocutter.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Local file:
```bash
python musicvideocutter.py "my_video.mp4"
```

Disable grouping (only cut segments):
```bash
python musicvideocutter.py my_video.mp4 --no-grouping
```

### Examples

YouTube video:
```bash
python musicvideocutter.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

YouTube playlist:
```bash
python musicvideocutter.py "https://www.youtube.com/playlist?list=PLAYLIST_ID"
```

Cut only, no grouping/merging:
```bash
python musicvideocutter.py my_music_video.mp4 --no-grouping
```

## Configuration

`config.yaml` (current structure):

```yaml
scene_detection:
  method: adaptive           # adaptive | content | threshold_params | histogram | hash
  min_scene_len: 6           # Minimum scene length (frames) if seconds not used
  # min_scene_len_seconds: 1 # Alternative in seconds
  adaptive:
    adaptive_threshold: 3.0
    window_width: 2
    min_content_val: 15.0
  content:
    threshold: 27.0
    luma_only: false
  threshold_params:
    threshold: 12
    fade_bias: 0.0
    add_final_scene: false
  histogram:
    threshold: 0.05
    bins: 256
  hash:
    threshold: 0.395
    size: 16
    lowpass: 2

grouping:
  similarity_threshold: 0.45  # Combined distance (histogram + hash)
  hist_bins: 32
  hash_size: 16
  weight_hist: 0.5
  weight_hash: 0.5
  min_cluster_size: 2
  show_progress: true
  log_details: true
  skip_concat: false
  debug_dir: debug_grouping

transition:
  type: hard_cut       # placeholder for future transitions
  fade_duration: 1.0

output:
  download_dir: output
  temp_dir: temp_segments
  merged_dir: merged_videos
```

### Key Options

- `scene_detection.method`: Select algorithm.
- `min_scene_len` / `min_scene_len_seconds`: Minimum duration to avoid micro segments.
- `grouping.similarity_threshold`: Lower = stricter (fewer, purer clusters).
- `weight_hist` / `weight_hash`: Influence of components in combined distance.
- `min_cluster_size`: Minimum cluster size to merge.
- `skip_concat`: Perform clustering only (no merged outputs) for debugging.

### Grouping Mechanics
1. 5 keyframes per segment (start/25%/50%/75%/end)
2. HSV histograms (concatenated & averaged) + perceptual hash (pHash)
3. Distance = w_hist * Bhattacharyya + w_hash * min(normalized Hamming)
4. Union-Find clustering for all pairs below threshold
5. FFmpeg concat (lossless) for clusters ≥ `min_cluster_size`

## Pipeline

1. Download / input normalization
2. Scene detection (selected method)
3. Segmentation (FFmpeg copy, no re-encode loss)
4. Feature extraction (5 frames, hist + hash)
5. Distance computation & clustering
6. Merge per cluster (optional disable)
7. Output structure write

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

- Fewer, purer set videos: lower `similarity_threshold` (e.g. 0.40)
- Too many tiny clusters: raise threshold (0.50) or set `min_cluster_size: 3`
- Need more cuts: try `hash` or `histogram` methods
- Debug without merge: `skip_concat: true` or use `--no-grouping`
- Faster test runs: lower `hist_bins` to 16

## Troubleshooting

- No segments: adjust scene detection method/parameters
- No clusters: increase `similarity_threshold` (e.g. 0.55)
- Too many clusters: lower threshold (0.40) or increase histogram bins
- FFmpeg errors: ensure `ffmpeg` is in PATH (`ffmpeg -version`)
- Slow: reduce `hist_bins`, test shorter clips

## License

Open source – feel free to use & adapt (attribution appreciated).
