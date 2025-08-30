import os
import argparse
import yaml
import scenedetect
from scenedetect import detect, split_video_ffmpeg
from scenedetect.detectors import (
    AdaptiveDetector,
    ContentDetector,
    ThresholdDetector,
    HistogramDetector,
    HashDetector,
)
from src.downloader import download_video

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def _build_detector(cfg: dict, fps: float | None):
    """Return a detector instance based on config.

    Expected structure in config (scene_detection section):
      method: adaptive|content|threshold|histogram|hash (default adaptive)
      min_scene_len: seconds (optional)
      <method-specific sub-dict> with parameter overrides.
    Unspecified params use library defaults.
    """
    sd_cfg = cfg.get('scene_detection', {})
    method = (sd_cfg.get('method') or 'adaptive').lower()
    # Allow either frames (min_scene_len) or seconds (min_scene_len_seconds). Seconds take precedence.
    min_scene_len_frames = None
    if 'min_scene_len_seconds' in sd_cfg and isinstance(sd_cfg['min_scene_len_seconds'], (int, float)) and fps:
        if sd_cfg['min_scene_len_seconds'] > 0:
            min_scene_len_frames = max(1, int(sd_cfg['min_scene_len_seconds'] * fps))
    elif 'min_scene_len' in sd_cfg and isinstance(sd_cfg['min_scene_len'], int):
        if sd_cfg['min_scene_len'] > 0:
            min_scene_len_frames = sd_cfg['min_scene_len']
    # Fallback default used by detectors if None
    # Per-detector overrides
    if method == 'content':
        mcfg = sd_cfg.get('content', {})
        return ContentDetector(
            threshold=mcfg.get('threshold', 27.0),
            min_scene_len=min_scene_len_frames or 15,
            luma_only=mcfg.get('luma_only', False),
        )
    if method == 'threshold':
        mcfg = sd_cfg.get('threshold_params', {})
        return ThresholdDetector(
            threshold=mcfg.get('threshold', 12),
            min_scene_len=min_scene_len_frames or 15,
            fade_bias=mcfg.get('fade_bias', 0.0),
            add_final_scene=mcfg.get('add_final_scene', False),
        )
    if method == 'histogram':
        mcfg = sd_cfg.get('histogram', {})
        return HistogramDetector(
            threshold=mcfg.get('threshold', 0.05),
            bins=mcfg.get('bins', 256),
            min_scene_len=min_scene_len_frames or 15,
        )
    if method == 'hash':
        mcfg = sd_cfg.get('hash', {})
        return HashDetector(
            threshold=mcfg.get('threshold', 0.395),
            size=mcfg.get('size', 16),
            lowpass=mcfg.get('lowpass', 2),
            min_scene_len=min_scene_len_frames or 15,
        )
    # default adaptive
    mcfg = sd_cfg.get('adaptive', {})
    return AdaptiveDetector(
        adaptive_threshold=mcfg.get('adaptive_threshold', 3.0),
        min_scene_len=min_scene_len_frames or 15,
        window_width=mcfg.get('window_width', 2),
        min_content_val=mcfg.get('min_content_val', 15.0),
    )


def detect_and_split(video_path, output_dir, cfg):
    os.makedirs(output_dir, exist_ok=True)
    fps = None
    try:
        # Lightweight fps probe via OpenCV if available
        import cv2  # optional
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps_val = cap.get(cv2.CAP_PROP_FPS)
            if fps_val and fps_val > 0:
                fps = fps_val
        cap.release()
    except Exception:
        pass
    detector = _build_detector(cfg, fps)
    scene_list = detect(video_path, detector, show_progress=True)
    split_video_ffmpeg(
        video_path,
        scene_list,
        output_dir=output_dir,
        show_progress=True,
        arg_override='-map 0:v:0 -map 0:a? -map 0:s? -c copy',  # Lossless copy
    )
    return scene_list


def main():
    config = load_config()
    parser = argparse.ArgumentParser(description="Cut and merge music video segments by sets.")
    parser.add_argument('input', help='YouTube URL or local file path')
    args = parser.parse_args()
    input_path = args.input

    if input_path.startswith('http'):
        download_dir = config['output']['download_dir']
        videos = download_video(input_path, download_dir)
    else:
        videos = [(input_path, os.path.splitext(os.path.basename(input_path))[0])]

    for video_path, title in videos:
        # For downloaded videos, the video_path already includes the subdirectory
        # For local files, create output structure based on file location
        if input_path.startswith('http'):
            # video_path is already in output/video_name/video.mp4
            video_dir = os.path.dirname(video_path)
            temp_dir = os.path.join(video_dir, config['output']['temp_dir'])
            merged_dir = os.path.join(video_dir, config['output']['merged_dir'])
        else:
            # Check if local file is already in a structured directory (e.g., output/video_name/video.mp4)
            video_dir = os.path.dirname(os.path.abspath(video_path))
            video_filename = os.path.basename(video_path)
            
            # If the parent directory name matches the video filename (without extension), use that directory
            parent_dir_name = os.path.basename(video_dir)
            video_name_without_ext = os.path.splitext(video_filename)[0]
            
            if parent_dir_name == video_name_without_ext or video_dir != os.getcwd():
                # File is already in a structured directory, use it
                temp_dir = os.path.join(video_dir, config['output']['temp_dir'])
                merged_dir = os.path.join(video_dir, config['output']['merged_dir'])
            else:
                # File is in current directory, create new structure
                output_base = title.replace('/', '_').replace('\\', '_').replace(':', '_')
                output_dir = output_base
                temp_dir = os.path.join(output_dir, config['output']['temp_dir'])
                merged_dir = os.path.join(output_dir, config['output']['merged_dir'])

        print(f"Processing {title}")
        scenes = detect_and_split(video_path, temp_dir, config)
        print(f"Detected and split video into {len(scenes)} segments")
        print(f"Done processing {title}")

if __name__ == "__main__":
    main()
