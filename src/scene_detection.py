from __future__ import annotations
import os
from typing import Any
from scenedetect import detect, split_video_ffmpeg
from scenedetect.detectors import (
    AdaptiveDetector,
    ContentDetector,
    ThresholdDetector,
    HistogramDetector,
    HashDetector,
)

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


def build_detector(cfg: dict, fps: float | None):
    sd_cfg = cfg.get('scene_detection', {})
    method = (sd_cfg.get('method') or 'adaptive').lower()
    min_scene_len_frames = None
    if (
        'min_scene_len_seconds' in sd_cfg
        and isinstance(sd_cfg['min_scene_len_seconds'], (int, float))
        and fps
        and sd_cfg['min_scene_len_seconds'] > 0
    ):
        min_scene_len_frames = max(1, int(sd_cfg['min_scene_len_seconds'] * fps))
    elif 'min_scene_len' in sd_cfg and isinstance(sd_cfg['min_scene_len'], int) and sd_cfg['min_scene_len'] > 0:
        min_scene_len_frames = sd_cfg['min_scene_len']

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

    mcfg = sd_cfg.get('adaptive', {})
    return AdaptiveDetector(
        adaptive_threshold=mcfg.get('adaptive_threshold', 3.0),
        min_scene_len=min_scene_len_frames or 15,
        window_width=mcfg.get('window_width', 2),
        min_content_val=mcfg.get('min_content_val', 15.0),
    )


def detect_and_split(video_path: str, output_dir: str, cfg: dict):
    os.makedirs(output_dir, exist_ok=True)
    fps = None
    if cv2 is not None:
        try:
            cap = cv2.VideoCapture(video_path)  # type: ignore
            if cap.isOpened():
                fps_val = cap.get(cv2.CAP_PROP_FPS)  # type: ignore
                if fps_val and fps_val > 0:
                    fps = fps_val
            cap.release()
        except Exception:
            pass

    detector = build_detector(cfg, fps)
    scene_list = detect(video_path, detector, show_progress=True)
    split_video_ffmpeg(
        video_path,
        scene_list,
        output_dir=output_dir,
        show_progress=True,
        arg_override='-map 0:v:0 -map 0:a? -map 0:s? -c copy',
    )
    return scene_list
