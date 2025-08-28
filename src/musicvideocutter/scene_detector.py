"""Scene detection functionality using scenedetect and OpenCV."""

import logging
from pathlib import Path
from typing import List, Tuple, Optional
import json

import cv2
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector, AdaptiveDetector, ThresholdDetector
from scenedetect.video_splitter import split_video_ffmpeg

from .utils import SceneDetectionError, ProgressTracker, format_duration


logger = logging.getLogger(__name__)


class SceneInfo:
    """Information about a detected scene."""
    
    def __init__(self, start_time: float, end_time: float, scene_number: int):
        """Initialize scene information."""
        self.start_time = start_time
        self.end_time = end_time
        self.scene_number = scene_number
        self.duration = end_time - start_time
    
    def __repr__(self) -> str:
        return f"Scene({self.scene_number}: {format_duration(self.start_time)} - {format_duration(self.end_time)})"
    
    def to_dict(self) -> dict:
        """Convert scene info to dictionary."""
        return {
            'scene_number': self.scene_number,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SceneInfo':
        """Create SceneInfo from dictionary."""
        return cls(
            start_time=data['start_time'],
            end_time=data['end_time'],
            scene_number=data['scene_number']
        )


class SceneDetector:
    """Handles scene detection in videos."""
    
    def __init__(self, config):
        """Initialize scene detector with configuration."""
        self.config = config
        self.method = config.get('scene_detection.method', 'adaptive')
        self.threshold = config.get('scene_detection.threshold', 30.0)
        self.min_scene_length = config.get('scene_detection.min_scene_length', 1.0)
        self.cache_enabled = config.get('advanced.cache_scenes', True)
    
    def _get_detector(self):
        """Get the appropriate scene detector based on configuration."""
        if self.method == 'content':
            return ContentDetector(threshold=self.threshold)
        elif self.method == 'adaptive':
            return AdaptiveDetector(adaptive_threshold=self.threshold)
        elif self.method == 'threshold':
            return ThresholdDetector(threshold=self.threshold)
        else:
            logger.warning(f"Unknown detection method: {self.method}, using adaptive")
            return AdaptiveDetector(adaptive_threshold=self.threshold)
    
    def _get_cache_path(self, video_path: Path) -> Path:
        """Get cache file path for scene detection results."""
        cache_dir = video_path.parent / '.scene_cache'
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / f"{video_path.stem}_scenes.json"
    
    def _save_scenes_to_cache(self, video_path: Path, scenes: List[SceneInfo]) -> None:
        """Save scene detection results to cache."""
        if not self.cache_enabled:
            return
        
        cache_path = self._get_cache_path(video_path)
        try:
            scenes_data = {
                'video_path': str(video_path),
                'method': self.method,
                'threshold': self.threshold,
                'min_scene_length': self.min_scene_length,
                'scenes': [scene.to_dict() for scene in scenes]
            }
            
            with open(cache_path, 'w') as f:
                json.dump(scenes_data, f, indent=2)
            
            logger.debug(f"Scene detection results cached: {cache_path}")
            
        except Exception as e:
            logger.warning(f"Failed to cache scene detection results: {e}")
    
    def _load_scenes_from_cache(self, video_path: Path) -> Optional[List[SceneInfo]]:
        """Load scene detection results from cache."""
        if not self.cache_enabled:
            return None
        
        cache_path = self._get_cache_path(video_path)
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                scenes_data = json.load(f)
            
            # Check if cache is valid (same parameters)
            if (scenes_data.get('method') == self.method and
                scenes_data.get('threshold') == self.threshold and
                scenes_data.get('min_scene_length') == self.min_scene_length):
                
                scenes = [SceneInfo.from_dict(scene_data) for scene_data in scenes_data['scenes']]
                logger.info(f"Loaded {len(scenes)} scenes from cache")
                return scenes
            
        except Exception as e:
            logger.warning(f"Failed to load scene detection cache: {e}")
        
        return None
    
    def detect_scenes(self, video_path: Path) -> List[SceneInfo]:
        """Detect scenes in a video file."""
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise SceneDetectionError(f"Video file not found: {video_path}")
        
        # Try to load from cache first
        cached_scenes = self._load_scenes_from_cache(video_path)
        if cached_scenes is not None:
            return cached_scenes
        
        logger.info(f"Detecting scenes in video: {video_path.name}")
        logger.info(f"Using method: {self.method}, threshold: {self.threshold}")
        
        try:
            # Create video manager and scene manager
            video_manager = VideoManager([str(video_path)])
            scene_manager = SceneManager()
            
            # Add detector
            detector = self._get_detector()
            scene_manager.add_detector(detector)
            
            # Start video manager
            video_manager.start()
            
            # Get video info for progress tracking
            total_frames = video_manager.get_duration().get_frames()
            fps = video_manager.get_framerate()
            
            logger.info(f"Video duration: {format_duration(video_manager.get_duration().get_seconds())}")
            logger.info(f"Processing {total_frames} frames at {fps:.2f} FPS")
            
            # Detect scenes with progress tracking
            with ProgressTracker(total_frames, "Detecting scenes", "frames") as progress:
                scene_manager.detect_scenes(
                    frame_source=video_manager,
                    callback=lambda frame_num, frame_img: progress.update(1)
                )
            
            # Get scene list
            scene_list = scene_manager.get_scene_list()
            
            # Convert to SceneInfo objects
            scenes = []
            for i, (start_time, end_time) in enumerate(scene_list):
                start_seconds = start_time.get_seconds()
                end_seconds = end_time.get_seconds()
                
                # Filter scenes by minimum length
                if end_seconds - start_seconds >= self.min_scene_length:
                    scene = SceneInfo(start_seconds, end_seconds, i + 1)
                    scenes.append(scene)
            
            logger.info(f"Detected {len(scenes)} scenes (filtered by min length {self.min_scene_length}s)")
            
            # Save to cache
            self._save_scenes_to_cache(video_path, scenes)
            
            return scenes
            
        except Exception as e:
            logger.error(f"Error during scene detection: {e}")
            raise SceneDetectionError(f"Scene detection failed: {e}")
        
        finally:
            if 'video_manager' in locals():
                video_manager.release()
    
    def extract_scenes(self, video_path: Path, scenes: List[SceneInfo], output_dir: Path) -> List[Path]:
        """Extract individual scene files from video."""
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not scenes:
            logger.warning("No scenes to extract")
            return []
        
        logger.info(f"Extracting {len(scenes)} scenes to {output_dir}")
        
        try:
            # Create scene list for ffmpeg splitter
            scene_list = []
            for scene in scenes:
                # Convert back to FrameTimecode format expected by scenedetect
                from scenedetect import FrameTimecode
                
                # Get video manager to get framerate
                video_manager = VideoManager([str(video_path)])
                video_manager.start()
                fps = video_manager.get_framerate()
                video_manager.release()
                
                start_frame = FrameTimecode(scene.start_time, fps=fps)
                end_frame = FrameTimecode(scene.end_time, fps=fps)
                scene_list.append((start_frame, end_frame))
            
            # Use ffmpeg to split video
            output_files = split_video_ffmpeg(
                input_video_path=str(video_path),
                scene_list=scene_list,
                output_file_template=str(output_dir / f"{video_path.stem}_scene_$SCENE_NUMBER.mp4"),
                video_name=video_path.stem,
                suppress_output=True
            )
            
            logger.info(f"Successfully extracted {len(output_files)} scene files")
            return [Path(f) for f in output_files]
            
        except Exception as e:
            logger.error(f"Error extracting scenes: {e}")
            raise SceneDetectionError(f"Scene extraction failed: {e}")
    
    def get_scene_thumbnails(self, video_path: Path, scenes: List[SceneInfo], output_dir: Path) -> List[Path]:
        """Extract thumbnail images for each scene."""
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        thumbnails = []
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            for scene in scenes:
                # Seek to middle of scene
                middle_time = (scene.start_time + scene.end_time) / 2
                frame_number = int(middle_time * fps)
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if ret:
                    thumbnail_path = output_dir / f"{video_path.stem}_scene_{scene.scene_number:03d}_thumb.jpg"
                    cv2.imwrite(str(thumbnail_path), frame)
                    thumbnails.append(thumbnail_path)
                else:
                    logger.warning(f"Failed to extract thumbnail for scene {scene.scene_number}")
            
            cap.release()
            logger.info(f"Generated {len(thumbnails)} thumbnails")
            return thumbnails
            
        except Exception as e:
            logger.error(f"Error generating thumbnails: {e}")
            return []


def create_scene_detector(config) -> SceneDetector:
    """Create and return a SceneDetector instance."""
    return SceneDetector(config)