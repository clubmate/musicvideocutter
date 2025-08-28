"""Utility functions for Music Video Cutter."""

import logging
import sys
from pathlib import Path
from typing import Optional
from tqdm import tqdm


def setup_logging(config) -> logging.Logger:
    """Set up logging based on configuration."""
    log_level = config.get('logging.level', 'INFO')
    log_file = config.get('logging.log_file', 'musicvideocutter.log')
    verbose = config.get('logging.verbose', False)
    
    # Configure logging
    logger = logging.getLogger('musicvideocutter')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_format = '%(levelname)s: %(message)s'
    if verbose:
        console_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    console_handler.setFormatter(logging.Formatter(console_format))
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        file_handler.setFormatter(logging.Formatter(file_format))
        logger.addHandler(file_handler)
    
    return logger


class ProgressTracker:
    """Progress tracking utility with tqdm integration."""
    
    def __init__(self, total: int, description: str = "Processing", unit: str = "items"):
        """Initialize progress tracker."""
        self.total = total
        self.description = description
        self.unit = unit
        self.progress_bar = None
        self.current = 0
    
    def __enter__(self):
        """Enter context manager."""
        self.progress_bar = tqdm(
            total=self.total,
            desc=self.description,
            unit=self.unit,
            ncols=80
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if self.progress_bar:
            self.progress_bar.close()
    
    def update(self, increment: int = 1, description: Optional[str] = None):
        """Update progress."""
        if self.progress_bar:
            self.progress_bar.update(increment)
            if description:
                self.progress_bar.set_description(description)
        self.current += increment
    
    def set_description(self, description: str):
        """Set progress description."""
        if self.progress_bar:
            self.progress_bar.set_description(description)
    
    def set_postfix(self, **kwargs):
        """Set progress postfix information."""
        if self.progress_bar:
            self.progress_bar.set_postfix(**kwargs)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing invalid characters."""
    import re
    
    # Remove invalid characters for file names
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove multiple consecutive spaces/underscores
    filename = re.sub(r'[_\s]+', '_', filename)
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    
    return filename or "untitled"


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
    else:
        return f"{minutes:02d}:{seconds:05.2f}"


def format_size(bytes_size: int) -> str:
    """Format file size in bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"


def ensure_directory(path: Path) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_video_info_string(filepath: Path, duration: float, size: int) -> str:
    """Generate a formatted string with video information."""
    return (
        f"Video: {filepath.name}\n"
        f"Duration: {format_duration(duration)}\n"
        f"Size: {format_size(size)}\n"
        f"Path: {filepath}"
    )


def validate_video_file(filepath: Path) -> bool:
    """Validate if file is a supported video format."""
    if not filepath.exists():
        return False
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    return filepath.suffix.lower() in video_extensions


class VideoProcessingError(Exception):
    """Custom exception for video processing errors."""
    pass


class SceneDetectionError(Exception):
    """Custom exception for scene detection errors."""
    pass


class DownloadError(Exception):
    """Custom exception for download errors."""
    pass