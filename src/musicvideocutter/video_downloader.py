"""Video downloading functionality using yt-dlp."""

import logging
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

import yt_dlp

from .utils import sanitize_filename, DownloadError


logger = logging.getLogger(__name__)


class VideoDownloader:
    """Handles downloading videos from YouTube and other platforms."""
    
    def __init__(self, config):
        """Initialize video downloader with configuration."""
        self.config = config
        self.output_dir = Path(config.get('output.base_directory', './outputs'))
        self.quality = config.get('download.quality', '720p')
        self.audio_quality = config.get('download.audio_quality', '128k')
        self.format_preference = config.get('download.format', 'mp4')
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_ydl_options(self, output_path: str) -> Dict[str, Any]:
        """Get yt-dlp options based on configuration."""
        return {
            'format': f'best[height<={self.quality[:-1]}]/best',
            'outtmpl': output_path,
            'writesubtitles': False,
            'writeautomaticsub': False,
            'ignoreerrors': True,
            'no_warnings': True,
            'extractflat': False,
        }
    
    def is_youtube_url(self, url: str) -> bool:
        """Check if URL is a YouTube URL."""
        youtube_patterns = [
            r'(?:https?://)?(?:www\.)?youtube\.com/',
            r'(?:https?://)?(?:www\.)?youtu\.be/',
            r'(?:https?://)?(?:m\.)?youtube\.com/',
        ]
        return any(re.match(pattern, url) for pattern in youtube_patterns)
    
    def is_playlist_url(self, url: str) -> bool:
        """Check if URL is a YouTube playlist URL."""
        return 'playlist' in url or 'list=' in url
    
    def get_video_info(self, url: str) -> Dict[str, Any]:
        """Get video information without downloading."""
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                return info
        except Exception as e:
            logger.error(f"Error extracting video info: {e}")
            raise DownloadError(f"Failed to extract video information: {e}")
    
    def get_playlist_info(self, url: str) -> List[Dict[str, Any]]:
        """Get playlist information without downloading."""
        try:
            with yt_dlp.YoutubeDL({'quiet': True, 'extract_flat': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                if 'entries' in info:
                    return list(info['entries'])
                else:
                    # Single video, not a playlist
                    return [info]
        except Exception as e:
            logger.error(f"Error extracting playlist info: {e}")
            raise DownloadError(f"Failed to extract playlist information: {e}")
    
    def download_video(self, url: str, output_name: Optional[str] = None) -> Path:
        """Download a single video from URL."""
        if not self.is_youtube_url(url):
            raise DownloadError(f"URL is not a supported YouTube URL: {url}")
        
        try:
            # Get video info first
            info = self.get_video_info(url)
            title = info.get('title', 'Unknown Video')
            
            # Sanitize the title for filename
            if output_name:
                filename = sanitize_filename(output_name)
            else:
                filename = sanitize_filename(title)
            
            # Ensure unique filename
            output_path = self.output_dir / f"{filename}.%(ext)s"
            counter = 1
            while (self.output_dir / f"{filename}.mp4").exists():
                filename = f"{sanitize_filename(title)}_{counter}"
                output_path = self.output_dir / f"{filename}.%(ext)s"
                counter += 1
            
            # Download options
            ydl_opts = self._get_ydl_options(str(output_path))
            
            logger.info(f"Downloading video: {title}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Find the downloaded file
            downloaded_file = self.output_dir / f"{filename}.mp4"
            if not downloaded_file.exists():
                # Try different extensions
                for ext in ['webm', 'mkv', 'avi']:
                    alt_file = self.output_dir / f"{filename}.{ext}"
                    if alt_file.exists():
                        downloaded_file = alt_file
                        break
            
            if not downloaded_file.exists():
                raise DownloadError(f"Downloaded file not found: {downloaded_file}")
            
            logger.info(f"Video downloaded successfully: {downloaded_file}")
            return downloaded_file
            
        except yt_dlp.DownloadError as e:
            logger.error(f"yt-dlp download error: {e}")
            raise DownloadError(f"Failed to download video: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during download: {e}")
            raise DownloadError(f"Unexpected error: {e}")
    
    def download_playlist(self, url: str) -> List[Path]:
        """Download all videos from a YouTube playlist."""
        if not self.is_youtube_url(url):
            raise DownloadError(f"URL is not a supported YouTube URL: {url}")
        
        try:
            # Get playlist info
            playlist_info = self.get_playlist_info(url)
            
            if not playlist_info:
                raise DownloadError("No videos found in playlist")
            
            logger.info(f"Found {len(playlist_info)} videos in playlist")
            
            downloaded_files = []
            for i, entry in enumerate(playlist_info, 1):
                try:
                    video_url = entry.get('url') or f"https://www.youtube.com/watch?v={entry['id']}"
                    title = entry.get('title', f'Video_{i}')
                    
                    logger.info(f"Downloading video {i}/{len(playlist_info)}: {title}")
                    
                    # Download individual video
                    downloaded_file = self.download_video(video_url, f"playlist_{i:03d}_{title}")
                    downloaded_files.append(downloaded_file)
                    
                except Exception as e:
                    logger.warning(f"Failed to download video {i}: {e}")
                    continue
            
            logger.info(f"Downloaded {len(downloaded_files)} videos from playlist")
            return downloaded_files
            
        except Exception as e:
            logger.error(f"Error downloading playlist: {e}")
            raise DownloadError(f"Failed to download playlist: {e}")
    
    def process_input(self, input_source: str) -> List[Path]:
        """Process input source (URL, playlist, or local file) and return list of video files."""
        input_source = input_source.strip()
        
        # Check if it's a local file
        if not (input_source.startswith('http://') or input_source.startswith('https://')):
            local_path = Path(input_source)
            if local_path.exists() and local_path.is_file():
                logger.info(f"Using local file: {local_path}")
                return [local_path]
            else:
                raise DownloadError(f"Local file not found: {local_path}")
        
        # It's a URL - check if it's a playlist
        if self.is_playlist_url(input_source):
            logger.info("Detected playlist URL")
            return self.download_playlist(input_source)
        else:
            logger.info("Detected single video URL")
            return [self.download_video(input_source)]


def create_downloader(config) -> VideoDownloader:
    """Create and return a VideoDownloader instance."""
    return VideoDownloader(config)