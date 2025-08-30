"""
YouTube Video Downloader

Dedicated module for downloading videos from YouTube URLs.
Returns video file paths and metadata for further processing.
"""

import os
from yt_dlp import YoutubeDL


def download_video(url, output_dir='.', format_selector=None):
    """
    Download video from YouTube URL into a subdirectory named after the video.
    
    Args:
        url (str): YouTube URL to download
        output_dir (str): Base directory to save the video (default: current directory)
        format_selector (str): Format selector for yt-dlp (default: best mp4)
        
    Returns:
        list: List of tuples (file_path, title) for downloaded videos
    """
    if format_selector is None:
        format_selector = 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'
    
    # Ensure base output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # First pass: get video info to create subdirectory
    temp_ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }
    
    with YoutubeDL(temp_ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    
    downloaded_videos = []
    
    if 'entries' in info:
        # Playlist - create subdirectory for playlist
        playlist_title = info.get('title', 'Playlist').replace('/', '_').replace('\\', '_').replace(':', '_')
        video_output_dir = os.path.join(output_dir, playlist_title)
        os.makedirs(video_output_dir, exist_ok=True)
        
        # Configure yt-dlp options for playlist
        ydl_opts = {
            'format': format_selector,
            'outtmpl': os.path.join(video_output_dir, '%(title)s.%(ext)s'),
        }
        
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            for entry in info['entries']:
                filename = ydl.prepare_filename(entry)
                downloaded_videos.append((filename, entry['title']))
    else:
        # Single video - create subdirectory named after video
        video_title = info.get('title', 'Video').replace('/', '_').replace('\\', '_').replace(':', '_')
        video_output_dir = os.path.join(output_dir, video_title)
        os.makedirs(video_output_dir, exist_ok=True)
        
        # Configure yt-dlp options for single video
        ydl_opts = {
            'format': format_selector,
            'outtmpl': os.path.join(video_output_dir, '%(title)s.%(ext)s'),
        }
        
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            downloaded_videos.append((filename, info['title']))
    
    return downloaded_videos


def main():
    """CLI interface for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download videos from YouTube URLs")
    parser.add_argument('url', help='YouTube URL to download')
    parser.add_argument('-o', '--output', default='.', 
                       help='Output directory (default: current directory)')
    parser.add_argument('-f', '--format', 
                       help='Format selector (default: best mp4)')
    
    args = parser.parse_args()
    
    try:
        videos = download_video(args.url, args.output, args.format)
        print(f"Successfully downloaded {len(videos)} video(s):")
        for filepath, title in videos:
            print(f"  - {title}: {filepath}")
    except Exception as e:
        print(f"Error downloading video: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
