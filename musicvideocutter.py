import os
import argparse
import yaml
import scenedetect  # kept for detector creation
from src.scene_detection import detect_and_split
from src.downloader import download_video

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

## Scene detection logic moved to src/scene_detection.py


def main():
    config = load_config()
    # (logging removed per request)

    parser = argparse.ArgumentParser(description="Detect and cut scenes from music videos.")
    parser.add_argument('input', help='YouTube URL oder lokaler Dateipfad')
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

        print(f"Processing {title} (Detection Method: {config['scene_detection']['method']})")
        scenes = detect_and_split(video_path, temp_dir, config)
        print(f"Detected and split video into {len(scenes)} segments")
    print(f"Done processing {title}")

if __name__ == "__main__":
    main()
