#!/usr/bin/env python3
"""
Music Video Cutter - Scene Detection and Similarity-based Grouping
Erkennt und schneidet Szenen aus Musikvideos und gruppiert ähnliche Szenen basierend auf visueller Ähnlichkeit.
"""

import argparse
import os
import yaml
from src.scene_detection import detect_and_split
from src.downloader import download_video

def load_config():
    """Lädt die Konfiguration aus config.yaml"""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    parser = argparse.ArgumentParser(description="Detect and cut scenes from music videos.")
    parser.add_argument('input', help='YouTube URL oder lokaler Dateipfad')
    parser.add_argument('--no-group', action='store_true', help='Überspringe Gruppierung ähnlicher Szenen (Standard: Gruppierung aktiviert)')
    parser.add_argument('--group-method', default=None, 
                       choices=['histogram', 'orb', 'sift', 'phash', 'cnn', 'audio'],
                       help='Methode für Ähnlichkeitserkennung (Standard: aus config.yaml)')
    
    # Ähnlichkeitsbasierte Gruppierung (standardmäßig aktiviert)
    parser.add_argument('--min-similarity', type=float, default=None,
                       help='Mindest-Ähnlichkeit für Gruppierung (0-1, Standard: aus config.yaml)')
    parser.add_argument('--min-group-size', type=int, default=None,
                       help='Mindestanzahl Videos pro Gruppe (Standard: aus config.yaml)')
    parser.add_argument('--orphan-threshold', type=float, default=None,
                       help='Schwellwert für Einzelvideos (Standard: aus config.yaml)')
    parser.add_argument('--similarity-metric', default=None,
                       choices=['cosine', 'euclidean', 'correlation'],
                       help='Ähnlichkeitsmetrik (Standard: aus config.yaml)')
    parser.add_argument('--group-expansion-mode', default=None,
                       choices=['strict', 'average'],
                       help='Gruppenerweiterungslogik (Standard: aus config.yaml)')
    
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
                output_base = title.replace('/', '_').replace('\\\\', '_').replace(':', '_')
                output_dir = output_base
                temp_dir = os.path.join(output_dir, config['output']['temp_dir'])
                merged_dir = os.path.join(output_dir, config['output']['merged_dir'])

        print(f"Processing {title} (Detection Method: {config['scene_detection']['method']})")
        scenes = detect_and_split(video_path, temp_dir, config)
        print(f"Detected and split video into {len(scenes)} segments")
        
        # Gruppierung der Segmente (standardmäßig aktiviert, außer wenn --no-group verwendet wird)
        if not args.no_group:
            grouping_config = config.get('grouping', {})
            
            # Parameter aus config.yaml laden, mit Command-Line-Overrides
            method = args.group_method or grouping_config.get('method', 'cnn')
            min_similarity = args.min_similarity or grouping_config.get('min_similarity', 0.75)
            min_group_size = args.min_group_size or grouping_config.get('min_group_size', 2)
            orphan_threshold = args.orphan_threshold or grouping_config.get('orphan_threshold', 0.5)
            similarity_metric = args.similarity_metric or grouping_config.get('similarity_metric', 'cosine')
            group_expansion_mode = args.group_expansion_mode or grouping_config.get('group_expansion_mode', 'strict')
            
            print(f"Similarity-based grouping:")
            print(f"  Method: {method}")
            print(f"  Min similarity: {min_similarity:.2f}")
            print(f"  Min group size: {min_group_size}")
            print(f"  Orphan threshold: {orphan_threshold:.2f}")
            print(f"  Similarity metric: {similarity_metric}")
            print(f"  Group expansion mode: {group_expansion_mode}")
            
            from src.grouping import group_videos_by_similarity
            
            try:
                merged_files = group_videos_by_similarity(
                    segments_dir=temp_dir,
                    method=method,
                    min_similarity=min_similarity,
                    min_group_size=min_group_size,
                    orphan_threshold=orphan_threshold,
                    similarity_metric=similarity_metric,
                    group_expansion_mode=group_expansion_mode
                )
                print(f"Similarity-based grouping completed. {len(merged_files)} final videos created.")
                for group_name, video_path in merged_files.items():
                    print(f"  {group_name}: {os.path.basename(video_path)}")
                    
            except Exception as e:
                print(f"Error in similarity-based grouping: {e}")
                print("Tip: Install additional packages for advanced features:")
                if method in ['cnn']:
                    print("     pip install tensorflow")
                if method == 'audio':
                    print("     pip install librosa")
                print("     Or use --group-method histogram as fallback")

        print(f"Done processing {title}")

        print(f"Done processing {title}")

if __name__ == "__main__":
    main()
