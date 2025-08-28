"""Video processing functionality for cutting and merging scenes."""

import logging
from pathlib import Path
from typing import List, Optional
import tempfile
import shutil

from moviepy import VideoFileClip, concatenate_videoclips, CompositeVideoClip
from moviepy.video.fx import FadeIn as fadein, FadeOut as fadeout

from .scene_detector import SceneInfo
from .scene_grouper import SceneGroup
from .utils import VideoProcessingError, ProgressTracker, format_duration, sanitize_filename


logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handles video cutting and merging operations."""
    
    def __init__(self, config):
        """Initialize video processor with configuration."""
        self.config = config
        self.transition_effect = config.get('video_processing.transition_effect', 'fade')
        self.transition_duration = config.get('video_processing.transition_duration', 0.5)
        self.output_quality = config.get('video_processing.output_quality', 'high')
        self.cleanup_temp = config.get('advanced.cleanup_temp', True)
        
        # Quality settings
        self.quality_settings = {
            'low': {'bitrate': '1M', 'audio_bitrate': '128k'},
            'medium': {'bitrate': '3M', 'audio_bitrate': '192k'},
            'high': {'bitrate': '8M', 'audio_bitrate': '320k'}
        }
    
    def extract_scene_clips(self, video_path: Path, scenes: List[SceneInfo], output_dir: Path) -> List[Path]:
        """Extract individual scene clips from video."""
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not scenes:
            logger.warning("No scenes to extract")
            return []
        
        logger.info(f"Extracting {len(scenes)} scene clips from {video_path.name}")
        
        extracted_files = []
        
        try:
            with VideoFileClip(str(video_path)) as main_clip:
                with ProgressTracker(len(scenes), "Extracting scenes", "clips") as progress:
                    for scene in scenes:
                        try:
                            # Extract scene clip
                            scene_clip = main_clip.subclip(scene.start_time, scene.end_time)
                            
                            # Generate output filename
                            scene_filename = f"{video_path.stem}_scene_{scene.scene_number:03d}.mp4"
                            scene_output_path = output_dir / scene_filename
                            
                            # Write scene clip
                            quality_settings = self.quality_settings[self.output_quality]
                            scene_clip.write_videofile(
                                str(scene_output_path),
                                bitrate=quality_settings['bitrate'],
                                audio_bitrate=quality_settings['audio_bitrate'],
                                verbose=False,
                                logger=None
                            )
                            
                            scene_clip.close()
                            extracted_files.append(scene_output_path)
                            
                            logger.debug(f"Extracted scene {scene.scene_number}: {format_duration(scene.duration)}")
                            progress.update(1)
                            
                        except Exception as e:
                            logger.error(f"Error extracting scene {scene.scene_number}: {e}")
                            continue
            
            logger.info(f"Successfully extracted {len(extracted_files)} scene clips")
            return extracted_files
            
        except Exception as e:
            logger.error(f"Error during scene extraction: {e}")
            raise VideoProcessingError(f"Scene extraction failed: {e}")
    
    def apply_transition_effect(self, clips: List[VideoFileClip]) -> List[VideoFileClip]:
        """Apply transition effects to a list of clips."""
        if len(clips) < 2 or self.transition_effect == 'hard_cut':
            return clips
        
        processed_clips = []
        
        for i, clip in enumerate(clips):
            processed_clip = clip
            
            if self.transition_effect == 'fade':
                # Apply fade in to all clips except first
                if i > 0:
                    processed_clip = fadein(processed_clip, self.transition_duration)
                
                # Apply fade out to all clips except last
                if i < len(clips) - 1:
                    processed_clip = fadeout(processed_clip, self.transition_duration)
            
            elif self.transition_effect == 'dissolve':
                # For dissolve, we need to handle overlapping clips
                # This is more complex and would require custom implementation
                # For now, fall back to fade
                if i > 0:
                    processed_clip = fadein(processed_clip, self.transition_duration)
                if i < len(clips) - 1:
                    processed_clip = fadeout(processed_clip, self.transition_duration)
            
            processed_clips.append(processed_clip)
        
        return processed_clips
    
    def merge_scene_group(self, scene_group: SceneGroup, video_paths: dict, output_path: Path) -> Path:
        """Merge all scenes in a group into a single video."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not scene_group.scenes:
            raise VideoProcessingError("No scenes in group to merge")
        
        logger.info(f"Merging group {scene_group.group_id} with {len(scene_group.scenes)} scenes")
        
        clips = []
        temp_files = []
        
        try:
            # Load and prepare clips
            with ProgressTracker(len(scene_group.scenes), f"Loading group {scene_group.group_id}", "scenes") as progress:
                for scene_feature in scene_group.scenes:
                    video_source = scene_feature.video_source
                    scene_info = scene_feature.scene_info
                    
                    if video_source not in video_paths:
                        logger.warning(f"Video source not found: {video_source}")
                        continue
                    
                    video_path = video_paths[video_source]
                    
                    try:
                        # Load main video and extract scene
                        main_clip = VideoFileClip(str(video_path))
                        scene_clip = main_clip.subclip(scene_info.start_time, scene_info.end_time)
                        
                        clips.append(scene_clip)
                        temp_files.append(main_clip)  # Keep reference for cleanup
                        
                        logger.debug(f"Loaded scene {scene_info.scene_number} from {video_source}")
                        progress.update(1)
                        
                    except Exception as e:
                        logger.warning(f"Error loading scene {scene_info.scene_number} from {video_source}: {e}")
                        continue
            
            if not clips:
                raise VideoProcessingError("No valid clips found for merging")
            
            # Apply transition effects
            logger.info(f"Applying {self.transition_effect} transitions")
            processed_clips = self.apply_transition_effect(clips)
            
            # Concatenate clips
            logger.info("Concatenating clips")
            final_clip = concatenate_videoclips(processed_clips)
            
            # Write final video
            logger.info(f"Writing merged video: {output_path.name}")
            quality_settings = self.quality_settings[self.output_quality]
            final_clip.write_videofile(
                str(output_path),
                bitrate=quality_settings['bitrate'],
                audio_bitrate=quality_settings['audio_bitrate'],
                verbose=False,
                logger=None
            )
            
            # Cleanup
            final_clip.close()
            for clip in clips:
                clip.close()
            for temp_clip in temp_files:
                temp_clip.close()
            
            duration = scene_group.get_total_duration()
            logger.info(f"Successfully merged group {scene_group.group_id}: {format_duration(duration)}")
            
            return output_path
            
        except Exception as e:
            # Cleanup on error
            for clip in clips:
                try:
                    clip.close()
                except:
                    pass
            for temp_clip in temp_files:
                try:
                    temp_clip.close()
                except:
                    pass
            
            logger.error(f"Error merging scene group {scene_group.group_id}: {e}")
            raise VideoProcessingError(f"Scene group merging failed: {e}")
    
    def create_montage_video(self, scene_groups: List[SceneGroup], video_paths: dict, output_path: Path) -> Path:
        """Create a montage video with representative scenes from each group."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating montage video with {len(scene_groups)} representative scenes")
        
        clips = []
        temp_files = []
        
        try:
            with ProgressTracker(len(scene_groups), "Creating montage", "groups") as progress:
                for group in scene_groups:
                    representative = group.representative_scene
                    video_source = representative.video_source
                    scene_info = representative.scene_info
                    
                    if video_source not in video_paths:
                        logger.warning(f"Video source not found for group {group.group_id}: {video_source}")
                        continue
                    
                    video_path = video_paths[video_source]
                    
                    try:
                        # Load clip for representative scene
                        main_clip = VideoFileClip(str(video_path))
                        
                        # Use a shorter clip for montage (max 10 seconds or full scene if shorter)
                        scene_duration = scene_info.duration
                        clip_duration = min(scene_duration, 10.0)
                        
                        # Extract from the beginning of the scene
                        scene_clip = main_clip.subclip(
                            scene_info.start_time,
                            scene_info.start_time + clip_duration
                        )
                        
                        clips.append(scene_clip)
                        temp_files.append(main_clip)
                        
                        logger.debug(f"Added representative scene from group {group.group_id}")
                        progress.update(1)
                        
                    except Exception as e:
                        logger.warning(f"Error loading representative scene for group {group.group_id}: {e}")
                        continue
            
            if not clips:
                raise VideoProcessingError("No valid clips found for montage")
            
            # Apply transitions
            processed_clips = self.apply_transition_effect(clips)
            
            # Create final montage
            montage_clip = concatenate_videoclips(processed_clips)
            
            # Write montage video
            quality_settings = self.quality_settings[self.output_quality]
            montage_clip.write_videofile(
                str(output_path),
                bitrate=quality_settings['bitrate'],
                audio_bitrate=quality_settings['audio_bitrate'],
                verbose=False,
                logger=None
            )
            
            # Cleanup
            montage_clip.close()
            for clip in clips:
                clip.close()
            for temp_clip in temp_files:
                temp_clip.close()
            
            logger.info(f"Successfully created montage video: {output_path.name}")
            return output_path
            
        except Exception as e:
            # Cleanup on error
            for clip in clips:
                try:
                    clip.close()
                except:
                    pass
            for temp_clip in temp_files:
                try:
                    temp_clip.close()
                except:
                    pass
            
            logger.error(f"Error creating montage video: {e}")
            raise VideoProcessingError(f"Montage creation failed: {e}")
    
    def process_all_groups(self, scene_groups: List[SceneGroup], video_paths: dict, output_dir: Path) -> List[Path]:
        """Process all scene groups and create merged videos."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processed_videos = []
        
        logger.info(f"Processing {len(scene_groups)} scene groups")
        
        for group in scene_groups:
            try:
                # Generate output filename
                group_name = f"group_{group.group_id:02d}_{len(group.scenes)}scenes"
                output_path = output_dir / f"{group_name}.mp4"
                
                # Merge group
                merged_video = self.merge_scene_group(group, video_paths, output_path)
                processed_videos.append(merged_video)
                
            except Exception as e:
                logger.error(f"Failed to process group {group.group_id}: {e}")
                continue
        
        # Create montage if we have multiple groups
        if len(scene_groups) > 1:
            try:
                montage_path = output_dir / "montage_all_groups.mp4"
                montage_video = self.create_montage_video(scene_groups, video_paths, montage_path)
                processed_videos.append(montage_video)
            except Exception as e:
                logger.error(f"Failed to create montage video: {e}")
        
        logger.info(f"Successfully processed {len(processed_videos)} videos")
        return processed_videos


def create_video_processor(config) -> VideoProcessor:
    """Create and return a VideoProcessor instance."""
    return VideoProcessor(config)