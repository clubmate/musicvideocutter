"""Main CLI interface for Music Video Cutter."""

import sys
import logging
from pathlib import Path
from typing import Optional

import click

from .config import load_config
from .utils import setup_logging, VideoProcessingError, SceneDetectionError, DownloadError
from .video_downloader import create_downloader
from .scene_detector import create_scene_detector
from .scene_grouper import create_scene_grouper
from .video_processor import create_video_processor


logger = logging.getLogger(__name__)


def print_banner():
    """Print application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    Music Video Cutter                       ║
║              Cut and Merge Music Video Scenes               ║
╚══════════════════════════════════════════════════════════════╝
"""
    click.echo(banner)


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--log-file', type=click.Path(), help='Log file path')
@click.pass_context
def cli(ctx, config, verbose, log_file):
    """Music Video Cutter - Cut and merge music video scenes based on similarity."""
    # Ensure that ctx.obj exists and is a dict
    ctx.ensure_object(dict)
    
    # Load configuration
    try:
        app_config = load_config(config)
        
        # Override config with CLI options
        if verbose:
            app_config.override('logging.verbose', True)
        if log_file:
            app_config.override('logging.log_file', log_file)
        
        # Validate configuration
        app_config.validate()
        
        # Setup logging
        logger = setup_logging(app_config)
        
        # Store config in context
        ctx.obj['config'] = app_config
        ctx.obj['logger'] = logger
        
    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_source')
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory')
@click.option('--method', type=click.Choice(['adaptive', 'content', 'threshold']), 
              help='Scene detection method')
@click.option('--threshold', type=float, help='Scene detection threshold (0-100)')
@click.option('--similarity', type=float, help='Scene similarity threshold (0-1)')
@click.option('--max-groups', type=int, help='Maximum number of scene groups')
@click.option('--transition', type=click.Choice(['fade', 'hard_cut', 'dissolve']), 
              help='Transition effect between scenes')
@click.option('--cross-video', is_flag=True, help='Enable cross-video scene grouping')
@click.option('--extract-only', is_flag=True, help='Only extract scenes, do not group or merge')
@click.pass_context
def process(ctx, input_source, output_dir, method, threshold, similarity, max_groups, 
           transition, cross_video, extract_only):
    """Process a video file or YouTube URL to cut and merge scenes."""
    print_banner()
    
    config = ctx.obj['config']
    logger = ctx.obj['logger']
    
    try:
        # Override config with CLI options
        if output_dir:
            config.override('output.base_directory', output_dir)
        if method:
            config.override('scene_detection.method', method)
        if threshold is not None:
            config.override('scene_detection.threshold', threshold)
        if similarity is not None:
            config.override('scene_grouping.similarity_threshold', similarity)
        if max_groups is not None:
            config.override('scene_grouping.max_groups', max_groups)
        if transition:
            config.override('video_processing.transition_effect', transition)
        if cross_video:
            config.override('scene_grouping.cross_video_grouping', True)
        
        # Re-validate configuration after overrides
        config.validate()
        
        logger.info("Starting video processing")
        logger.info(f"Input source: {input_source}")
        
        # Step 1: Download/load videos
        click.echo("🎬 Loading videos...")
        downloader = create_downloader(config)
        video_files = downloader.process_input(input_source)
        
        if not video_files:
            click.echo("❌ No video files found to process", err=True)
            return
        
        logger.info(f"Found {len(video_files)} video(s) to process")
        click.echo(f"✅ Loaded {len(video_files)} video(s)")
        
        # Step 2: Detect scenes in each video
        click.echo("🔍 Detecting scenes...")
        scene_detector = create_scene_detector(config)
        
        all_scenes = {}
        all_scene_features = {}
        video_paths = {}
        
        for video_file in video_files:
            try:
                video_name = video_file.stem
                video_paths[video_name] = video_file
                
                click.echo(f"  Processing: {video_file.name}")
                
                # Detect scenes
                scenes = scene_detector.detect_scenes(video_file)
                all_scenes[video_name] = scenes
                
                logger.info(f"Detected {len(scenes)} scenes in {video_file.name}")
                
                # Create output directory for this video
                video_output_dir = config.get_output_dir(video_name)
                
                # Extract scene clips if requested
                if extract_only:
                    click.echo(f"  Extracting {len(scenes)} scene clips...")
                    video_processor = create_video_processor(config)
                    extracted_files = video_processor.extract_scene_clips(
                        video_file, scenes, video_output_dir / "scenes"
                    )
                    click.echo(f"  ✅ Extracted {len(extracted_files)} scene clips")
                
            except SceneDetectionError as e:
                logger.error(f"Scene detection failed for {video_file.name}: {e}")
                click.echo(f"  ❌ Scene detection failed: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error processing {video_file.name}: {e}")
                click.echo(f"  ❌ Error: {e}")
                continue
        
        if extract_only:
            click.echo("✅ Scene extraction completed")
            return
        
        # Step 3: Group similar scenes
        click.echo("🎭 Grouping similar scenes...")
        scene_grouper = create_scene_grouper(config)
        
        for video_name, scenes in all_scenes.items():
            if not scenes:
                continue
            
            try:
                video_file = video_paths[video_name]
                click.echo(f"  Analyzing: {video_file.name}")
                
                # Extract visual features
                scene_features = scene_grouper.extract_visual_features(video_file, scenes)
                all_scene_features[video_name] = scene_features
                
            except Exception as e:
                logger.error(f"Feature extraction failed for {video_name}: {e}")
                continue
        
        # Group scenes (within videos and optionally across videos)
        all_groups = []
        
        # Group scenes within each video
        for video_name, scene_features in all_scene_features.items():
            if not scene_features:
                continue
            
            groups = scene_grouper.group_scenes_by_similarity(scene_features)
            all_groups.extend(groups)
            
            logger.info(f"Created {len(groups)} groups for {video_name}")
        
        # Optionally group across all videos
        if config.get('scene_grouping.cross_video_grouping') and len(all_scene_features) > 1:
            click.echo("  Cross-video grouping...")
            cross_video_groups = scene_grouper.group_scenes_across_videos(all_scene_features)
            if cross_video_groups:
                all_groups = cross_video_groups
                logger.info(f"Created {len(cross_video_groups)} cross-video groups")
        
        if not all_groups:
            click.echo("❌ No scene groups created")
            return
        
        click.echo(f"✅ Created {len(all_groups)} scene groups")
        
        # Display group summary
        for group in all_groups:
            duration = group.get_total_duration()
            click.echo(f"  Group {group.group_id}: {len(group.scenes)} scenes, {duration:.1f}s")
        
        # Step 4: Merge scene groups
        click.echo("🎞️  Merging scene groups...")
        video_processor = create_video_processor(config)
        
        # Create main output directory
        main_output_dir = Path(config.get('output.base_directory')) / "merged_videos"
        main_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process all groups
        merged_videos = video_processor.process_all_groups(all_groups, video_paths, main_output_dir)
        
        click.echo(f"✅ Created {len(merged_videos)} merged videos")
        
        # Save grouping results
        results_file = main_output_dir / "grouping_results.json"
        scene_grouper.save_grouping_results(all_groups, results_file)
        
        # Display final results
        click.echo("\n🎉 Processing completed successfully!")
        click.echo(f"📁 Output directory: {main_output_dir}")
        click.echo("📊 Results:")
        for video in merged_videos:
            click.echo(f"  📹 {video.name}")
        
    except (DownloadError, SceneDetectionError, VideoProcessingError) as e:
        logger.error(f"Processing failed: {e}")
        click.echo(f"❌ {e}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        click.echo("\n⏹️  Processing interrupted")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('config_path', type=click.Path())
@click.pass_context
def create_config(ctx, config_path):
    """Create a default configuration file."""
    config = ctx.obj['config']
    
    try:
        config.save_config(config_path)
        click.echo(f"✅ Configuration file created: {config_path}")
    except Exception as e:
        click.echo(f"❌ Error creating configuration file: {e}", err=True)


@cli.command()
@click.pass_context
def show_config(ctx):
    """Show current configuration."""
    config = ctx.obj['config']
    click.echo("Current configuration:")
    click.echo(str(config))


@cli.command()
@click.argument('input_source')
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory')
@click.pass_context
def download(ctx, input_source, output_dir):
    """Download video(s) from YouTube URL without processing."""
    config = ctx.obj['config']
    logger = ctx.obj['logger']
    
    if output_dir:
        config.override('output.base_directory', output_dir)
    
    try:
        click.echo("🎬 Downloading videos...")
        downloader = create_downloader(config)
        video_files = downloader.process_input(input_source)
        
        click.echo(f"✅ Downloaded {len(video_files)} video(s)")
        for video_file in video_files:
            click.echo(f"  📹 {video_file}")
            
    except DownloadError as e:
        click.echo(f"❌ Download failed: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()