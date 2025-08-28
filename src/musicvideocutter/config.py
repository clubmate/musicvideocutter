"""Configuration management for Music Video Cutter."""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration manager that handles YAML/JSON config files and CLI overrides."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration with default values."""
        self.config = self._load_default_config()
        
        if config_path:
            self.load_config_file(config_path)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration values."""
        return {
            'output': {
                'base_directory': './outputs',
                'video_format': 'mp4'
            },
            'scene_detection': {
                'method': 'adaptive',
                'threshold': 30.0,
                'min_scene_length': 1.0
            },
            'scene_grouping': {
                'similarity_threshold': 0.7,
                'max_groups': 10,
                'cross_video_grouping': False
            },
            'video_processing': {
                'transition_effect': 'fade',
                'transition_duration': 0.5,
                'output_quality': 'high'
            },
            'download': {
                'quality': '720p',
                'audio_quality': '128k',
                'format': 'mp4'
            },
            'logging': {
                'level': 'INFO',
                'log_file': 'musicvideocutter.log',
                'verbose': False
            },
            'advanced': {
                'max_workers': 4,
                'cache_scenes': True,
                'cleanup_temp': True
            }
        }
    
    def load_config_file(self, config_path: str) -> None:
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    file_config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    file_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
                
                if file_config:
                    self._merge_config(file_config)
                    
        except Exception as e:
            raise ValueError(f"Error loading configuration file: {e}")
    
    def _merge_config(self, new_config: Dict[str, Any]) -> None:
        """Merge new configuration with existing configuration."""
        def merge_dict(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
            """Recursively merge dictionaries."""
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
            return base
        
        merge_dict(self.config, new_config)
    
    def override(self, path: str, value: Any) -> None:
        """Override a configuration value using dot notation (e.g., 'output.base_directory')."""
        keys = path.split('.')
        current = self.config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation."""
        keys = path.split('.')
        current = self.config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def get_output_dir(self, video_name: str) -> Path:
        """Get the output directory for a specific video."""
        base_dir = Path(self.get('output.base_directory'))
        video_dir = base_dir / video_name
        video_dir.mkdir(parents=True, exist_ok=True)
        return video_dir
    
    def save_config(self, output_path: str) -> None:
        """Save current configuration to a file."""
        output_path = Path(output_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if output_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            elif output_path.suffix.lower() == '.json':
                json.dump(self.config, f, indent=2)
            else:
                raise ValueError(f"Unsupported output format: {output_path.suffix}")
    
    def validate(self) -> None:
        """Validate configuration values."""
        # Validate scene detection threshold
        threshold = self.get('scene_detection.threshold')
        if not (0 <= threshold <= 100):
            raise ValueError("Scene detection threshold must be between 0 and 100")
        
        # Validate similarity threshold
        sim_threshold = self.get('scene_grouping.similarity_threshold')
        if not (0 <= sim_threshold <= 1):
            raise ValueError("Similarity threshold must be between 0 and 1")
        
        # Validate transition duration
        transition_duration = self.get('video_processing.transition_duration')
        if transition_duration < 0:
            raise ValueError("Transition duration must be non-negative")
        
        # Validate max workers
        max_workers = self.get('advanced.max_workers')
        if max_workers < 1:
            raise ValueError("Max workers must be at least 1")
    
    def __str__(self) -> str:
        """Return string representation of configuration."""
        return yaml.dump(self.config, default_flow_style=False, indent=2)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or use defaults."""
    if config_path is None:
        # Try to find default config file
        default_paths = [
            'config.yaml',
            'config.yml',
            'configs/default_config.yaml',
            os.path.expanduser('~/.musicvideocutter/config.yaml')
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    return Config(config_path)