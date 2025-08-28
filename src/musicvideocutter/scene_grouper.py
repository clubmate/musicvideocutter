"""Scene grouping functionality for finding and combining similar scenes."""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from collections import defaultdict

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from .scene_detector import SceneInfo
from .utils import ProgressTracker, VideoProcessingError


logger = logging.getLogger(__name__)


class SceneFeatures:
    """Container for scene visual features."""
    
    def __init__(self, scene_info: SceneInfo, features: np.ndarray, 
                 histogram: Optional[np.ndarray] = None, 
                 thumbnail_path: Optional[Path] = None):
        """Initialize scene features."""
        self.scene_info = scene_info
        self.features = features
        self.histogram = histogram
        self.thumbnail_path = thumbnail_path
        self.video_source = None  # Will be set by grouper
    
    def __repr__(self) -> str:
        return f"SceneFeatures({self.scene_info})"


class SceneGroup:
    """A group of similar scenes."""
    
    def __init__(self, group_id: int, scenes: List[SceneFeatures]):
        """Initialize scene group."""
        self.group_id = group_id
        self.scenes = scenes
        self.representative_scene = self._find_representative()
    
    def _find_representative(self) -> SceneFeatures:
        """Find the most representative scene in the group."""
        if len(self.scenes) == 1:
            return self.scenes[0]
        
        # Calculate centroid of features
        features_matrix = np.array([scene.features for scene in self.scenes])
        centroid = np.mean(features_matrix, axis=0)
        
        # Find scene closest to centroid
        distances = [np.linalg.norm(scene.features - centroid) for scene in self.scenes]
        representative_idx = np.argmin(distances)
        
        return self.scenes[representative_idx]
    
    def get_total_duration(self) -> float:
        """Get total duration of all scenes in group."""
        return sum(scene.scene_info.duration for scene in self.scenes)
    
    def __repr__(self) -> str:
        return f"SceneGroup({self.group_id}: {len(self.scenes)} scenes, {self.get_total_duration():.1f}s)"


class SceneGrouper:
    """Handles grouping of similar scenes."""
    
    def __init__(self, config):
        """Initialize scene grouper with configuration."""
        self.config = config
        self.similarity_threshold = config.get('scene_grouping.similarity_threshold', 0.7)
        self.max_groups = config.get('scene_grouping.max_groups', 10)
        self.cross_video_grouping = config.get('scene_grouping.cross_video_grouping', False)
        
        # Feature extraction parameters
        self.thumbnail_size = (224, 224)  # Standard size for feature extraction
        self.histogram_bins = 64
    
    def extract_visual_features(self, video_path: Path, scenes: List[SceneInfo]) -> List[SceneFeatures]:
        """Extract visual features from scenes."""
        video_path = Path(video_path)
        scene_features = []
        
        logger.info(f"Extracting visual features from {len(scenes)} scenes")
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            with ProgressTracker(len(scenes), "Extracting features", "scenes") as progress:
                for scene in scenes:
                    try:
                        # Extract frame from middle of scene
                        middle_time = (scene.start_time + scene.end_time) / 2
                        frame_number = int(middle_time * fps)
                        
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                        ret, frame = cap.read()
                        
                        if not ret:
                            logger.warning(f"Failed to extract frame for scene {scene.scene_number}")
                            continue
                        
                        # Extract features
                        features = self._extract_frame_features(frame)
                        histogram = self._extract_color_histogram(frame)
                        
                        scene_feature = SceneFeatures(scene, features, histogram)
                        scene_feature.video_source = video_path.stem
                        scene_features.append(scene_feature)
                        
                        progress.update(1)
                        
                    except Exception as e:
                        logger.warning(f"Error processing scene {scene.scene_number}: {e}")
                        continue
            
            cap.release()
            logger.info(f"Extracted features from {len(scene_features)} scenes")
            return scene_features
            
        except Exception as e:
            logger.error(f"Error extracting visual features: {e}")
            raise VideoProcessingError(f"Feature extraction failed: {e}")
    
    def _extract_frame_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract visual features from a single frame."""
        # Resize frame
        frame_resized = cv2.resize(frame, self.thumbnail_size)
        
        # Convert to different color spaces for richer features
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
        
        # Extract different types of features
        features = []
        
        # 1. Basic statistics
        features.extend([
            np.mean(gray), np.std(gray),
            np.mean(frame_resized), np.std(frame_resized)
        ])
        
        # 2. Edge density (using Canny edge detection)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)
        
        # 3. Texture features (using Local Binary Pattern approximation)
        # Simplified LBP - check neighboring pixels
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        texture_response = cv2.filter2D(gray, -1, kernel)
        features.extend([np.mean(texture_response), np.std(texture_response)])
        
        # 4. Color distribution in HSV
        for channel in range(3):
            features.extend([np.mean(hsv[:, :, channel]), np.std(hsv[:, :, channel])])
        
        # 5. Dominant colors (simplified - just sample key points)
        h, w = frame_resized.shape[:2]
        sample_points = [
            frame_resized[h//4, w//4],
            frame_resized[h//4, 3*w//4],
            frame_resized[3*h//4, w//4],
            frame_resized[3*h//4, 3*w//4],
            frame_resized[h//2, w//2]
        ]
        for point in sample_points:
            features.extend(point.astype(float))
        
        return np.array(features, dtype=np.float32)
    
    def _extract_color_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Extract color histogram from frame."""
        # Calculate histogram for each channel
        histograms = []
        for i in range(3):  # BGR channels
            hist = cv2.calcHist([frame], [i], None, [self.histogram_bins], [0, 256])
            histograms.append(hist.flatten())
        
        # Concatenate all histograms
        combined_hist = np.concatenate(histograms)
        
        # Normalize
        combined_hist = combined_hist / (np.sum(combined_hist) + 1e-7)
        
        return combined_hist
    
    def calculate_similarity_matrix(self, scene_features: List[SceneFeatures]) -> np.ndarray:
        """Calculate similarity matrix between all scenes."""
        n_scenes = len(scene_features)
        similarity_matrix = np.zeros((n_scenes, n_scenes))
        
        logger.info(f"Calculating similarity matrix for {n_scenes} scenes")
        
        # Extract feature matrices
        features_matrix = np.array([sf.features for sf in scene_features])
        histograms_matrix = np.array([sf.histogram for sf in scene_features if sf.histogram is not None])
        
        # Calculate feature similarities
        feature_similarity = cosine_similarity(features_matrix)
        
        # Calculate histogram similarities if available
        if histograms_matrix.shape[0] == n_scenes:
            histogram_similarity = cosine_similarity(histograms_matrix)
            # Combine both similarities
            similarity_matrix = 0.7 * feature_similarity + 0.3 * histogram_similarity
        else:
            similarity_matrix = feature_similarity
        
        return similarity_matrix
    
    def group_scenes_by_similarity(self, scene_features: List[SceneFeatures]) -> List[SceneGroup]:
        """Group scenes by visual similarity using clustering."""
        if len(scene_features) < 2:
            # Single scene - create one group
            return [SceneGroup(1, scene_features)]
        
        logger.info(f"Grouping {len(scene_features)} scenes by similarity")
        
        # Calculate similarity matrix
        similarity_matrix = self.calculate_similarity_matrix(scene_features)
        
        # Convert similarity to distance matrix
        distance_matrix = 1 - similarity_matrix
        
        # Determine optimal number of clusters
        n_clusters = min(self.max_groups, len(scene_features))
        
        # Use KMeans clustering on the features
        features_matrix = np.array([sf.features for sf in scene_features])
        
        if n_clusters == 1:
            labels = np.zeros(len(scene_features), dtype=int)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_matrix)
        
        # Create scene groups
        groups_dict = defaultdict(list)
        for i, label in enumerate(labels):
            groups_dict[label].append(scene_features[i])
        
        # Convert to SceneGroup objects
        groups = []
        for group_id, scenes in groups_dict.items():
            group = SceneGroup(group_id + 1, scenes)  # 1-indexed group IDs
            groups.append(group)
        
        # Sort groups by total duration (descending)
        groups.sort(key=lambda g: g.get_total_duration(), reverse=True)
        
        # Re-assign group IDs based on sorted order
        for i, group in enumerate(groups):
            group.group_id = i + 1
        
        logger.info(f"Created {len(groups)} scene groups")
        for group in groups:
            logger.info(f"  {group}")
        
        return groups
    
    def group_scenes_across_videos(self, all_scene_features: Dict[str, List[SceneFeatures]]) -> List[SceneGroup]:
        """Group scenes across multiple videos."""
        if not self.cross_video_grouping:
            logger.info("Cross-video grouping is disabled")
            return []
        
        # Flatten all scene features
        all_features = []
        for video_name, features in all_scene_features.items():
            all_features.extend(features)
        
        if not all_features:
            return []
        
        logger.info(f"Grouping scenes across {len(all_scene_features)} videos")
        logger.info(f"Total scenes to group: {len(all_features)}")
        
        return self.group_scenes_by_similarity(all_features)
    
    def refine_groups_by_threshold(self, groups: List[SceneGroup]) -> List[SceneGroup]:
        """Refine groups by splitting those with low internal similarity."""
        refined_groups = []
        
        for group in groups:
            if len(group.scenes) < 2:
                refined_groups.append(group)
                continue
            
            # Calculate internal similarity
            features_matrix = np.array([scene.features for scene in group.scenes])
            similarity_matrix = cosine_similarity(features_matrix)
            
            # Get average similarity (excluding diagonal)
            n = len(group.scenes)
            total_similarity = np.sum(similarity_matrix) - n  # Exclude diagonal
            avg_similarity = total_similarity / (n * (n - 1))
            
            if avg_similarity >= self.similarity_threshold:
                # Group is cohesive
                refined_groups.append(group)
            else:
                # Split group further
                logger.info(f"Splitting group {group.group_id} (avg similarity: {avg_similarity:.3f})")
                
                # Re-cluster this group with more clusters
                n_sub_clusters = min(3, len(group.scenes))
                if n_sub_clusters > 1:
                    kmeans = KMeans(n_clusters=n_sub_clusters, random_state=42, n_init=10)
                    sub_labels = kmeans.fit_predict(features_matrix)
                    
                    # Create sub-groups
                    sub_groups_dict = defaultdict(list)
                    for i, label in enumerate(sub_labels):
                        sub_groups_dict[label].append(group.scenes[i])
                    
                    for sub_scenes in sub_groups_dict.values():
                        if sub_scenes:  # Only add non-empty groups
                            new_group_id = len(refined_groups) + 1
                            refined_groups.append(SceneGroup(new_group_id, sub_scenes))
                else:
                    refined_groups.append(group)
        
        logger.info(f"Refined to {len(refined_groups)} groups")
        return refined_groups
    
    def save_grouping_results(self, groups: List[SceneGroup], output_path: Path) -> None:
        """Save grouping results to JSON file."""
        results = {
            'config': {
                'similarity_threshold': self.similarity_threshold,
                'max_groups': self.max_groups,
                'cross_video_grouping': self.cross_video_grouping
            },
            'groups': []
        }
        
        for group in groups:
            group_data = {
                'group_id': group.group_id,
                'scene_count': len(group.scenes),
                'total_duration': group.get_total_duration(),
                'representative_scene': {
                    'video_source': group.representative_scene.video_source,
                    'scene_number': group.representative_scene.scene_info.scene_number,
                    'start_time': group.representative_scene.scene_info.start_time,
                    'end_time': group.representative_scene.scene_info.end_time
                },
                'scenes': []
            }
            
            for scene_feature in group.scenes:
                scene_data = {
                    'video_source': scene_feature.video_source,
                    'scene_info': scene_feature.scene_info.to_dict()
                }
                group_data['scenes'].append(scene_data)
            
            results['groups'].append(group_data)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Grouping results saved to: {output_path}")


def create_scene_grouper(config) -> SceneGrouper:
    """Create and return a SceneGrouper instance."""
    return SceneGrouper(config)