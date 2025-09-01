"""
Video Grouping - Similarity-Based (einzige verfügbare Methode)
Gruppiert Videos basierend auf tatsächlicher Ähnlichkeit mit CNN Features
"""

import numpy as np
import os
import json
import cv2
import subprocess
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Optional imports for advanced features
try:
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing import image
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

class VideoSegmentAnalyzer:
    """Extrahiert Features aus Video-Segmenten für Ähnlichkeitsvergleiche"""
    
    def __init__(self, method='cnn'):
        """
        Args:
            method: Feature-Extraktions-Methode ('histogram', 'orb', 'sift', 'phash', 'cnn', 'audio')
        """
        self.method = method
        self.model = None
        
        if method == 'cnn':
            if not TENSORFLOW_AVAILABLE:
                print("TensorFlow not available. Fallback to histogram method.")
                self.method = 'histogram'
            else:
                # Load ResNet50 for CNN features
                try:
                    self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
                    print(f"CNN model loaded: ResNet50")
                except Exception as e:
                    print(f"Error loading CNN model: {e}")
                    print("Fallback to histogram method.")
                    self.method = 'histogram'
        
        elif method == 'audio':
            if not LIBROSA_AVAILABLE:
                print("Librosa not available. Fallback to histogram method.")
                self.method = 'histogram'
    
    def extract_features(self, video_path: str) -> np.ndarray:
        """Extrahiert Features aus einem Video-Segment"""
        
        if self.method == 'histogram':
            return self._extract_histogram_features(video_path)
        elif self.method == 'cnn':
            return self._extract_cnn_features(video_path)
        elif self.method == 'audio':
            return self._extract_audio_features(video_path)
        else:
            # Fallback auf Histogram
            return self._extract_histogram_features(video_path)
    
    def _extract_histogram_features(self, video_path: str) -> np.ndarray:
        """Extrahiert Farb-Histogram Features"""
        cap = cv2.VideoCapture(video_path)
        
        # Mehrere Frames sampeln
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_frames = min(5, max(1, frame_count // 10))
        
        histograms = []
        
        for i in range(sample_frames):
            frame_idx = i * (frame_count // sample_frames) if sample_frames > 1 else frame_count // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            ret, frame = cap.read()
            if ret:
                # RGB konvertieren
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Histogram für jeden Kanal
                hist_r = cv2.calcHist([frame_rgb], [0], None, [64], [0, 256])
                hist_g = cv2.calcHist([frame_rgb], [1], None, [64], [0, 256])
                hist_b = cv2.calcHist([frame_rgb], [2], None, [64], [0, 256])
                
                # Normalisieren
                hist_r = hist_r.flatten() / (hist_r.sum() + 1e-7)
                hist_g = hist_g.flatten() / (hist_g.sum() + 1e-7)
                hist_b = hist_b.flatten() / (hist_b.sum() + 1e-7)
                
                histograms.extend([hist_r, hist_g, hist_b])
        
        cap.release()
        
        # Durchschnittliche Features
        if histograms:
            features = np.concatenate(histograms)
        else:
            features = np.zeros(64 * 3 * sample_frames)  # Fallback
        
        return features.astype(np.float32)
    
    def _extract_cnn_features(self, video_path: str) -> np.ndarray:
        """Extrahiert CNN Features mit ResNet50"""
        if self.model is None:
            return self._extract_histogram_features(video_path)
        
        cap = cv2.VideoCapture(video_path)
        
        # Mehrere Frames sampeln
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_frames = min(3, max(1, frame_count // 5))
        
        features_list = []
        
        for i in range(sample_frames):
            frame_idx = i * (frame_count // sample_frames) if sample_frames > 1 else frame_count // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            ret, frame = cap.read()
            if ret:
                # Für ResNet50 vorbereiten
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (224, 224))
                frame_array = np.expand_dims(frame_resized, axis=0)
                frame_preprocessed = preprocess_input(frame_array)
                
                # Features extrahieren
                features = self.model.predict(frame_preprocessed, verbose=0)
                features_list.append(features.flatten())
        
        cap.release()
        
        if features_list:
            # Durchschnittliche Features
            avg_features = np.mean(features_list, axis=0)
        else:
            avg_features = np.zeros(2048)  # ResNet50 Ausgabegröße
        
        return avg_features.astype(np.float32)
    
    def _extract_audio_features(self, video_path: str) -> np.ndarray:
        """Extrahiert Audio Features mit librosa"""
        if not LIBROSA_AVAILABLE:
            return self._extract_histogram_features(video_path)
        
        try:
            # Audio aus Video extrahieren (nur erste 30 Sekunden)
            y, sr = librosa.load(video_path, duration=30)
            
            # MFCC Features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            
            # Spectral Features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_mean = np.mean(spectral_centroids)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            rolloff_mean = np.mean(spectral_rolloff)
            
            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_mean = np.mean(zcr)
            
            # Kombiniere Features
            features = np.concatenate([
                mfcc_mean,
                [spectral_mean, rolloff_mean, zcr_mean]
            ])
            
            return features.astype(np.float32)
            
        except Exception as e:
            print(f"Error in audio feature extraction: {e}")
            return self._extract_histogram_features(video_path)


def merge_video_groups(groups: Dict[str, List[str]], output_dir: str, group_prefix: str = "group") -> Dict[str, str]:
    """
    Fügt Video-Gruppen zu finalen Videos zusammen
    
    Args:
        groups: Dictionary mit Gruppen-IDs und Video-Listen
        output_dir: Ausgabeverzeichnis
        group_prefix: Präfix für Ausgabedateien
    
    Returns:
        Dictionary mit Gruppen-IDs und finalen Video-Pfaden
    """
    os.makedirs(output_dir, exist_ok=True)
    merged_files = {}
    
    for group_id, video_list in groups.items():
        if not video_list:
            continue
        
        # Ausgabedatei erstellen
        output_filename = f"{group_prefix}_{group_id}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        if len(video_list) == 1:
            # Einzelnes Video kopieren
            try:
                subprocess.run([
                    'ffmpeg', '-y', '-i', video_list[0], 
                    '-c', 'copy', output_path
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                merged_files[group_id] = output_path
                print(f"Group {group_id}: 1 video -> {os.path.basename(output_path)}")
            except subprocess.CalledProcessError as e:
                print(f"Error copying {group_id}: {e}")
        else:
            # Mehrere Videos zusammenfügen
            try:
                # Temporäre Filelist erstellen
                filelist_path = os.path.join(output_dir, f"temp_{group_id}.txt")
                with open(filelist_path, 'w', encoding='utf-8') as f:
                    for video_path in video_list:
                        # Windows-Pfade für ffmpeg korrekt formatieren
                        abs_path = os.path.abspath(video_path)
                        # Backslashes durch Forward-Slashes ersetzen für ffmpeg
                        ffmpeg_path = abs_path.replace('\\', '/')
                        f.write(f"file '{ffmpeg_path}'\n")
                
                # Videos mit ffmpeg concatenaten - mit Fehlerausgabe für Debugging
                result = subprocess.run([
                    'ffmpeg', '-y', '-f', 'concat', '-safe', '0', 
                    '-i', filelist_path, '-c', 'copy', output_path
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"FFmpeg error for {group_id}:")
                    print(f"Stdout: {result.stdout}")
                    print(f"Stderr: {result.stderr}")
                    # Show filelist content for debugging
                    with open(filelist_path, 'r', encoding='utf-8') as f:
                        print(f"Filelist content:\n{f.read()}")
                    raise subprocess.CalledProcessError(result.returncode, result.args)
                
                # Delete temporary file
                os.remove(filelist_path)
                
                merged_files[group_id] = output_path
                print(f"Group {group_id}: {len(video_list)} videos -> {os.path.basename(output_path)}")
                
            except subprocess.CalledProcessError as e:
                print(f"Error merging {group_id}: {e}")
            except Exception as e:
                print(f"Unexpected error for {group_id}: {e}")
    
    return merged_files


class SimilarityGrouper:
    """Gruppiert Videos basierend auf Ähnlichkeits-Schwellwerten"""
    
    def __init__(self, analyzer, similarity_metric='cosine'):
        """
        Args:
            analyzer: VideoSegmentAnalyzer instance
            similarity_metric: 'cosine', 'euclidean', 'correlation'
        """
        self.analyzer = analyzer
        self.similarity_metric = similarity_metric
    
    def group_by_similarity(self, video_paths: List[str], 
                          min_similarity: float = 0.75,
                          min_group_size: int = 2,
                          orphan_threshold: float = 0.5,
                          group_expansion_mode: str = 'strict') -> Dict[str, List[str]]:
        """
        Groups videos based on similarity (Greedy algorithm)
        
        Args:
            video_paths: List of video file paths
            min_similarity: Minimum similarity for grouping (0.0 - 1.0)
            min_group_size: Minimum number of videos per group
            orphan_threshold: Threshold for orphan videos
            group_expansion_mode: 'strict' (all connections must meet threshold) or 'average' (average connection must meet threshold)
        
        Returns:
            Dictionary with group IDs and associated video lists
        """
        if len(video_paths) < 2:
            return {}
        
        print(f"Found: {len(video_paths)} video segments")
        print(f"Method: {self.analyzer.method}")
        print(f"Min similarity: {min_similarity}")
        print(f"Min group size: {min_group_size}")
        print(f"Orphan threshold: {orphan_threshold}")
        print(f"Similarity metric: {self.similarity_metric}")
        print(f"Group expansion mode: {group_expansion_mode}")
        
        # Features extrahieren
        features = []
        valid_paths = []
        
        for i, video_path in enumerate(video_paths):
            try:
                feature = self.analyzer.extract_features(video_path)
                if feature is not None and len(feature) > 0:
                    features.append(feature)
                    valid_paths.append(video_path)
                    
                    if (i + 1) % 10 == 0:
                        print(f"Features extracted: {i + 1}/{len(video_paths)}")
                        
            except Exception as e:
                print(f"Error with {video_path}: {e}")
        
        if len(features) < 2:
            print("Not enough valid features extracted")
            return {}
        
        print(f"Calculating similarity matrix...")
        
        # Ähnlichkeitsmatrix berechnen
        features_array = np.array(features)
        
        if self.similarity_metric == 'cosine':
            similarity_matrix = cosine_similarity(features_array)
        elif self.similarity_metric == 'euclidean':
            # Euclidean distance zu similarity umwandeln
            distance_matrix = euclidean_distances(features_array)
            max_dist = np.max(distance_matrix)
            similarity_matrix = 1 - (distance_matrix / max_dist)
        else:
            # Fallback auf Cosine
            similarity_matrix = cosine_similarity(features_array)
        
        # Greedy clustering with strict quality control
        print(f"Searching for groups with min. {min_similarity} similarity...")
        
        n_videos = len(valid_paths)
        used = [False] * n_videos
        groups = {}
        group_counter = 0
        
        # Collect all valid connections
        valid_connections = []
        for i in range(n_videos):
            for j in range(i + 1, n_videos):
                sim = similarity_matrix[i][j]
                if sim >= min_similarity:
                    valid_connections.append((i, j, sim))
        
        valid_connections.sort(key=lambda x: x[2], reverse=True)  # Sort by similarity
        print(f"Found: {len(valid_connections)} qualifying similarity connections")
        
        # Greedy-Gruppierung: Beginne mit bester Verbindung
        for i, j, sim in valid_connections:
            if used[i] or used[j]:
                continue
            
            # Neue Gruppe starten
            current_group = [i, j]
            used[i] = used[j] = True
            
            # Erweitere Gruppe: Finde weitere Videos mit hoher Ähnlichkeit zu Gruppen-Mitgliedern
            improved = True
            while improved:
                improved = False
                best_candidate = None
                best_avg_sim = 0
                
                for k in range(n_videos):
                    if used[k]:
                        continue
                    
                    # Berechne Durchschnitts-Ähnlichkeit zu allen Gruppen-Mitgliedern
                    similarities_to_group = [similarity_matrix[k][member] for member in current_group]
                    avg_sim = np.mean(similarities_to_group)
                    min_sim = np.min(similarities_to_group)
                    
                    # Verschiedene Bedingungen je nach Modus
                    if group_expansion_mode == 'strict':
                        # Strikte Bedingung: ALLE Verbindungen müssen >= min_similarity sein
                        condition_met = min_sim >= min_similarity
                    else:  # 'average' mode
                        # Durchschnitts-Bedingung: Durchschnittliche Ähnlichkeit muss >= min_similarity sein
                        condition_met = avg_sim >= min_similarity
                    
                    if condition_met and avg_sim > best_avg_sim:
                        best_candidate = k
                        best_avg_sim = avg_sim
                
                if best_candidate is not None:
                    current_group.append(best_candidate)
                    used[best_candidate] = True
                    improved = True
            
            # Gruppe nur hinzufügen wenn sie groß genug ist
            if len(current_group) >= min_group_size:
                group_paths = [valid_paths[idx] for idx in current_group]
                
                # Berechne Gruppen-Statistiken
                group_similarities = []
                for a in range(len(current_group)):
                    for b in range(a + 1, len(current_group)):
                        group_similarities.append(similarity_matrix[current_group[a]][current_group[b]])
                
                avg_similarity = np.mean(group_similarities)
                min_similarity_in_group = np.min(group_similarities)
                max_similarity_in_group = np.max(group_similarities)
                
                group_id = f"group_{group_counter:03d}_sim{avg_similarity:.3f}"
                groups[group_id] = group_paths
                
                print(f"  Group found: {len(group_paths)} videos, avg similarity: {avg_similarity:.3f}")
                print(f"    Similarity range: {min_similarity_in_group:.3f} - {max_similarity_in_group:.3f}")
                
                group_counter += 1
        
        # Sammle Waisen-Videos
        orphan_indices = [i for i in range(n_videos) if not used[i]]
        
        if orphan_indices:
            orphan_paths = [valid_paths[idx] for idx in orphan_indices]
            
            # Berechne durchschnittliche Ähnlichkeit der Waisen untereinander
            if len(orphan_indices) > 1:
                orphan_similarities = []
                for a in range(len(orphan_indices)):
                    for b in range(a + 1, len(orphan_indices)):
                        orphan_similarities.append(similarity_matrix[orphan_indices[a]][orphan_indices[b]])
                avg_orphan_sim = np.mean(orphan_similarities)
            else:
                avg_orphan_sim = 0.0
            
            orphan_group_id = f"group_{group_counter:03d}_orphans"
            groups[orphan_group_id] = orphan_paths
            
            print(f"{len(orphan_paths)} orphan videos found, avg similarity: {avg_orphan_sim:.3f}")
        
        # Statistiken speichern
        self.group_stats = []
        for group_id, group_paths in groups.items():
            if 'orphans' in group_id:
                quality = avg_orphan_sim if 'avg_orphan_sim' in locals() else 0.0
                self.group_stats.append({
                    'group_id': group_id,
                    'size': len(group_paths),
                    'quality': quality,
                    'type': 'orphans'
                })
            else:
                # Berechne Qualität für reguläre Gruppen
                group_indices = [valid_paths.index(path) for path in group_paths]
                group_similarities = []
                for a in range(len(group_indices)):
                    for b in range(a + 1, len(group_indices)):
                        group_similarities.append(similarity_matrix[group_indices[a]][group_indices[b]])
                
                quality = np.mean(group_similarities) if group_similarities else 0.0
                self.group_stats.append({
                    'group_id': group_id,
                    'size': len(group_paths),
                    'quality': quality,
                    'type': 'similarity'
                })
        
        # Sortiere nach Qualität
        self.group_stats.sort(key=lambda x: x['quality'], reverse=True)
        
        print("\\nGrouping completed:")
        for i, stats in enumerate(self.group_stats, 1):
            if stats['type'] == 'orphans':
                print(f"  Rank {i}: {stats['group_id']} - {stats['size']} videos, Quality: {stats['quality']:.3f} (Individual videos)")
            else:
                print(f"  Rank {i}: {stats['group_id']} - {stats['size']} videos, Quality: {stats['quality']:.3f}")
        
        return groups
    
    def get_group_statistics(self) -> List[Dict]:
        """Gibt Gruppen-Statistiken zurück"""
        return getattr(self, 'group_stats', [])


def group_videos_by_similarity(segments_dir: str, 
                             method: str = 'cnn',
                             min_similarity: float = 0.75,
                             min_group_size: int = 2,
                             orphan_threshold: float = 0.5,
                             similarity_metric: str = 'cosine',
                             group_expansion_mode: str = 'strict') -> Dict[str, str]:
    """
    Hauptfunktion für ähnlichkeitsbasierte Video-Gruppierung
    
    Args:
        segments_dir: Pfad zum Verzeichnis mit Video-Segmenten
        method: Feature-Extraktions-Methode ('cnn', 'histogram', 'audio')
        min_similarity: Minimale Ähnlichkeit für Gruppierung (0.0 - 1.0)
        min_group_size: Minimale Anzahl Videos pro Gruppe
        orphan_threshold: Schwellwert für Waisen-Videos
        similarity_metric: Ähnlichkeitsmetrik ('cosine', 'euclidean')
    
    Returns:
        Dictionary mit Dateipfaden der gruppierten Videos
    """
    # Video-Dateien finden
    video_files = []
    for file in os.listdir(segments_dir):
        if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_files.append(os.path.join(segments_dir, file))
    
    if len(video_files) < 2:
        print(f"Not enough video segments found in {segments_dir}")
        return {}
    
    # Ausgabeverzeichnis erstellen
    output_dir = os.path.join(os.path.dirname(segments_dir), 'merged_videos')
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyzer und Grouper initialisieren
    analyzer = VideoSegmentAnalyzer(method=method)
    grouper = SimilarityGrouper(analyzer, similarity_metric=similarity_metric)
    
    # Gruppierung durchführen
    groups = grouper.group_by_similarity(
        video_files, 
        min_similarity=min_similarity,
        min_group_size=min_group_size,
        orphan_threshold=orphan_threshold,
        group_expansion_mode=group_expansion_mode
    )
    
    if not groups:
        print("No groups found")
        return {}
    
    # Merge videos
    print(f"\\nMerging {len(groups)} groups...")
    merged_files = merge_video_groups(groups, output_dir, group_prefix="similarity")
    
    # Detaillierte Info speichern
    group_stats = grouper.get_group_statistics()
    
    # Konvertiere numpy float32 zu Python float für JSON-Kompatibilität
    def convert_to_json_safe(obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_safe(item) for item in obj]
        return obj
    
    group_stats_safe = convert_to_json_safe(group_stats)
    
    groups_info = {
        'method': method,
        'grouping_type': 'similarity_based',
        'min_similarity': float(min_similarity),
        'min_group_size': int(min_group_size),
        'orphan_threshold': float(orphan_threshold),
        'similarity_metric': similarity_metric,
        'groups': {k: [os.path.basename(v) for v in videos] for k, videos in groups.items()},
        'merged_files': {k: os.path.basename(v) for k, v in merged_files.items()},
        'group_statistics': group_stats_safe
    }
    
    info_path = os.path.join(output_dir, 'similarity_grouping_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(groups_info, f, indent=2, ensure_ascii=False)
    
    print(f"Detailed information saved: {info_path}")
    
    return merged_files
