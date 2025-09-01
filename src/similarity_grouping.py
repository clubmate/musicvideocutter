"""
Similarity-based Video Grouping - Vollständige Version
Gruppiert Videos basierend auf tatsächlicher Ähnlichkeit (einzige verfügbare Methode)
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
                print("Librosa not available. Audio-based grouping will be disabled.")
                print("TensorFlow nicht verfügbar. Fallback auf Histogram-Methode.")
                self.method = 'histogram'
            else:
                # ResNet50 für CNN-Features laden
                try:
                    self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
                    print(f"CNN-Modell geladen: ResNet50")
                except Exception as e:
                    print(f"Fehler beim Laden des CNN-Modells: {e}")
                    print("Fallback auf Histogram-Methode.")
                    self.method = 'histogram'
        
        elif method == 'audio':
            if not LIBROSA_AVAILABLE:
                print("Librosa nicht verfügbar. Fallback auf Histogram-Methode.")
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
            print(f"Fehler bei Audio-Feature-Extraktion: {e}")
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
                print(f"Gruppe {group_id}: 1 Video -> {os.path.basename(output_path)}")
            except subprocess.CalledProcessError as e:
                print(f"Fehler beim Kopieren von {group_id}: {e}")
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
                    print(f"FFmpeg Fehler bei {group_id}:")
                    print(f"Stdout: {result.stdout}")
                    print(f"Stderr: {result.stderr}")
                    # Filelist-Inhalt zur Debugging anzeigen
                    with open(filelist_path, 'r', encoding='utf-8') as f:
                        print(f"Filelist Inhalt:\n{f.read()}")
                    raise subprocess.CalledProcessError(result.returncode, result.args)
                
                # Temporäre Datei löschen
                os.remove(filelist_path)
                
                merged_files[group_id] = output_path
                print(f"Gruppe {group_id}: {len(video_list)} Videos -> {os.path.basename(output_path)}")
                
            except subprocess.CalledProcessError as e:
                print(f"Fehler beim Zusammenfügen von {group_id}: {e}")
            except Exception as e:
                print(f"Unerwarteter Fehler bei {group_id}: {e}")
    
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
                          orphan_threshold: float = 0.5) -> Dict[str, List[str]]:
        """
        Gruppiert Videos basierend auf Ähnlichkeit
        
        Args:
            video_paths: Liste der Video-Pfade
            min_similarity: Mindest-Ähnlichkeit für Gruppierung (0-1)
            min_group_size: Mindestanzahl Videos pro Gruppe
            orphan_threshold: Schwellwert für "Restgruppe" (niedriger als min_similarity)
        
        Returns:
            Dictionary mit Gruppen nach Qualität sortiert
        """
        print(f"🔍 Gruppierung basierend auf Ähnlichkeit (Schwellwert: {min_similarity:.2f})")
        
        # Features extrahieren
        features = []
        valid_paths = []
        
        for i, path in enumerate(video_paths):
            try:
                feature = self.analyzer.extract_features(path)
                features.append(feature)
                valid_paths.append(path)
                if (i + 1) % 10 == 0:
                    print(f"Features extrahiert: {i + 1}/{len(video_paths)}")
            except Exception as e:
                print(f"Fehler bei {path}: {e}")
        
        if len(features) < 2:
            return {"group_000_single": valid_paths} if valid_paths else {}
        
        features = np.array(features)
        
        # Ähnlichkeitsmatrix berechnen
        print("📊 Berechne Ähnlichkeitsmatrix...")
        similarity_matrix = self._calculate_similarity_matrix(features)
        
        # Gruppierung basierend auf Ähnlichkeit
        groups = self._find_similarity_groups(
            valid_paths, similarity_matrix, min_similarity, min_group_size
        )
        
        # Restgruppe für niedrige Ähnlichkeiten
        orphan_group = self._find_orphans(
            valid_paths, similarity_matrix, groups, orphan_threshold
        )
        
        # Nach Qualität sortieren
        sorted_groups = self._sort_groups_by_quality(
            groups, similarity_matrix, valid_paths, orphan_group
        )
        
        return sorted_groups
    
    def _calculate_similarity_matrix(self, features: np.ndarray) -> np.ndarray:
        """Berechnet Ähnlichkeitsmatrix zwischen allen Features"""
        
        if self.similarity_metric == 'cosine':
            # Cosine Similarity (0-1, höher = ähnlicher)
            similarity = cosine_similarity(features)
        elif self.similarity_metric == 'euclidean':
            # Euclidean Distance → Similarity (0-1, höher = ähnlicher)
            distances = euclidean_distances(features)
            max_dist = np.max(distances)
            similarity = 1 - (distances / max_dist) if max_dist > 0 else np.ones_like(distances)
        elif self.similarity_metric == 'correlation':
            # Pearson Correlation (-1 bis 1) → (0-1)
            correlation = np.corrcoef(features)
            similarity = (correlation + 1) / 2  # Normalisierung auf 0-1
        else:
            raise ValueError(f"Unbekannte Ähnlichkeitsmetrik: {self.similarity_metric}")
        
        return similarity
    
    def _find_similarity_groups(self, video_paths: List[str], 
                               similarity_matrix: np.ndarray,
                               min_similarity: float,
                               min_group_size: int) -> List[Tuple[List[str], float]]:
        """
        Findet Gruppen basierend auf Ähnlichkeitsschwellwert
        OPTIMIERTE VERSION: Keine transitiven Verbindungen, nur direkte Ähnlichkeiten
        
        Returns:
            Liste von (video_liste, durchschnittliche_ähnlichkeit) Tupeln
        """
        n_videos = len(video_paths)
        print(f"🔗 Suche Gruppen mit min. {min_similarity:.2f} Ähnlichkeit...")
        
        # Finde qualifizierende Verbindungen
        qualifying_pairs = []
        for i in range(n_videos):
            for j in range(i + 1, n_videos):
                if similarity_matrix[i, j] >= min_similarity:
                    qualifying_pairs.append((i, j, similarity_matrix[i, j]))
        
        print(f"📊 Gefunden: {len(qualifying_pairs)} qualifizierende Ähnlichkeitsverbindungen")
        
        # Greedy Clustering: Starte mit besten Paaren und erweitere nur wenn alle Verbindungen qualifizieren
        used_videos = set()
        groups = []
        
        # Sortiere Paare nach Ähnlichkeit (beste zuerst)
        qualifying_pairs.sort(key=lambda x: x[2], reverse=True)
        
        for video1_idx, video2_idx, similarity in qualifying_pairs:
            if video1_idx in used_videos or video2_idx in used_videos:
                continue  # Videos bereits in anderen Gruppen
            
            # Starte neue Gruppe mit diesem Paar
            group_indices = [video1_idx, video2_idx]
            group_similarities = [similarity]
            
            # Versuche Gruppe zu erweitern
            candidates = [i for i in range(n_videos) if i not in used_videos and i not in group_indices]
            
            for candidate_idx in candidates:
                # Prüfe ob Kandidat zu ALLEN Videos in der Gruppe ähnlich genug ist
                candidate_qualifies = True
                candidate_similarities = []
                
                for group_member_idx in group_indices:
                    pair_sim = similarity_matrix[candidate_idx, group_member_idx]
                    if pair_sim < min_similarity:
                        candidate_qualifies = False
                        break
                    candidate_similarities.append(pair_sim)
                
                if candidate_qualifies:
                    # Kandidat qualifiziert sich für die Gruppe
                    group_indices.append(candidate_idx)
                    group_similarities.extend(candidate_similarities)
            
            # Gruppe nur erstellen wenn sie groß genug ist
            if len(group_indices) >= min_group_size:
                group_videos = [video_paths[idx] for idx in group_indices]
                avg_similarity = np.mean(group_similarities)
                
                groups.append((group_videos, avg_similarity))
                used_videos.update(group_indices)
                
                print(f"  ✅ Gruppe gefunden: {len(group_videos)} Videos, Ø Ähnlichkeit: {avg_similarity:.3f}")
                
                # Debug: Zeige Ähnlichkeitsbereich für diese Gruppe
                min_sim_in_group = min(group_similarities)
                max_sim_in_group = max(group_similarities)
                print(f"    📊 Ähnlichkeitsbereich: {min_sim_in_group:.3f} - {max_sim_in_group:.3f}")
        
        return groups
    
    def _find_orphans(self, video_paths: List[str], 
                     similarity_matrix: np.ndarray,
                     existing_groups: List[Tuple[List[str], float]],
                     orphan_threshold: float) -> List[str]:
        """Findet Videos, die in keine Gruppe passen (Waisen)"""
        
        # Sammle alle bereits gruppierten Videos
        grouped_videos = set()
        for group_videos, _ in existing_groups:
            grouped_videos.update(group_videos)
        
        # Finde ungrupierte Videos
        orphans = []
        for video in video_paths:
            if video not in grouped_videos:
                orphans.append(video)
        
        # Prüfe ob Waisen untereinander ähnlich genug sind
        if len(orphans) >= 2:
            orphan_indices = [video_paths.index(video) for video in orphans]
            
            # Berechne durchschnittliche Ähnlichkeit unter Waisen
            orphan_similarities = []
            for i, idx1 in enumerate(orphan_indices):
                for j, idx2 in enumerate(orphan_indices):
                    if i < j:
                        orphan_similarities.append(similarity_matrix[idx1, idx2])
            
            avg_orphan_similarity = np.mean(orphan_similarities) if orphan_similarities else 0
            
            print(f"🏠 {len(orphans)} Waisen-Videos gefunden, Ø Ähnlichkeit: {avg_orphan_similarity:.3f}")
        
        return orphans
    
    def _sort_groups_by_quality(self, groups: List[Tuple[List[str], float]],
                               similarity_matrix: np.ndarray,
                               all_videos: List[str],
                               orphans: List[str]) -> Dict[str, List[str]]:
        """Sortiert Gruppen nach Qualität (Ähnlichkeit) und fügt Metadaten hinzu"""
        
        # Sortiere Gruppen nach durchschnittlicher Ähnlichkeit (höchste zuerst)
        sorted_groups = sorted(groups, key=lambda x: x[1], reverse=True)
        
        result = {}
        group_metadata = {}
        
        # Hauptgruppen (nach Qualität sortiert)
        for i, (group_videos, avg_similarity) in enumerate(sorted_groups):
            group_id = f"group_{i:03d}_sim{avg_similarity:.3f}"
            result[group_id] = group_videos
            
            # Zusätzliche Statistiken für die Gruppe
            group_indices = [all_videos.index(video) for video in group_videos]
            similarities_in_group = []
            for idx1 in group_indices:
                for idx2 in group_indices:
                    if idx1 < idx2:
                        similarities_in_group.append(similarity_matrix[idx1, idx2])
            
            group_metadata[group_id] = {
                'avg_similarity': float(avg_similarity),
                'min_similarity': float(np.min(similarities_in_group)) if similarities_in_group else 0,
                'max_similarity': float(np.max(similarities_in_group)) if similarities_in_group else 0,
                'video_count': len(group_videos),
                'quality_rank': i + 1
            }
        
        # Waisen-/Einzelgruppe (niedrigste Qualität)
        if orphans:
            # Einzelvideos nach ihrer besten Ähnlichkeit zu anderen sortieren
            orphan_qualities = []
            for orphan in orphans:
                orphan_idx = all_videos.index(orphan)
                # Finde beste Ähnlichkeit zu allen anderen Videos
                best_sim = 0
                for other_idx in range(len(all_videos)):
                    if other_idx != orphan_idx:
                        sim = similarity_matrix[orphan_idx, other_idx]
                        best_sim = max(best_sim, sim)
                orphan_qualities.append((orphan, best_sim))
            
            # Sortiere Waisen nach ihrer besten Ähnlichkeit
            orphan_qualities.sort(key=lambda x: x[1], reverse=True)
            sorted_orphans = [video for video, _ in orphan_qualities]
            
            orphan_group_id = f"group_{len(sorted_groups):03d}_orphans"
            result[orphan_group_id] = sorted_orphans
            
            avg_orphan_quality = np.mean([quality for _, quality in orphan_qualities])
            group_metadata[orphan_group_id] = {
                'avg_similarity': float(avg_orphan_quality),
                'min_similarity': float(min(quality for _, quality in orphan_qualities)) if orphan_qualities else 0,
                'max_similarity': float(max(quality for _, quality in orphan_qualities)) if orphan_qualities else 0,
                'video_count': len(orphans),
                'quality_rank': len(sorted_groups) + 1,
                'is_orphan_group': True
            }
        
        # Ergebnisse ausgeben
        print(f"\\n📊 Gruppierung abgeschlossen:")
        for group_id, videos in result.items():
            meta = group_metadata.get(group_id, {})
            quality = meta.get('avg_similarity', 0)
            rank = meta.get('quality_rank', 0)
            orphan_marker = " (Einzelvideos)" if meta.get('is_orphan_group', False) else ""
            print(f"  Rang {rank}: {group_id} - {len(videos)} Videos, Qualität: {quality:.3f}{orphan_marker}")
        
        # Metadaten für spätere Verwendung speichern
        self.last_group_metadata = group_metadata
        
        return result
    
    def get_group_statistics(self) -> Dict:
        """Gibt detaillierte Statistiken der letzten Gruppierung zurück"""
        return getattr(self, 'last_group_metadata', {})


def analyze_and_group_by_similarity(segments_dir: str, output_dir: str,
                                  method: str = 'cnn',
                                  min_similarity: float = 0.75,
                                  min_group_size: int = 2,
                                  orphan_threshold: float = 0.5,
                                  similarity_metric: str = 'cosine') -> Dict[str, str]:
    """
    Hauptfunktion: Gruppierung basierend auf Ähnlichkeit
    
    Args:
        segments_dir: Verzeichnis mit Video-Segmenten
        output_dir: Ausgabeverzeichnis
        method: Feature-Extraktions-Methode
        min_similarity: Mindest-Ähnlichkeit für Gruppierung (0-1)
        min_group_size: Mindestanzahl Videos pro Gruppe
        orphan_threshold: Schwellwert für Einzelvideos
        similarity_metric: Ähnlichkeitsmetrik ('cosine', 'euclidean', 'correlation')
    
    Returns:
        Dictionary mit Dateipfaden der gruppierten Videos
    """
    # Video-Dateien finden
    video_files = []
    for file in os.listdir(segments_dir):
        if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_files.append(os.path.join(segments_dir, file))
    
    if not video_files:
        print("Keine Video-Dateien gefunden!")
        return {}
    
    print(f"📁 Gefunden: {len(video_files)} Video-Segmente")
    print(f"🔧 Methode: {method}")
    print(f"🎯 Min-Ähnlichkeit: {min_similarity:.2f}")
    print(f"👥 Min-Gruppengröße: {min_group_size}")
    print(f"🏠 Waisen-Schwellwert: {orphan_threshold:.2f}")
    print(f"📏 Ähnlichkeitsmetrik: {similarity_metric}")
    
    # Analyzer und Grouper erstellen
    analyzer = VideoSegmentAnalyzer(method=method)
    grouper = SimilarityGrouper(analyzer, similarity_metric=similarity_metric)
    
    # Ähnlichkeitsbasierte Gruppierung
    groups = grouper.group_by_similarity(
        video_files,
        min_similarity=min_similarity,
        min_group_size=min_group_size,
        orphan_threshold=orphan_threshold
    )
    
    if not groups:
        print("❌ Keine Gruppen gefunden!")
        return {}
    
    # Videos zusammenfügen
    print(f"\\n🔗 Füge {len(groups)} Gruppen zusammen...")
    merged_files = merge_video_groups(groups, output_dir, group_prefix="similarity")
    
    # Detaillierte Info speichern
    group_stats = grouper.get_group_statistics()
    groups_info = {
        'method': method,
        'grouping_type': 'similarity_based',
        'min_similarity': min_similarity,
        'min_group_size': min_group_size,
        'orphan_threshold': orphan_threshold,
        'similarity_metric': similarity_metric,
        'groups': {k: [os.path.basename(v) for v in videos] for k, videos in groups.items()},
        'merged_files': {k: os.path.basename(v) for k, v in merged_files.items()},
        'group_statistics': group_stats
    }
    
    info_path = os.path.join(output_dir, 'similarity_grouping_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(groups_info, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Detaillierte Informationen gespeichert: {info_path}")
    
    return merged_files
