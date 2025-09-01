"""
Erweiterte ähnlichkeitsbasierte Gruppierung mit optimierter Community Detection
"""

import numpy as np
import os
import json
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

class AdvancedSimilarityGrouper:
    """Erweiterte Gruppierung mit Community Detection Algorithmen"""
    
    def __init__(self, analyzer, similarity_metric='cosine'):
        self.analyzer = analyzer
        self.similarity_metric = similarity_metric
    
    def group_by_similarity_advanced(self, video_paths: List[str], 
                                   min_similarity: float = 0.75,
                                   min_group_size: int = 2,
                                   orphan_threshold: float = 0.5,
                                   max_group_size: int = None,
                                   merge_threshold: float = 0.8) -> Dict[str, List[str]]:
        """
        VERBESSERTE Gruppierung mit intelligenter Community Detection
        
        Args:
            video_paths: Liste der Video-Pfade
            min_similarity: Mindest-Ähnlichkeit für Gruppierung (0-1)
            min_group_size: Mindestanzahl Videos pro Gruppe
            orphan_threshold: Schwellwert für "Restgruppe"
            max_group_size: Maximale Gruppengröße (None = unbegrenzt)
            merge_threshold: Schwellwert für das Mergen von ähnlichen Gruppen
        
        Returns:
            Dictionary mit Gruppen nach Qualität sortiert
        """
        print(f"🔍 Erweiterte ähnlichkeitsbasierte Gruppierung (Schwellwert: {min_similarity:.2f})")
        
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
        
        # SCHRITT 1: Greedy Community Detection mit Optimierung
        initial_groups = self._greedy_community_detection(
            valid_paths, similarity_matrix, min_similarity, min_group_size, max_group_size
        )
        
        # SCHRITT 2: Merge ähnliche Gruppen
        merged_groups = self._merge_similar_groups(
            initial_groups, similarity_matrix, valid_paths, merge_threshold
        )
        
        # SCHRITT 3: Restgruppe für niedrige Ähnlichkeiten
        orphan_group = self._find_orphans_advanced(
            valid_paths, similarity_matrix, merged_groups, orphan_threshold
        )
        
        # SCHRITT 4: Nach Qualität sortieren
        sorted_groups = self._sort_groups_by_quality(
            merged_groups, similarity_matrix, valid_paths, orphan_group
        )
        
        return sorted_groups
    
    def _calculate_similarity_matrix(self, features: np.ndarray) -> np.ndarray:
        """Berechnet Ähnlichkeitsmatrix zwischen allen Features"""
        
        if self.similarity_metric == 'cosine':
            similarity = cosine_similarity(features)
        elif self.similarity_metric == 'euclidean':
            distances = euclidean_distances(features)
            max_dist = np.max(distances)
            similarity = 1 - (distances / max_dist) if max_dist > 0 else np.ones_like(distances)
        elif self.similarity_metric == 'correlation':
            correlation = np.corrcoef(features)
            similarity = (correlation + 1) / 2
        else:
            raise ValueError(f"Unbekannte Ähnlichkeitsmetrik: {self.similarity_metric}")
        
        return similarity
    
    def _greedy_community_detection(self, video_paths: List[str], 
                                  similarity_matrix: np.ndarray,
                                  min_similarity: float,
                                  min_group_size: int,
                                  max_group_size: Optional[int]) -> List[Tuple[List[str], float]]:
        """
        Greedy Community Detection mit Maximierung der Gruppengröße
        """
        n_videos = len(video_paths)
        unassigned = set(range(n_videos))
        groups = []
        
        print(f"🧩 Greedy Community Detection...")
        
        while unassigned:
            # Finde den besten "Seed" (Video mit den meisten ähnlichen Verbindungen)
            best_seed = None
            best_connections = 0
            
            for candidate in unassigned:
                connections = sum(1 for other in unassigned 
                                if other != candidate and similarity_matrix[candidate, other] >= min_similarity)
                if connections > best_connections:
                    best_connections = connections
                    best_seed = candidate
            
            if best_seed is None:
                # Keine ähnlichen Videos mehr gefunden
                break
            
            # Starte Community um den besten Seed
            community = {best_seed}
            unassigned.remove(best_seed)
            
            # Expandiere Community greedily
            improved = True
            while improved and (max_group_size is None or len(community) < max_group_size):
                improved = False
                best_addition = None
                best_avg_similarity = 0
                
                for candidate in list(unassigned):
                    # Berechne durchschnittliche Ähnlichkeit zu allen in der Community
                    similarities = [similarity_matrix[candidate, member] for member in community]
                    avg_sim = np.mean(similarities)
                    
                    # Prüfe ob alle Ähnlichkeiten über Schwellwert sind
                    if all(sim >= min_similarity for sim in similarities) and avg_sim > best_avg_similarity:
                        best_avg_similarity = avg_sim
                        best_addition = candidate
                
                if best_addition is not None:
                    community.add(best_addition)
                    unassigned.remove(best_addition)
                    improved = True
            
            # Community validieren und hinzufügen
            if len(community) >= min_group_size:
                community_list = list(community)
                group_videos = [video_paths[idx] for idx in community_list]
                
                # Berechne Community-Qualität
                similarities = []
                for i, idx1 in enumerate(community_list):
                    for idx2 in community_list[i+1:]:
                        similarities.append(similarity_matrix[idx1, idx2])
                
                avg_similarity = np.mean(similarities) if similarities else 0
                groups.append((group_videos, avg_similarity))
                
                print(f"  ✅ Community gefunden: {len(group_videos)} Videos, Ø Ähnlichkeit: {avg_similarity:.3f}")
            else:
                # Community zu klein, Videos zurück zu unassigned
                unassigned.update(community)
        
        return groups
    
    def _merge_similar_groups(self, groups: List[Tuple[List[str], float]], 
                            similarity_matrix: np.ndarray,
                            all_videos: List[str],
                            merge_threshold: float) -> List[Tuple[List[str], float]]:
        """
        Merged ähnliche Gruppen basierend auf Inter-Group-Ähnlichkeit
        """
        if len(groups) <= 1:
            return groups
        
        print(f"🔗 Prüfe Merging von Gruppen (Schwellwert: {merge_threshold:.2f})...")
        
        merged_groups = list(groups)
        improved = True
        
        while improved:
            improved = False
            
            for i in range(len(merged_groups)):
                for j in range(i + 1, len(merged_groups)):
                    if i >= len(merged_groups) or j >= len(merged_groups):
                        continue
                        
                    group1_videos, _ = merged_groups[i]
                    group2_videos, _ = merged_groups[j]
                    
                    # Berechne Inter-Group-Ähnlichkeit
                    inter_similarities = []
                    for video1 in group1_videos:
                        for video2 in group2_videos:
                            idx1 = all_videos.index(video1)
                            idx2 = all_videos.index(video2)
                            inter_similarities.append(similarity_matrix[idx1, idx2])
                    
                    avg_inter_similarity = np.mean(inter_similarities)
                    
                    # Merge wenn Ähnlichkeit hoch genug
                    if avg_inter_similarity >= merge_threshold:
                        # Merge Gruppe j in Gruppe i
                        merged_videos = group1_videos + group2_videos
                        
                        # Berechne neue durchschnittliche Ähnlichkeit
                        all_similarities = []
                        for video1 in merged_videos:
                            for video2 in merged_videos:
                                if video1 != video2:
                                    idx1 = all_videos.index(video1)
                                    idx2 = all_videos.index(video2)
                                    all_similarities.append(similarity_matrix[idx1, idx2])
                        
                        new_avg_similarity = np.mean(all_similarities) if all_similarities else 0
                        
                        # Ersetze Gruppen
                        merged_groups[i] = (merged_videos, new_avg_similarity)
                        merged_groups.pop(j)
                        
                        print(f"  🔗 Gruppen gemerged: {len(group1_videos)} + {len(group2_videos)} = {len(merged_videos)} Videos")
                        print(f"     Inter-Ähnlichkeit: {avg_inter_similarity:.3f}, Neue Ø Ähnlichkeit: {new_avg_similarity:.3f}")
                        
                        improved = True
                        break
                
                if improved:
                    break
        
        return merged_groups
    
    def _find_orphans_advanced(self, video_paths: List[str], 
                             similarity_matrix: np.ndarray,
                             existing_groups: List[Tuple[List[str], float]],
                             orphan_threshold: float) -> List[str]:
        """Findet Videos, die in keine Gruppe passen (erweiterte Version)"""
        
        # Sammle alle bereits gruppierten Videos
        grouped_videos = set()
        for group_videos, _ in existing_groups:
            grouped_videos.update(group_videos)
        
        # Finde ungrupierte Videos
        orphans = []
        for video in video_paths:
            if video not in grouped_videos:
                orphans.append(video)
        
        if len(orphans) >= 2:
            orphan_indices = [video_paths.index(video) for video in orphans]
            
            # Versuche Sub-Gruppen unter Waisen zu finden
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
        """Sortiert Gruppen nach Qualität und fügt Metadaten hinzu"""
        
        # Sortiere Gruppen nach durchschnittlicher Ähnlichkeit (höchste zuerst)
        sorted_groups = sorted(groups, key=lambda x: x[1], reverse=True)
        
        result = {}
        group_metadata = {}
        
        # Hauptgruppen (nach Qualität sortiert)
        for i, (group_videos, avg_similarity) in enumerate(sorted_groups):
            group_id = f"group_{i:03d}_sim{avg_similarity:.3f}"
            result[group_id] = group_videos
            
            # Zusätzliche Statistiken
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
        
        # Waisen-Gruppe
        if orphans:
            orphan_qualities = []
            for orphan in orphans:
                orphan_idx = all_videos.index(orphan)
                best_sim = 0
                for other_idx in range(len(all_videos)):
                    if other_idx != orphan_idx:
                        sim = similarity_matrix[orphan_idx, other_idx]
                        best_sim = max(best_sim, sim)
                orphan_qualities.append((orphan, best_sim))
            
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
        print(f"\n📊 Erweiterte Gruppierung abgeschlossen:")
        for group_id, videos in result.items():
            meta = group_metadata.get(group_id, {})
            quality = meta.get('avg_similarity', 0)
            rank = meta.get('quality_rank', 0)
            orphan_marker = " (Einzelvideos)" if meta.get('is_orphan_group', False) else ""
            print(f"  Rang {rank}: {group_id} - {len(videos)} Videos, Qualität: {quality:.3f}{orphan_marker}")
        
        self.last_group_metadata = group_metadata
        return result
    
    def get_group_statistics(self) -> Dict:
        """Gibt detaillierte Statistiken der letzten Gruppierung zurück"""
        return getattr(self, 'last_group_metadata', {})
