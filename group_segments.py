#!/usr/bin/env python3
"""
Manual Video Grouping Tool
Manuelles Tool zur Gruppierung von bereits erstellten Video-Segmenten
"""

import os
import sys
import argparse
from src.grouping import analyze_and_group_segments

def main():
    parser = argparse.ArgumentParser(
        description="Gruppiere bestehende Video-Segmente nach Ähnlichkeit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Verfügbare Methoden:
  histogram - Farbhistogramm-Vergleich (Standard, schnell)
  orb       - ORB Feature-Detection (robust)
  sift      - SIFT Feature-Detection (präzise, langsamer)
  phash     - Perceptual Hashing (gut für Duplikate)
  cnn       - CNN Features (sehr präzise, benötigt TensorFlow)
  audio     - Audio-basierte Gruppierung (benötigt librosa)

Clustering-Algorithmen:
  kmeans    - K-Means Clustering (Standard)
  dbscan    - DBSCAN (automatische Cluster-Anzahl)

Beispiele:
  python group_segments.py "output/Video Name/temp_segments" "output/Video Name/grouped"
  python group_segments.py input_dir output_dir --method orb --clustering dbscan
  python group_segments.py input_dir output_dir --method histogram --clusters 5
        """
    )
    
    parser.add_argument('input_dir', help='Verzeichnis mit Video-Segmenten')
    parser.add_argument('output_dir', help='Ausgabeverzeichnis für gruppierte Videos')
    parser.add_argument('--method', default='cnn',
                       choices=['histogram', 'orb', 'sift', 'phash', 'cnn', 'audio'],
                       help='Methode für Ähnlichkeitserkennung (default: cnn)')
    parser.add_argument('--clustering', default='kmeans',
                       choices=['kmeans', 'dbscan'],
                       help='Clustering-Algorithmus (default: kmeans)')
    parser.add_argument('--clusters', type=int,
                       help='Anzahl der Cluster (deaktiviert Auto-Optimierung)')
    parser.add_argument('--min-quality', type=float, default=85.0,
                       help='Mindest-Qualitätsscore für Auto-Optimierung (default: 85.0)')
    parser.add_argument('--min-clusters', type=int, default=2,
                       help='Minimale Anzahl Cluster (default: 2)')
    parser.add_argument('--max-clusters', type=int,
                       help='Maximale Anzahl Cluster (default: auto)')
    parser.add_argument('--no-auto-optimize', action='store_true',
                       help='Deaktiviere Auto-Optimierung (benötigt --clusters)')
    parser.add_argument('--preview', action='store_true',
                       help='Zeige nur Vorschau der Gruppierung, ohne Videos zu mergen')
    
    args = parser.parse_args()
    
    # Validierung der Eingabe
    if not os.path.exists(args.input_dir):
        print(f"Fehler: Eingabeverzeichnis '{args.input_dir}' existiert nicht!")
        sys.exit(1)
    
    if not os.path.isdir(args.input_dir):
        print(f"Fehler: '{args.input_dir}' ist kein Verzeichnis!")
        sys.exit(1)
    
    # Video-Dateien zählen
    video_files = []
    for file in os.listdir(args.input_dir):
        if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            video_files.append(file)
    
    if not video_files:
        print(f"Keine Video-Dateien in '{args.input_dir}' gefunden!")
        sys.exit(1)
    
    print(f"Gefunden: {len(video_files)} Video-Segmente in '{args.input_dir}'")
    print(f"Methode: {args.method}")
    print(f"Clustering: {args.clustering}")
    
    # Auto-Optimierung konfigurieren
    auto_optimize = not args.no_auto_optimize and args.clusters is None
    
    if auto_optimize:
        print(f"Auto-Optimierung: Aktiviert (Min-Qualität: {args.min_quality}%)")
        print(f"Cluster-Bereich: {args.min_clusters}-{args.max_clusters or 'auto'}")
        n_clusters = None
    else:
        if args.clusters is None:
            print(f"❌ Fehler: --clusters erforderlich wenn --no-auto-optimize verwendet wird")
            sys.exit(1)
        print(f"Cluster: {args.clusters} (fest)")
        n_clusters = args.clusters
    
    print(f"Ausgabe: {'VORSCHAU MODUS' if args.preview else args.output_dir}")
    print("-" * 50)
    
    try:
        if args.preview:
            # Nur Analyse, kein Merging
            from src.grouping import VideoSegmentAnalyzer, VideoGrouper
            
            video_paths = [os.path.join(args.input_dir, f) for f in video_files]
            
            analyzer = VideoSegmentAnalyzer(method=args.method)
            grouper = VideoGrouper(analyzer, clustering_method=args.clustering)
            
            groups = grouper.group_videos(
                video_paths, 
                n_clusters=n_clusters,
                min_quality_score=args.min_quality,
                min_clusters=args.min_clusters,
                max_clusters=args.max_clusters
            )
            
            print(f"\nVorschau der Gruppierung:")
            print(f"Gefunden: {len(groups)} Gruppen")
            
            for group_id, videos in groups.items():
                print(f"\n{group_id} ({len(videos)} Videos):")
                for video in videos:
                    print(f"  - {os.path.basename(video)}")
            
            total_grouped = sum(len(videos) for videos in groups.values())
            print(f"\nZusammenfassung:")
            print(f"  Videos total: {len(video_files)}")
            print(f"  Videos gruppiert: {total_grouped}")
            print(f"  Gruppen: {len(groups)}")
            
        else:
            # Vollständige Analyse und Merging
            merged_files = analyze_and_group_segments(
                args.input_dir,
                args.output_dir,
                method=args.method,
                clustering=args.clustering,
                n_clusters=n_clusters,
                min_quality_score=args.min_quality,
                min_clusters=args.min_clusters,
                max_clusters=args.max_clusters
            )
            
            print(f"\nGruppierung abgeschlossen!")
            print(f"Erstellte Dateien in '{args.output_dir}':")
            for group_id, file_path in merged_files.items():
                print(f"  {group_id}: {os.path.basename(file_path)}")
            
            # Info-Datei zeigen
            info_file = os.path.join(args.output_dir, 'grouping_info.json')
            if os.path.exists(info_file):
                print(f"\nDetaillierte Informationen in: {info_file}")
    
    except ImportError as e:
        print(f"\nFehler: Benötigte Bibliothek fehlt!")
        print(f"Details: {e}")
        print("\nInstallieren Sie fehlende Pakete:")
        if args.method == 'cnn':
            print("  pip install tensorflow")
        elif args.method == 'audio':
            print("  pip install librosa")
        else:
            print("  pip install scikit-learn matplotlib")
    
    except Exception as e:
        print(f"\nFehler bei der Verarbeitung: {e}")
        print("\nMögliche Lösungen:")
        print("- Prüfen Sie, ob alle Video-Dateien lesbar sind")
        print("- Versuchen Sie eine andere Methode (z.B. --method histogram)")
        print("- Reduzieren Sie die Anzahl der Cluster")
        sys.exit(1)

if __name__ == "__main__":
    main()
