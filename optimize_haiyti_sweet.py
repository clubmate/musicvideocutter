#!/usr/bin/env python3
"""
Haiyti - Sweet Video Grouping Optimizer
Spezielles Test-Skript für die Optimierung der Gruppierungsparameter für das Haiyti - Sweet Video
Ziel: 10-15 optimale Gruppen finden
"""

import os
import json
import time
from src.grouping import analyze_and_group_segments

# Konfiguration für Haiyti - Sweet Video
TEST_VIDEO = "Haiyti - Sweet"
SEGMENTS_DIR = r"output\Haiyti - Sweet\temp_segments"
BASE_OUTPUT_DIR = r"output\Haiyti - Sweet\grouping_tests"
TARGET_GROUPS_MIN = 10
TARGET_GROUPS_MAX = 15

def test_method_configurations():
    """Testet verschiedene Methoden und Konfigurationen für das Haiyti - Sweet Video"""
    
    print(f"🎵 Haiyti - Sweet Video Gruppierungs-Optimierung")
    print("=" * 60)
    print(f"Ziel: {TARGET_GROUPS_MIN}-{TARGET_GROUPS_MAX} optimale Gruppen")
    print(f"Segmente: {SEGMENTS_DIR}")
    print("=" * 60)
    
    # Prüfen ob Segmente existieren
    if not os.path.exists(SEGMENTS_DIR):
        print(f"❌ Segmente-Verzeichnis nicht gefunden: {SEGMENTS_DIR}")
        print("Bitte zuerst das Video mit musicvideocutter.py verarbeiten:")
        print(f"  python musicvideocutter.py \"output\\{TEST_VIDEO}\\{TEST_VIDEO}.mp4\"")
        return
    
    # Anzahl Segmente zählen
    segments = [f for f in os.listdir(SEGMENTS_DIR) if f.lower().endswith('.mp4')]
    print(f"📁 Gefunden: {len(segments)} Video-Segmente")
    
    if len(segments) == 0:
        print("❌ Keine Video-Segmente gefunden!")
        return
    
    # Test-Konfigurationen definieren
    test_configs = [
        # Histogram-Tests mit verschiedenen Cluster-Anzahlen
        {"method": "histogram", "clustering": "kmeans", "clusters": 10, "name": "Histogram_K10"},
        {"method": "histogram", "clustering": "kmeans", "clusters": 12, "name": "Histogram_K12"},
        {"method": "histogram", "clustering": "kmeans", "clusters": 15, "name": "Histogram_K15"},
        {"method": "histogram", "clustering": "dbscan", "clusters": None, "name": "Histogram_DBSCAN"},
        
        # ORB Feature-Detection Tests
        {"method": "orb", "clustering": "kmeans", "clusters": 10, "name": "ORB_K10"},
        {"method": "orb", "clustering": "kmeans", "clusters": 12, "name": "ORB_K12"},
        {"method": "orb", "clustering": "kmeans", "clusters": 15, "name": "ORB_K15"},
        {"method": "orb", "clustering": "dbscan", "clusters": None, "name": "ORB_DBSCAN"},
        
        # SIFT Feature-Detection Tests
        {"method": "sift", "clustering": "kmeans", "clusters": 12, "name": "SIFT_K12"},
        {"method": "sift", "clustering": "dbscan", "clusters": None, "name": "SIFT_DBSCAN"},
        
        # Perceptual Hash Tests
        {"method": "phash", "clustering": "kmeans", "clusters": 12, "name": "PHash_K12"},
        {"method": "phash", "clustering": "dbscan", "clusters": None, "name": "PHash_DBSCAN"},
        
        # CNN Tests (TensorFlow)
        {"method": "cnn", "clustering": "kmeans", "clusters": 10, "name": "CNN_K10"},
        {"method": "cnn", "clustering": "kmeans", "clusters": 12, "name": "CNN_K12"},
        {"method": "cnn", "clustering": "kmeans", "clusters": 15, "name": "CNN_K15"},
        {"method": "cnn", "clustering": "dbscan", "clusters": None, "name": "CNN_DBSCAN"},
    ]
    
    results = []
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n🧪 Test {i}/{len(test_configs)}: {config['name']}")
        print(f"   Methode: {config['method']}, Clustering: {config['clustering']}")
        if config['clusters']:
            print(f"   Cluster: {config['clusters']}")
        
        output_dir = os.path.join(BASE_OUTPUT_DIR, config['name'])
        
        start_time = time.time()
        
        try:
            # Gruppierung durchführen
            merged_files = analyze_and_group_segments(
                SEGMENTS_DIR,
                output_dir,
                method=config['method'],
                clustering=config['clustering'],
                n_clusters=config['clusters']
            )
            
            duration = time.time() - start_time
            
            # Ergebnisse analysieren
            info_path = os.path.join(output_dir, 'grouping_info.json')
            if os.path.exists(info_path):
                with open(info_path, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                
                groups = info.get('groups', {})
                num_groups = len(groups)
                group_sizes = [len(videos) for videos in groups.values()]
                
                # Qualitätsbewertung
                avg_size = sum(group_sizes) / len(group_sizes) if group_sizes else 0
                variance = sum((size - avg_size)**2 for size in group_sizes) / len(group_sizes) if group_sizes else 0
                balance_score = max(0, 100 - variance/10)
                
                # Ziel-Score (wie nah an 10-15 Gruppen)
                if TARGET_GROUPS_MIN <= num_groups <= TARGET_GROUPS_MAX:
                    target_score = 100
                else:
                    distance = min(abs(num_groups - TARGET_GROUPS_MIN), abs(num_groups - TARGET_GROUPS_MAX))
                    target_score = max(0, 100 - distance * 10)
                
                # Gesamtscore
                total_score = (balance_score + target_score) / 2
                
                result = {
                    "config": config,
                    "num_groups": num_groups,
                    "group_sizes": group_sizes,
                    "avg_group_size": avg_size,
                    "balance_score": balance_score,
                    "target_score": target_score,
                    "total_score": total_score,
                    "duration": duration,
                    "success": True
                }
                
                print(f"   ✅ Gruppen: {num_groups}, Balance: {balance_score:.1f}, Ziel: {target_score:.1f}, Score: {total_score:.1f}")
                print(f"   ⏱️  Zeit: {duration:.1f}s")
                
            else:
                result = {"config": config, "success": False, "error": "Keine Info-Datei"}
                print(f"   ❌ Keine Info-Datei erstellt")
            
        except Exception as e:
            result = {"config": config, "success": False, "error": str(e)}
            print(f"   ❌ Fehler: {e}")
        
        results.append(result)
    
    # Ergebnisse zusammenfassen und bewerten
    print(f"\n🏆 ERGEBNISSE FÜR {TEST_VIDEO}")
    print("=" * 60)
    
    # Erfolgreiche Tests filtern und sortieren
    successful_results = [r for r in results if r.get("success", False)]
    successful_results.sort(key=lambda x: x.get("total_score", 0), reverse=True)
    
    print(f"{'Rang':<4} {'Name':<15} {'Gruppen':<8} {'Balance':<8} {'Ziel':<6} {'Score':<8} {'Zeit':<6}")
    print("-" * 60)
    
    for i, result in enumerate(successful_results[:10], 1):  # Top 10
        config = result["config"]
        print(f"{i:<4} {config['name']:<15} {result['num_groups']:<8} "
              f"{result['balance_score']:<8.1f} {result['target_score']:<6.1f} "
              f"{result['total_score']:<8.1f} {result['duration']:<6.1f}s")
    
    # Beste Konfiguration identifizieren
    if successful_results:
        best = successful_results[0]
        print(f"\n🥇 BESTE KONFIGURATION FÜR {TEST_VIDEO}:")
        print(f"   Methode: {best['config']['method']}")
        print(f"   Clustering: {best['config']['clustering']}")
        if best['config']['clusters']:
            print(f"   Cluster: {best['config']['clusters']}")
        print(f"   Ergebnis: {best['num_groups']} Gruppen")
        print(f"   Score: {best['total_score']:.1f}/100")
        
        # Empfehlung für zukünftige Verwendung
        best_config = best['config']
        print(f"\n📋 EMPFOHLENER BEFEHL:")
        cmd_parts = [
            f"python group_segments.py",
            f"\"{SEGMENTS_DIR}\"",
            f"\"output/{TEST_VIDEO}/optimal_groups\"",
            f"--method {best_config['method']}",
            f"--clustering {best_config['clustering']}"
        ]
        if best_config['clusters']:
            cmd_parts.append(f"--clusters {best_config['clusters']}")
        
        print(f"   {' '.join(cmd_parts)}")
        
        # Gruppierungsdetails anzeigen
        best_output_dir = os.path.join(BASE_OUTPUT_DIR, best['config']['name'])
        info_path = os.path.join(best_output_dir, 'grouping_info.json')
        
        if os.path.exists(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            
            print(f"\n📊 GRUPPIERUNGSDETAILS:")
            groups = info.get('groups', {})
            for group_id, videos in sorted(groups.items()):
                print(f"   {group_id}: {len(videos)} Videos")
    
    # Alle Ergebnisse speichern
    summary_path = os.path.join(BASE_OUTPUT_DIR, "test_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "test_video": TEST_VIDEO,
            "target_groups": [TARGET_GROUPS_MIN, TARGET_GROUPS_MAX],
            "total_segments": len(segments),
            "test_configs": len(test_configs),
            "successful_tests": len(successful_results),
            "results": results,
            "best_config": successful_results[0] if successful_results else None
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Vollständige Ergebnisse gespeichert: {summary_path}")

def apply_best_configuration():
    """Wendet die beste gefundene Konfiguration an"""
    
    summary_path = os.path.join(BASE_OUTPUT_DIR, "test_summary.json")
    
    if not os.path.exists(summary_path):
        print("❌ Keine Test-Ergebnisse gefunden. Bitte zuerst test_method_configurations() ausführen.")
        return
    
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    best_config = summary.get('best_config')
    if not best_config:
        print("❌ Keine beste Konfiguration in den Ergebnissen gefunden.")
        return
    
    config = best_config['config']
    
    print(f"🎯 Wende beste Konfiguration für {TEST_VIDEO} an:")
    print(f"   Methode: {config['method']}")
    print(f"   Clustering: {config['clustering']}")
    if config['clusters']:
        print(f"   Cluster: {config['clusters']}")
    
    output_dir = os.path.join("output", TEST_VIDEO, "optimal_grouping")
    
    try:
        merged_files = analyze_and_group_segments(
            SEGMENTS_DIR,
            output_dir,
            method=config['method'],
            clustering=config['clustering'],
            n_clusters=config['clusters']
        )
        
        print(f"✅ Optimale Gruppierung erstellt in: {output_dir}")
        print(f"📁 {len(merged_files)} Gruppen-Videos erstellt")
        
        return output_dir
        
    except Exception as e:
        print(f"❌ Fehler bei der Anwendung: {e}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Haiyti - Sweet Video Gruppierungs-Optimierung")
    parser.add_argument('--test', action='store_true', help='Teste alle Konfigurationen')
    parser.add_argument('--apply', action='store_true', help='Wende beste Konfiguration an')
    parser.add_argument('--both', action='store_true', help='Teste und wende beste Konfiguration an')
    
    args = parser.parse_args()
    
    if args.both:
        test_method_configurations()
        apply_best_configuration()
    elif args.test:
        test_method_configurations()
    elif args.apply:
        apply_best_configuration()
    else:
        # Standard: Beide ausführen
        test_method_configurations()
        apply_best_configuration()
