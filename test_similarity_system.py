"""
Test der neuen ähnlichkeitsbasierten Gruppierung mit Haiyti - Sweet
"""

import os
import sys
sys.path.append('src')

from similarity_grouping import analyze_and_group_by_similarity

def test_similarity_grouping():
    """Testet das neue ähnlichkeitsbasierte System"""
    
    # Test mit Haiyti - Sweet Video
    video_name = "Haiyti - Sweet"
    segments_dir = f"output/{video_name}/temp_segments"
    output_dir = f"output/{video_name}/similarity_groups"
    
    if not os.path.exists(segments_dir):
        print(f"❌ Segmente-Verzeichnis nicht gefunden: {segments_dir}")
        print("💡 Führen Sie zuerst die Segmentierung aus:")
        print(f'   python musicvideocutter.py "output/{video_name}/{video_name}.mp4"')
        return
    
    # Zähle verfügbare Segmente
    segments = [f for f in os.listdir(segments_dir) if f.lower().endswith('.mp4')]
    print(f"📁 Gefunden: {len(segments)} Video-Segmente in {segments_dir}")
    
    if len(segments) < 2:
        print("❌ Nicht genügend Segmente für Gruppierung gefunden")
        return
    
    # Test verschiedene Ähnlichkeits-Schwellwerte
    test_configs = [
        {"min_similarity": 0.85, "name": "Sehr hohe Ähnlichkeit (0.85)"},
        {"min_similarity": 0.75, "name": "Hohe Ähnlichkeit (0.75)"},
        {"min_similarity": 0.65, "name": "Mittlere Ähnlichkeit (0.65)"},
        {"min_similarity": 0.55, "name": "Niedrige Ähnlichkeit (0.55)"},
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"🧪 Test: {config['name']}")
        print(f"{'='*60}")
        
        try:
            # Erstelle Unterordner für diesen Test
            test_output_dir = os.path.join(output_dir, f"test_sim_{config['min_similarity']:.2f}")
            
            merged_files = analyze_and_group_by_similarity(
                segments_dir=segments_dir,
                output_dir=test_output_dir,
                method='cnn',
                min_similarity=config['min_similarity'],
                min_group_size=2,
                orphan_threshold=config['min_similarity'] - 0.1,  # 10% niedriger
                similarity_metric='cosine'
            )
            
            results[config['min_similarity']] = {
                'groups_count': len(merged_files),
                'files': merged_files,
                'config': config
            }
            
            print(f"✅ Ergebnis: {len(merged_files)} finale Videos erstellt")
            
            # Analysiere Gruppenergebnisse
            group_sizes = []
            for group_name, video_path in merged_files.items():
                if os.path.exists(video_path):
                    # Schätze Gruppengröße basierend auf Video-Dauer (grob)
                    size_mb = os.path.getsize(video_path) / (1024 * 1024)
                    group_sizes.append(size_mb)
                    print(f"  📁 {group_name}: {size_mb:.1f} MB")
            
            if group_sizes:
                avg_size = sum(group_sizes) / len(group_sizes)
                print(f"📊 Durchschnittliche Gruppengröße: {avg_size:.1f} MB")
                
        except Exception as e:
            print(f"❌ Fehler bei Test {config['name']}: {e}")
            results[config['min_similarity']] = {'error': str(e)}
    
    # Zusammenfassung der Ergebnisse
    print(f"\n{'='*60}")
    print("📊 ERGEBNISSE ZUSAMMENFASSUNG")
    print(f"{'='*60}")
    
    for similarity, result in results.items():
        if 'error' in result:
            print(f"❌ Ähnlichkeit {similarity:.2f}: Fehler - {result['error']}")
        else:
            groups = result['groups_count']
            config_name = result['config']['name']
            print(f"✅ Ähnlichkeit {similarity:.2f}: {groups} finale Videos ({config_name})")
    
    # Empfehlung basierend auf Ergebnissen
    print(f"\n💡 EMPFEHLUNG für Haiyti - Sweet:")
    
    # Finde optimalen Schwellwert (sollte 10-15 Gruppen ergeben)
    best_config = None
    target_range = (10, 15)
    
    for similarity, result in results.items():
        if 'groups_count' in result:
            groups = result['groups_count']
            if target_range[0] <= groups <= target_range[1]:
                if best_config is None or abs(groups - 12.5) < abs(best_config[1]['groups_count'] - 12.5):
                    best_config = (similarity, result)
    
    if best_config:
        similarity, result = best_config
        print(f"🎯 Optimaler Schwellwert: {similarity:.2f}")
        print(f"   → Erstellt {result['groups_count']} finale Videos (Zielbereich: 10-15)")
        print(f"   → Verwendung: --min-similarity {similarity:.2f}")
    else:
        print("⚠️ Kein Schwellwert im Zielbereich (10-15 Gruppen) gefunden")
        print("💡 Versuchen Sie andere Schwellwerte zwischen 0.5 und 0.9")

def test_comparison_with_old_system():
    """Vergleicht das neue System mit dem alten Auto-Optimierung"""
    
    print(f"\n{'='*60}")
    print("🔄 VERGLEICH: Ähnlichkeitsbasiert vs. Cluster-basiert")
    print(f"{'='*60}")
    
    # Hinweis für manuellen Vergleich
    print("💡 Zum Vergleichen führen Sie aus:")
    print("1. Ähnlichkeitsbasiert (NEU):")
    print('   python musicvideocutter.py "output/Haiyti - Sweet/Haiyti - Sweet.mp4" --group --similarity-grouping --min-similarity 0.75')
    print()
    print("2. Cluster-basiert (ALT):")
    print('   python musicvideocutter.py "output/Haiyti - Sweet/Haiyti - Sweet.mp4" --group --no-similarity-grouping --auto-optimize')
    print()
    print("3. Cluster-basiert fix 10:")
    print('   python musicvideocutter.py "output/Haiyti - Sweet/Haiyti - Sweet.mp4" --group --no-similarity-grouping --clusters 10')

if __name__ == "__main__":
    print("🧪 Test der neuen ähnlichkeitsbasierten Gruppierung")
    print("=" * 60)
    
    test_similarity_grouping()
    test_comparison_with_old_system()
    
    print(f"\n✅ Test abgeschlossen!")
    print("💡 Die Ergebnisse finden Sie in 'output/Haiyti - Sweet/similarity_groups/'")
