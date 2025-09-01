"""
Vergleichstest: Standard vs. Erweiterte Ähnlichkeitsbasierte Gruppierung
"""

import os
import sys
sys.path.append('src')

def test_both_algorithms():
    """Vergleicht Standard Connected Components vs. Erweiterte Community Detection"""
    
    video_name = "Haiyti - Sweet"
    segments_dir = f"output/{video_name}/temp_segments"
    
    if not os.path.exists(segments_dir):
        print(f"❌ Segmente-Verzeichnis nicht gefunden: {segments_dir}")
        return
    
    segments = [f for f in os.listdir(segments_dir) if f.lower().endswith('.mp4')]
    print(f"📁 Gefunden: {len(segments)} Video-Segmente")
    
    test_similarity = 0.75
    
    print(f"\n{'='*80}")
    print(f"🧪 VERGLEICHSTEST mit Ähnlichkeitsschwellwert: {test_similarity:.2f}")
    print(f"{'='*80}")
    
    # Test 1: Standard Connected Components
    print(f"\n🔗 TEST 1: Standard Connected Components")
    print(f"{'='*50}")
    
    try:
        from src.similarity_grouping import analyze_and_group_by_similarity
        
        output_dir_standard = f"output/{video_name}/comparison_standard"
        
        merged_files_standard = analyze_and_group_by_similarity(
            segments_dir=segments_dir,
            output_dir=output_dir_standard,
            method='cnn',
            min_similarity=test_similarity,
            min_group_size=2,
            orphan_threshold=0.5,
            similarity_metric='cosine',
            use_advanced=False  # Standard-Algorithmus
        )
        
        print(f"✅ Standard-Algorithmus: {len(merged_files_standard)} finale Videos")
        
        # Analysiere Gruppengrößen
        standard_sizes = []
        for group_name, video_path in merged_files_standard.items():
            if os.path.exists(video_path):
                size_mb = os.path.getsize(video_path) / (1024 * 1024)
                standard_sizes.append(size_mb)
                segments_count = estimate_segments_count(size_mb)
                print(f"  📁 {group_name}: {size_mb:.1f} MB (~{segments_count} Segmente)")
        
    except Exception as e:
        print(f"❌ Fehler beim Standard-Algorithmus: {e}")
        merged_files_standard = {}
        standard_sizes = []
    
    # Test 2: Erweiterte Community Detection
    print(f"\n🧩 TEST 2: Erweiterte Community Detection")
    print(f"{'='*50}")
    
    try:
        output_dir_advanced = f"output/{video_name}/comparison_advanced"
        
        merged_files_advanced = analyze_and_group_by_similarity(
            segments_dir=segments_dir,
            output_dir=output_dir_advanced,
            method='cnn',
            min_similarity=test_similarity,
            min_group_size=2,
            orphan_threshold=0.5,
            similarity_metric='cosine',
            use_advanced=True,  # Erweiterter Algorithmus
            max_group_size=None,
            merge_threshold=0.8
        )
        
        print(f"✅ Erweiterter Algorithmus: {len(merged_files_advanced)} finale Videos")
        
        # Analysiere Gruppengrößen
        advanced_sizes = []
        for group_name, video_path in merged_files_advanced.items():
            if os.path.exists(video_path):
                size_mb = os.path.getsize(video_path) / (1024 * 1024)
                advanced_sizes.append(size_mb)
                segments_count = estimate_segments_count(size_mb)
                print(f"  📁 {group_name}: {size_mb:.1f} MB (~{segments_count} Segmente)")
        
    except Exception as e:
        print(f"❌ Fehler beim erweiterten Algorithmus: {e}")
        merged_files_advanced = {}
        advanced_sizes = []
    
    # Vergleichsanallyse
    print(f"\n📊 VERGLEICHSANALYSE")
    print(f"{'='*50}")
    
    if merged_files_standard and merged_files_advanced:
        print(f"Standard-Algorithmus:")
        print(f"  🔢 Anzahl Gruppen: {len(merged_files_standard)}")
        if standard_sizes:
            print(f"  📏 Ø Gruppengröße: {sum(standard_sizes)/len(standard_sizes):.1f} MB")
            print(f"  📏 Größte Gruppe: {max(standard_sizes):.1f} MB")
            print(f"  📏 Kleinste Gruppe: {min(standard_sizes):.1f} MB")
        
        print(f"\nErweiterter Algorithmus:")
        print(f"  🔢 Anzahl Gruppen: {len(merged_files_advanced)}")
        if advanced_sizes:
            print(f"  📏 Ø Gruppengröße: {sum(advanced_sizes)/len(advanced_sizes):.1f} MB")
            print(f"  📏 Größte Gruppe: {max(advanced_sizes):.1f} MB")
            print(f"  📏 Kleinste Gruppe: {min(advanced_sizes):.1f} MB")
        
        # Empfehlung
        print(f"\n💡 EMPFEHLUNG:")
        if len(merged_files_advanced) < len(merged_files_standard):
            print(f"✅ Erweiterter Algorithmus erstellt weniger, aber größere Gruppen")
            print(f"   → Bessere Konsolidierung ähnlicher Videos")
            print(f"   → Empfohlen für maximale Gruppierung")
        elif len(merged_files_advanced) > len(merged_files_standard):
            print(f"⚠️ Erweiterter Algorithmus erstellt mehr, kleinere Gruppen")
            print(f"   → Präzisere Ähnlichkeitserkennung")
        else:
            print(f"🔄 Beide Algorithmen erstellen gleich viele Gruppen")
            print(f"   → Vergleichen Sie die Gruppengrößen")
    
    print(f"\n🎯 OPTIMALE EINSTELLUNGEN für Haiyti - Sweet:")
    print(f"Ziel: 10-15 finale Videos")
    
    if merged_files_advanced:
        groups_count = len(merged_files_advanced)
        if 10 <= groups_count <= 15:
            print(f"✅ Erweiterte Community Detection mit {test_similarity:.2f} Schwellwert: OPTIMAL")
            print(f"   → {groups_count} Gruppen (perfekt im Zielbereich)")
        elif groups_count < 10:
            print(f"⬆️ Schwellwert zu niedrig ({test_similarity:.2f}) → Mehr Gruppen benötigt")
            print(f"   💡 Versuchen Sie --min-similarity 0.8 oder höher")
        else:
            print(f"⬇️ Schwellwert zu hoch ({test_similarity:.2f}) → Weniger Gruppen benötigt")
            print(f"   💡 Versuchen Sie --min-similarity 0.7 oder niedriger")
    
    print(f"\n🚀 EMPFOHLENE BEFEHLE:")
    print(f"1. Erweiterte Gruppierung (empfohlen):")
    print(f'   python musicvideocutter.py "output/{video_name}/{video_name}.mp4" --group --similarity-grouping --advanced-grouping --min-similarity 0.75')
    print(f"\n2. Standard-Gruppierung:")
    print(f'   python musicvideocutter.py "output/{video_name}/{video_name}.mp4" --group --similarity-grouping --min-similarity 0.75')

def estimate_segments_count(size_mb):
    """Schätzt Anzahl der Segmente basierend auf Dateigröße"""
    # Durchschnittlich ~0.5 MB pro Segment (grober Schätzwert)
    avg_segment_size = 0.5
    return int(size_mb / avg_segment_size)

if __name__ == "__main__":
    print("🧪 Vergleichstest: Standard vs. Erweiterte Ähnlichkeitsbasierte Gruppierung")
    test_both_algorithms()
    print(f"\n✅ Vergleichstest abgeschlossen!")
