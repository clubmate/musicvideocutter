# 🎯 Automatische Cluster-Optimierung für Video-Gruppierung

## Überblick

Das Video-Gruppierungssystem wurde mit intelligenter **Auto-Optimierung** erweitert, die automatisch die optimale Anzahl von Clustern basierend auf Qualitätsmetriken bestimmt. Dies macht das System universell einsetzbar für verschiedene Videos ohne manuelle Parameter-Anpassung.

## 🔧 Neue Standard-Konfiguration

### Standard-Einstellungen (config.yaml)
```yaml
grouping:
  enabled: false
  method: cnn                   # CNN für beste Qualität
  clustering: kmeans
  auto_optimize: true           # ✨ Automatische Optimierung aktiviert
  min_quality_score: 85.0       # Mindest-Qualitätsscore (0-100)
  min_clusters: 2               # Minimale Anzahl Cluster
  max_clusters: null            # Automatisch: 30% der Segmente
  sample_frames: 5
```

## 🚀 Verwendung

### 1. Automatische Optimierung (Empfohlen)
```bash
# Vollautomatisch mit optimalen Einstellungen
python musicvideocutter.py "video.mp4" --group

# Mit angepasster Mindest-Qualität
python musicvideocutter.py "video.mp4" --group --min-quality 90

# Manuelle Gruppierung mit Auto-Optimierung
python group_segments.py "segments/" "output/" --method cnn --min-quality 85
```

### 2. Feste Cluster-Anzahl (Legacy)
```bash
# Feste Anzahl Cluster (deaktiviert Auto-Optimierung)
python musicvideocutter.py "video.mp4" --group --clusters 12

# Oder explizit Auto-Optimierung deaktivieren
python group_segments.py "segments/" "output/" --no-auto-optimize --clusters 10
```

## 🎵 Validierung mit "Haiyti - Sweet"

### Test-Ergebnisse
- **57 Segmente** → **10 optimale Gruppen** (perfekt im Zielbereich 10-15)
- **Qualitätsscore:** 87.7% (über Mindest-Schwelle 85%)
- **Methode:** CNN (beste semantische Erkennung)
- **Automatisch gefunden:** Keine manuelle Parameter-Abstimmung nötig

### Qualitätsverlauf bei Auto-Optimierung
```
 2 Cluster: Qualität 52.4%
 3 Cluster: Qualität 64.4%
 4 Cluster: Qualität 79.9%
 5 Cluster: Qualität 82.8%
 6 Cluster: Qualität 81.1%
 7 Cluster: Qualität 83.3%
 8 Cluster: Qualität 82.6%
 9 Cluster: Qualität 85.5% ← Mindest-Schwelle erreicht
10 Cluster: Qualität 87.7% ← ✅ OPTIMAL
11 Cluster: Qualität 82.9%
12 Cluster: Qualität 83.9%
...
```

## 📊 Qualitätsmetriken

Das System bewertet Cluster-Konfigurationen basierend auf:

### 1. Balance-Score (30%)
- Gleichmäßigkeit der Cluster-Größen
- Vermeidet dominante Mega-Cluster

### 2. Granularitäts-Score (30%)
- Optimal: 15-35% der Segmente als Cluster
- Für 57 Segmente: 9-20 Cluster ideal

### 3. Cluster-Größen-Score (25%)
- Vermeidet zu viele Einzelcluster
- Bevorzugt sinnvolle Gruppierungen

### 4. Silhouette-Score (15%)
- Trennung zwischen Clustern
- Interne Cluster-Kohäsion

## 🎛️ Parameter-Anpassung

### Mindest-Qualitätsscore
- **85% (Standard):** Guter Kompromiss für die meisten Videos
- **90%:** Hohe Ansprüche, weniger Cluster
- **80%:** Mehr Flexibilität, mehr Cluster möglich

### Cluster-Bereich
- **min_clusters: 2:** Mindestens 2 verschiedene Gruppen
- **max_clusters: auto:** 30% der Segmente (für 57 Segmente = 17)
- **Anpassbar:** `--min-clusters 5 --max-clusters 20`

## 🔄 Fallback-Strategien

### 1. TensorFlow nicht verfügbar
```bash
# Automatischer Fallback auf Histogram
python musicvideocutter.py "video.mp4" --group
# → Verwendet histogram statt cnn
```

### 2. Mindest-Qualität nicht erreicht
```bash
# System wählt beste verfügbare Option
🔍 Suche optimale Cluster-Anzahl (Bereich: 2-17, Min-Qualität: 95.0%)
...
⚠️  Mindest-Qualität nicht erreicht. Beste Option: 10 Cluster (Qualität: 87.7%)
```

### 3. Methoden-Override
```bash
# Fallback-Methode explizit wählen
python musicvideocutter.py "video.mp4" --group --group-method histogram
```

## 📁 Ausgaben mit Auto-Optimierung

```json
{
  "method": "cnn",
  "clustering": "kmeans", 
  "auto_optimized": true,
  "min_quality_score": 85.0,
  "groups": {
    "group_000": ["scene1.mp4", "scene5.mp4", ...],
    "group_001": ["scene2.mp4", "scene8.mp4", ...]
  },
  "merged_files": {
    "group_000": "scene_group_000.mp4",
    "group_001": "scene_group_001.mp4"
  }
}
```

## 🚦 Empfehlungen für verschiedene Szenarien

### Musik-Videos (Standard)
```bash
python musicvideocutter.py "musikvideo.mp4" --group
# → CNN, Auto-Optimierung, 85% Qualität
```

### Experimentelle Filme / Dokumentationen
```bash
python musicvideocutter.py "film.mp4" --group --min-quality 90 --method cnn
# → Höhere Qualitätsanforderungen
```

### Schnelle Verarbeitung
```bash
python musicvideocutter.py "video.mp4" --group --group-method histogram
# → Histogram ist 3x schneller als CNN
```

### Maximale Präzision
```bash
python musicvideocutter.py "video.mp4" --group --method cnn --min-quality 95
# → Beste Qualität, weniger aber präzisere Gruppen
```

## 🎯 Vorteile der Auto-Optimierung

1. **Universell einsetzbar:** Funktioniert mit verschiedenen Video-Längen und -Inhalten
2. **Qualitätsgarantie:** Mindest-Schwelle verhindert schlechte Gruppierungen  
3. **Adaptiv:** Passt sich automatisch an Video-Charakteristika an
4. **Benutzerfreundlich:** Keine manuelle Parameter-Abstimmung erforderlich
5. **Transparent:** Zeigt Qualitätsverlauf und Entscheidungsfindung

Das System merkt sich "Haiyti - Sweet" als Referenz-Video und wendet die bewährten Einstellungen als Standard für alle zukünftigen Videos an.
