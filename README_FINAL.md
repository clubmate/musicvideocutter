# Music Video Cutter - Vereinfachte Version

## Übersicht

Das Music Video Cutter Tool erkennt und schneidet Szenen aus Musikvideos und gruppiert ähnliche Szenen basierend auf **tatsächlicher visueller Ähnlichkeit**. 

**Vereinfacht**: Alle alten Clustering-Methoden wurden entfernt - nur noch die optimale ähnlichkeitsbasierte Gruppierung ist verfügbar.

## Installation

```bash
# Klone das Repository
git clone <repo-url>
cd musicvideocutter

# Erstelle virtuelle Umgebung
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Installiere Dependencies
pip install -r requirements.txt
```

## Verwendung

### Basis-Verwendung
```bash
# Nur Szenen erkennen und schneiden
python musicvideocutter.py "path/to/video.mp4"

# Mit YouTube URL
python musicvideocutter.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Mit Ähnlichkeitsgruppierung
```bash
# Standard-Gruppierung (empfohlen)
python musicvideocutter.py "path/to/video.mp4" --group

# Mit angepasster Ähnlichkeit
python musicvideocutter.py "path/to/video.mp4" --group --min-similarity 0.8

# Alle verfügbaren Parameter
python musicvideocutter.py "path/to/video.mp4" --group \
    --group-method cnn \
    --min-similarity 0.75 \
    --min-group-size 2 \
    --orphan-threshold 0.5 \
    --similarity-metric cosine
```

## Parameter

### Ähnlichkeitsbasierte Gruppierung

| Parameter | Standard | Beschreibung |
|-----------|----------|--------------|
| `--group-method` | `cnn` | Feature-Extraktions-Methode (`histogram`, `orb`, `sift`, `phash`, `cnn`, `audio`) |
| `--min-similarity` | `0.75` | Mindest-Ähnlichkeit für Gruppierung (0-1) |
| `--min-group-size` | `2` | Mindestanzahl Videos pro Gruppe |
| `--orphan-threshold` | `0.5` | Schwellwert für Einzelvideos |
| `--similarity-metric` | `cosine` | Ähnlichkeitsmetrik (`cosine`, `euclidean`, `correlation`) |

## Konfiguration (config.yaml)

```yaml
scene_detection:
  method: adaptive                # Szenen-Erkennungsmethode
  min_scene_len: 7               # Mindestlänge einer Szene in Sekunden

grouping:
  enabled: false                # Automatische Gruppierung aktivieren
  method: cnn                   # Feature-Extraktions-Methode (empfohlen: cnn)
  min_similarity: 0.75          # Mindest-Ähnlichkeit für Gruppierung (0-1)
  min_group_size: 2             # Mindestanzahl Videos pro Gruppe
  orphan_threshold: 0.5         # Schwellwert für Einzelvideos
  similarity_metric: cosine     # Ähnlichkeitsmetrik (empfohlen: cosine)

output:
  download_dir: output          # Basis-Verzeichnis für Downloads
  temp_dir: temp_segments       # Zielordner für geschnittene Szenen
  merged_dir: merged_videos     # Ausgabeordner für gruppierte Videos
```

## Wie die Ähnlichkeitsgruppierung funktioniert

### 1. Feature-Extraktion
- **CNN (empfohlen)**: Deep Learning Features mit ResNet50
- **Histogram**: Farb-Histogramme
- **ORB/SIFT**: Computer Vision Keypoints
- **Audio**: Audio-Features (benötigt librosa)

### 2. Ähnlichkeitsberechnung
- **Cosine (empfohlen)**: Cosine Similarity zwischen Feature-Vektoren
- **Euclidean**: Euklidische Distanz
- **Correlation**: Pearson Korrelation

### 3. Gruppierungsalgorithmus
1. **Qualifizierende Paare finden**: Nur Video-Paare mit Ähnlichkeit ≥ `min_similarity`
2. **Greedy Expansion**: Starte mit bestem Paar, erweitere nur wenn **alle** Verbindungen qualifizieren
3. **Keine transitiven Schlüsse**: A-B (0.8) + B-C (0.8) führt nur zu A-B-C wenn auch A-C ≥ 0.75
4. **Qualitätssortierung**: Gruppen nach durchschnittlicher Ähnlichkeit sortiert
5. **Waisen-Gruppe**: Videos ohne ausreichende Ähnlichkeiten

### 4. Ausgabe
- **Qualitätssortierte Gruppen**: `group_000_sim0.891`, `group_001_sim0.838`, etc.
- **Waisen-Gruppe**: `group_XXX_orphans` für Videos mit niedrigen Ähnlichkeiten
- **Detaillierte Statistiken**: JSON-Datei mit allen Metadaten

## Beispiel-Ergebnisse

Für das Test-Video "Haiyti - Sweet" (57 Segmente) mit `--min-similarity 0.75`:

```
📊 Gruppierung abgeschlossen:
  Rang 1: group_000_sim0.891 - 5 Videos, Qualität: 0.891
  Rang 2: group_001_sim0.838 - 8 Videos, Qualität: 0.838
  Rang 3: group_002_sim0.838 - 6 Videos, Qualität: 0.838
  ...
  Rang 14: group_013_orphans - 10 Videos, Qualität: 0.738 (Einzelvideos)
```

**Ergebnis**: 14 finale Videos - sehr ähnliche Szenen sind zusammengefasst, unterschiedliche bleiben getrennt.

## Empfohlene Einstellungen

### Für Musikvideos (Standard)
```bash
python musicvideocutter.py "video.mp4" --group \
    --group-method cnn \
    --min-similarity 0.75 \
    --min-group-size 2
```

### Für strengere Gruppierung
```bash
python musicvideocutter.py "video.mp4" --group \
    --min-similarity 0.85 \
    --min-group-size 3
```

### Für lockerere Gruppierung
```bash
python musicvideocutter.py "video.mp4" --group \
    --min-similarity 0.65 \
    --orphan-threshold 0.4
```

## Fehlerbehebung

### TensorFlow nicht verfügbar
```bash
pip install tensorflow
```

### Librosa nicht verfügbar (für Audio-Features)
```bash
pip install librosa
```

### Fallback bei Problemen
```bash
# Verwende einfachere Histogram-Methode
python musicvideocutter.py "video.mp4" --group --group-method histogram
```

## Technische Details

- **Keine Cluster-Anzahl mehr**: Das System bestimmt automatisch die optimale Anzahl Gruppen
- **Qualitätsbasiert**: Gruppen werden nur erstellt wenn die Ähnlichkeit ausreichend hoch ist
- **Robuste Gruppierung**: Keine "schwachen" Gruppen durch transitive Verbindungen
- **Skalierbar**: Funktioniert mit 2-200+ Video-Segmenten

## Dateien-Struktur

```
output/
  video_name/
    video_name.mp4              # Original-Video
    temp_segments/              # Einzelne Szenen-Segmente
      Scene-001.mp4
      Scene-002.mp4
      ...
    merged_videos/              # Gruppierte finale Videos
      similarity_group_000_sim0.891.mp4
      similarity_group_001_sim0.838.mp4
      ...
      similarity_grouping_info.json  # Detaillierte Metadaten
```
