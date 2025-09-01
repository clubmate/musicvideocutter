# Music Video Cutter

Ein vereinfachtes Tool: lädt ein Musikvideo herunter (oder nutzt lokale Datei) und schneidet es automatisch in Szenen. Die frühere Funktion zum Gruppieren/Mergen ähnlicher Szenen wurde entfernt.

## Beschreibung

Erkennt Schnitte (mehrere Verfahren) und exportiert jede Szene als eigenes Segment (verlustfrei per FFmpeg Copy). Kein Clustering/Merging mehr – Fokus liegt auf zuverlässiger Schnitt-Erkennung.

## Features

- YouTube Videos & Playlists oder lokale Dateien
- Mehrere Schnitt-Detektionsmethoden: adaptive | content | threshold_params | histogram | hash
- Verlustfreier Segment-Export via FFmpeg Stream Copy
- Fortschrittsanzeigen (tqdm)
- Konfigurierbar über `config.yaml`

## Installation

1. Clone or download this repository
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Requirements

- Python 3.7+
- FFmpeg (for video processing)
- Benötigte Python Pakete: siehe `requirements.txt` (yt-dlp, scenedetect, moviepy, pyyaml, tqdm, opencv-python, numpy)

## Usage

Standard:
```bash
python musicvideocutter.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Local file:
```bash
python musicvideocutter.py "my_video.mp4"
```

Grouping-Optionen wurden entfernt – jeder erkannte Schnitt wird als Datei geschrieben.

### Examples

YouTube video:
```bash
python musicvideocutter.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

YouTube playlist:
```bash
python musicvideocutter.py "https://www.youtube.com/playlist?list=PLAYLIST_ID"
```


## Konfiguration

`config.yaml` (Ausschnitt):

```yaml
scene_detection:
  method: adaptive           # adaptive | content | threshold_params | histogram | hash
  min_scene_len: 6           # Minimum scene length (frames) if seconds not used
  # min_scene_len_seconds: 1 # Alternative in seconds
  adaptive:
    adaptive_threshold: 3.0
    window_width: 2
    min_content_val: 15.0
  content:
    threshold: 27.0
    luma_only: false
  threshold_params:
    threshold: 12
    fade_bias: 0.0
    add_final_scene: false
  histogram:
    threshold: 0.05
    bins: 256
  hash:
    threshold: 0.395
    size: 16
    lowpass: 2


transition:
  type: hard_cut       # placeholder for future transitions
  fade_duration: 1.0

output:
  download_dir: output
  temp_dir: temp_segments
  merged_dir: merged_videos
```

### Wichtige Optionen
- `scene_detection.method`
- `min_scene_len` oder `min_scene_len_seconds`

### Methoden Kurzinfo
- adaptive: adaptiver Content-Wert (robust gegen Bewegung)
- content: fester Farbänderungs-Schwellenwert
- threshold_params: Fade / Helligkeits-basiert
- histogram: Y-Kanal Histogrammdifferenz
- hash: Perceptual Hash Unterschiede

## Pipeline

1. Download / Input Normalisierung
2. Szenen-Detektion (gewählte Methode)
3. Segment-Export (FFmpeg Copy) in `temp_segments/`
4. Fertig

## Output Struktur
```
VideoTitel/
└── temp_segments/
  ├── segment_000.mp4
  ├── segment_001.mp4
  └── ...
```
`temp_segments/` enthält alle erkannten Szenen.

## Tipps

- Mehr Schnitte: `scene_detection.method` auf `hash` oder `histogram` testen
- Weniger Rauschen: `adaptive` oder `content` Parameter feinjustieren
- Mindestlänge: `min_scene_len` oder `min_scene_len_seconds` setzen

## Troubleshooting

- No segments: adjust scene detection method/parameters
- FFmpeg Fehler: `ffmpeg -version` testen / PATH prüfen
- Performance: Kürzere Testclips, andere Methode wählen

## License

Open source – frei nutzbar & anpassbar.
