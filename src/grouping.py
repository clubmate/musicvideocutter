import os
import cv2
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from PIL import Image
import imagehash
import subprocess
from tqdm import tqdm

@dataclass
class SceneFeatures:
    file: str
    hist: np.ndarray  # aggregated histogram (concatenated channels)
    hashes: List[imagehash.ImageHash]


def _extract_keyframe_indices(frame_count: int) -> List[int]:
    # Use 5 frames: start, 25%, 50%, 75%, end
    if frame_count <= 0:
        return [0]
    positions = [0, 0.25, 0.5, 0.75, 1.0]
    idxs = sorted({min(frame_count - 1, max(0, int(p * (frame_count - 1)))) for p in positions})
    return idxs


def _frame_hist_hsv(frame: np.ndarray, bins: int = 32) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
    h_hist = cv2.normalize(h_hist, h_hist).flatten()
    s_hist = cv2.normalize(s_hist, s_hist).flatten()
    v_hist = cv2.normalize(v_hist, v_hist).flatten()
    return np.concatenate([h_hist, s_hist, v_hist])  # shape 3*bins


def _aggregate_hists(hists: List[np.ndarray]) -> np.ndarray:
    return np.mean(hists, axis=0)


def _compute_scene_features(path: str, hash_size: int = 16, bins: int = 32) -> SceneFeatures:
    cap = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    indices = _extract_keyframe_indices(frame_count)
    hists: List[np.ndarray] = []
    hashes: List[imagehash.ImageHash] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        # optional dark filter
        if frame.mean() < 5 and frame_count > 10:
            continue  # skip near-black
        hists.append(_frame_hist_hsv(frame, bins=bins))
        # Convert to PIL for hash
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        try:
            hashes.append(imagehash.phash(img, hash_size=hash_size))
        except Exception:
            pass
    cap.release()
    if not hists:  # fallback single black vector
        hists = [np.zeros(3 * bins, dtype=np.float32)]
    hist = _aggregate_hists(hists)
    return SceneFeatures(file=path, hist=hist, hashes=hashes)


def _bhattacharyya(h1: np.ndarray, h2: np.ndarray) -> float:
    # Both vectors already normalized per channel; re-normalize whole for safety
    h1n = h1 / (np.sum(h1) + 1e-9)
    h2n = h2 / (np.sum(h2) + 1e-9)
    bc = np.sum(np.sqrt(h1n * h2n))  # Bhattacharyya coefficient
    return math.sqrt(max(0.0, 1.0 - bc))  # distance ~0..1


def _hash_distance(hashes_a: List[imagehash.ImageHash], hashes_b: List[imagehash.ImageHash]) -> float:
    if not hashes_a or not hashes_b:
        return 1.0  # maximal distance if missing
    # use minimum pairwise normalized Hamming (robust against one bad frame)
    min_bits = None
    hash_length = hashes_a[0].hash.size  # e.g. 256 for 16x16
    for ha in hashes_a:
        for hb in hashes_b:
            d = (ha - hb) / hash_length
            if min_bits is None or d < min_bits:
                min_bits = d
    return float(min_bits if min_bits is not None else 1.0)


def _combined_distance(a: SceneFeatures, b: SceneFeatures, w_hist: float, w_hash: float) -> float:
    dh = _bhattacharyya(a.hist, b.hist)
    dhash = _hash_distance(a.hashes, b.hashes)
    return w_hist * dh + w_hash * dhash


def _union_find(n: int):
    parent = list(range(n))
    size = [1] * n
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
    return find, union


def group_scenes(temp_dir: str, merged_dir: str, cfg: Dict) -> List[List[str]]:
    gcfg = cfg.get('grouping', {})
    show_progress = gcfg.get('show_progress', True)
    log_details = gcfg.get('log_details', True)

    files = [os.path.join(temp_dir, f) for f in sorted(os.listdir(temp_dir)) if f.lower().endswith(('.mp4', '.mkv', '.mov'))]
    if not files:
        print('[grouping] No segments found.')
        return []

    bins = gcfg.get('hist_bins', 32)
    hash_size = gcfg.get('hash_size', 16)
    weight_hist = gcfg.get('weight_hist', 0.5)
    weight_hash = gcfg.get('weight_hash', 0.5)
    threshold = gcfg.get('similarity_threshold', 0.45)
    min_cluster = gcfg.get('min_cluster_size', 2)
    skip_concat = gcfg.get('skip_concat', False)

    if log_details:
        print(f'[grouping] Features: bins={bins} hash_size={hash_size} weights=({weight_hist},{weight_hash}) threshold={threshold}')

    features: List[SceneFeatures] = []
    iterable = tqdm(files, desc='Features', unit='seg') if show_progress else files
    for f in iterable:
        try:
            feat = _compute_scene_features(f, hash_size=hash_size, bins=bins)
            features.append(feat)
        except Exception as e:
            if log_details:
                print(f'[grouping] Feature extraction failed for {f}: {e}')

    n = len(features)
    if n == 0:
        print('[grouping] No features extracted.')
        return []
    find, union = _union_find(n)

    pair_iter = range(n)
    outer_iter = tqdm(pair_iter, desc='Clustering', unit='row') if show_progress else pair_iter
    for i in outer_iter:
        for j in range(i + 1, n):
            d = _combined_distance(features[i], features[j], weight_hist, weight_hash)
            if d <= threshold:
                union(i, j)

    clusters_map: Dict[int, List[str]] = {}
    for idx, feat in enumerate(features):
        root = find(idx)
        clusters_map.setdefault(root, []).append(feat.file)

    clusters = list(clusters_map.values())
    for c in clusters:
        c.sort()

    if log_details:
        sizes = sorted([len(c) for c in clusters], reverse=True)
        print(f'[grouping] Total clusters: {len(clusters)} | Largest: {sizes[:5]}')

    os.makedirs(merged_dir, exist_ok=True)
    if skip_concat:
        if log_details:
            print('[grouping] Skipping concat (skip_concat=true).')
        return clusters

    merge_iter = enumerate(clusters, start=1)
    merge_iter = tqdm(list(merge_iter), desc='Merging', unit='cluster') if show_progress else merge_iter
    for ci, cluster in merge_iter:
        if len(cluster) < min_cluster:
            continue
        list_path = os.path.join(merged_dir, f'cluster_{ci:03d}.txt')
        out_path = os.path.join(merged_dir, f'cluster_{ci:03d}.mp4')
        try:
            with open(list_path, 'w', encoding='utf-8') as f:
                for seg in cluster:
                    abs_path = os.path.abspath(seg).replace('\\', '/')  # use forward slashes
                    safe_path = abs_path.replace("'", r"\'")  # escape single quotes for ffmpeg concat
                    f.write(f"file '{safe_path}'\n")
            cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_path, '-c', 'copy', out_path]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f'[grouping] Merge failed cluster_{ci:03d}: {e}')
    return clusters
