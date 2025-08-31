## (Removed advanced multi-feature implementation; simplified variant only)
# Simplified grouping: 3 frame screenshots -> average image + pHash.
# Steps per segment:
# 1. Pick first / middle / last frame
# 2. Resize to side x side
# 3. Average & L2-normalize -> feature vector
# 4. pHash (8x8) each frame
# 5. Similarity = (1-hash_weight)*cosine + hash_weight*best_hash_similarity
# 6. Union-Find merges if similarity >= threshold
# Config keys: simple_threshold, simple_hash_weight, simple_resize,
#   min_cluster_size, show_progress, log_details, skip_concat.

import os
import numpy as np
import cv2
from typing import List, Dict
from PIL import Image
import imagehash
import subprocess
from tqdm import tqdm


def _union_find(n: int):
    parent = list(range(n))
    size = [1]*n
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
        print('No segments found.')
        return []

    min_cluster = gcfg.get('min_cluster_size', 2)
    skip_concat = gcfg.get('skip_concat', False)
    threshold = float(gcfg.get('simple_threshold', 0.88))
    hash_weight = float(gcfg.get('simple_hash_weight', 0.30))
    side = int(gcfg.get('simple_resize', 64))
    frame_count_cfg = int(gcfg.get('simple_frames', 3))  # 1 => nur Mitte, >=2 => gleichmäßig verteilt

    if log_details:
        print(f"framecount={frame_count_cfg} segments={len(files)} side={side} thr={threshold} hash_w={hash_weight}")

    avg_vectors: List[np.ndarray] = []
    hashes: List[List[imagehash.ImageHash]] = []
    iterable = tqdm(files, desc='Features', unit='seg') if show_progress else files
    for f in iterable:
        cap = cv2.VideoCapture(f)
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if fc <= 0:
            avg_vectors.append(np.zeros(side*side*3, dtype=np.float32))
            hashes.append([])
            cap.release()
            continue
        if frame_count_cfg <= 1:
            idxs = [max(0, fc//2)]
        else:
            # Gleichmäßig verteilte Indizes inklusive Start/Ende
            positions = np.linspace(0, max(0, fc-1), frame_count_cfg)
            idxs = sorted({int(round(p)) for p in positions})
        acc = np.zeros((side, side, 3), dtype=np.float32)
        valid = 0
        ph_list: List[imagehash.ImageHash] = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (side, side), interpolation=cv2.INTER_AREA)
            acc += rgb.astype(np.float32)
            valid += 1
            try:
                ph_list.append(imagehash.phash(Image.fromarray(rgb), hash_size=8))
            except Exception:
                pass
        cap.release()
        if valid == 0:
            avg = acc
        else:
            avg = acc / valid
        vec = avg.flatten()
        nrm = np.linalg.norm(vec) + 1e-9
        vec = (vec / nrm).astype(np.float32)
        avg_vectors.append(vec)
        hashes.append(ph_list)

    def hash_sim(list_a: List[imagehash.ImageHash], list_b: List[imagehash.ImageHash]) -> float:
        if not list_a or not list_b:
            return 0.0
        best = None
        max_bits = list_a[0].hash.size
        for ha in list_a:
            for hb in list_b:
                d = (ha - hb)
                sim = 1.0 - (d / max_bits)
                if best is None or sim > best:
                    best = sim
        return float(best if best is not None else 0.0)

    n = len(files)
    find, union = _union_find(n)
    for i in range(n):
        vi = avg_vectors[i]
        for j in range(i+1, n):
            vj = avg_vectors[j]
            cos = float(np.dot(vi, vj))  # vectors normalized
            h_sim = hash_sim(hashes[i], hashes[j])
            sim = (1.0 - hash_weight) * cos + hash_weight * h_sim
            if sim >= threshold:
                union(i, j)

    clusters_map: Dict[int, List[str]] = {}
    for idx, f in enumerate(files):
        r = find(idx)
        clusters_map.setdefault(r, []).append(f)
    clusters = list(clusters_map.values())
    for c in clusters:
        c.sort()

    if log_details:
        sizes = sorted([len(c) for c in clusters], reverse=True)
        print(f"[grouping-simple] clusters={len(clusters)} largest={sizes[:5]} thr={threshold}")

    os.makedirs(merged_dir, exist_ok=True)
    if skip_concat:
        if log_details:
            print('[grouping-simple] skip_concat true - not merging.')
        return clusters

    merge_iter = enumerate(clusters, start=1)
    merge_iter = tqdm(list(merge_iter), desc='Merging', unit='cluster') if show_progress else merge_iter
    for ci, cluster in merge_iter:
        if len(cluster) < min_cluster:
            continue
        list_path = os.path.join(merged_dir, f'cluster_{ci:03d}.txt')
        out_path = os.path.join(merged_dir, f'cluster_{ci:03d}.mp4')
        try:
            with open(list_path, 'w', encoding='utf-8') as fh:
                for seg in cluster:
                    ap = os.path.abspath(seg).replace('\\', '/')
                    ap = ap.replace("'", r"\'")
                    fh.write(f"file '{ap}'\n")
            subprocess.run(['ffmpeg','-y','-f','concat','-safe','0','-i',list_path,'-c','copy',out_path],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[grouping-simple] merge error cluster_{ci:03d}: {e}")
    return clusters
