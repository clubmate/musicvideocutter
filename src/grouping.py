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

# Optional: scikit-learn nur laden, wenn KMeans Modus genutzt wird
try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
except ImportError:  # pragma: no cover
    KMeans = None
    PCA = None
    silhouette_score = None


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
    mode = gcfg.get('mode', 'simple').lower()
    if mode not in ('simple', 'kmeans'):
        print(f"[grouping] Unbekannter mode '{mode}', fallback auf 'simple'.")
        mode = 'simple'
    if mode == 'kmeans':
        return _group_kmeans(temp_dir, merged_dir, cfg)
    return _group_simple(temp_dir, merged_dir, cfg)


def _sample_frame_indices(total_frames: int, wanted: int) -> List[int]:
    if total_frames <= 0:
        return []
    if wanted <= 1:
        return [max(0, total_frames // 2)]
    positions = np.linspace(0, max(0, total_frames - 1), wanted)
    return sorted({int(round(p)) for p in positions})


def _merge_clusters(clusters: List[List[str]], merged_dir: str, gcfg: Dict, label: str):
    show_progress = gcfg.get('show_progress', True)
    log_details = gcfg.get('log_details', True)
    min_cluster = gcfg.get('min_cluster_size', 2)
    skip_concat = gcfg.get('skip_concat', False)
    os.makedirs(merged_dir, exist_ok=True)
    if skip_concat:
        if log_details:
            print(f'[{label}] skip_concat true - not merging.')
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
            print(f'[{label}] merge error cluster_{ci:03d}: {e}')
    return clusters


def _group_simple(temp_dir: str, merged_dir: str, cfg: Dict) -> List[List[str]]:
    gcfg = cfg.get('grouping', {})
    show_progress = gcfg.get('show_progress', True)
    log_details = gcfg.get('log_details', True)
    files = [os.path.join(temp_dir, f) for f in sorted(os.listdir(temp_dir)) if f.lower().endswith(('.mp4', '.mkv', '.mov'))]
    if not files:
        print('No segments found.')
        return []
    threshold = float(gcfg.get('simple_threshold', 0.88))
    hash_weight = float(gcfg.get('simple_hash_weight', 0.30))
    side = int(gcfg.get('simple_resize', 64))
    frame_count_cfg = int(gcfg.get('simple_frames', 3))
    if log_details:
        print(f"[grouping-simple] segments={len(files)} frames={frame_count_cfg} side={side} thr={threshold} hash_w={hash_weight}")
    avg_vectors: List[np.ndarray] = []
    hashes: List[List[imagehash.ImageHash]] = []
    iterable = tqdm(files, desc='Features', unit='seg') if show_progress else files
    for f in iterable:
        cap = cv2.VideoCapture(f)
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        idxs = _sample_frame_indices(fc, frame_count_cfg)
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
        avg = acc / valid if valid else acc
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
            cos = float(np.dot(vi, vj))
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
    return _merge_clusters(clusters, merged_dir, gcfg, 'grouping-simple')


def _group_kmeans(temp_dir: str, merged_dir: str, cfg: Dict) -> List[List[str]]:
    gcfg = cfg.get('grouping', {})
    show_progress = gcfg.get('show_progress', True)
    log_details = gcfg.get('log_details', True)
    if KMeans is None:
        print('[grouping-kmeans] scikit-learn nicht installiert. Bitte "pip install scikit-learn" ausführen.')
        return []
    files = [os.path.join(temp_dir, f) for f in sorted(os.listdir(temp_dir)) if f.lower().endswith(('.mp4', '.mkv', '.mov'))]
    if not files:
        print('No segments found.')
        return []
    frames_cfg = int(gcfg.get('kmeans_frames', 3))
    side = int(gcfg.get('kmeans_resize', 64))
    pca_components = int(gcfg.get('kmeans_pca_components', 0))
    pca_variance = float(gcfg.get('kmeans_pca_variance', 0.0))
    clusters_cfg = gcfg.get('kmeans_clusters', 'auto')
    auto_k = bool(gcfg.get('kmeans_auto_k', True))
    k_min = int(gcfg.get('kmeans_k_min', 2))
    k_max_cfg = gcfg.get('kmeans_k_max', 'auto')
    k_step = max(1, int(gcfg.get('kmeans_k_step', 1)))
    add_hash_bits = bool(gcfg.get('kmeans_add_hash_bits', True))
    hash_weight = float(gcfg.get('kmeans_hash_weight', 0.2))
    add_hsv_hist = bool(gcfg.get('kmeans_add_hsv_hist', True))
    hist_bins = int(gcfg.get('kmeans_hist_bins', 16))
    hist_weight = float(gcfg.get('kmeans_hist_weight', 0.3))
    do_scale = bool(gcfg.get('kmeans_scale', True))
    rnd_state = int(gcfg.get('kmeans_random_state', 42))
    export_csv = bool(gcfg.get('kmeans_export_csv', True))
    n_segments = len(files)
    if isinstance(clusters_cfg, str) and clusters_cfg.lower() == 'auto':
        k = max(2, int(round(np.sqrt(n_segments))))
    else:
        try:
            k = int(clusters_cfg)
        except Exception:
            k = max(2, int(round(np.sqrt(n_segments))))
    if k > n_segments:
        k = n_segments
    # Auto k Grenzen berechnen
    if isinstance(k_max_cfg, str) and k_max_cfg.lower() == 'auto':
        k_max = min(n_segments - 1, max(k_min, int(round(np.sqrt(n_segments) * 2))))
    else:
        try:
            k_max = int(k_max_cfg)
        except Exception:
            k_max = min(n_segments - 1, max(k_min, int(round(np.sqrt(n_segments) * 2))))
    if k_max < k_min:
        k_max = k_min
    if n_segments < 3:
        auto_k = False  # Silhouette nicht sinnvoll
    if log_details:
        print(f"[grouping-kmeans] segments={n_segments} frames={frames_cfg} side={side} k={k} auto_k={auto_k} range=[{k_min},{k_max}] pca={pca_components or ('var'+str(pca_variance) if pca_variance>0 else 0)} features: avg_rgb + {'hash' if add_hash_bits else ''}{' + hsv' if add_hsv_hist else ''}")
    feature_list: List[np.ndarray] = []
    hash_bit_list: List[np.ndarray] = []
    hsv_hist_list: List[np.ndarray] = []
    iterable = tqdm(files, desc='Features', unit='seg') if show_progress else files
    for f in iterable:
        cap = cv2.VideoCapture(f)
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        idxs = _sample_frame_indices(fc, frames_cfg)
        acc = np.zeros((side, side, 3), dtype=np.float32)
        valid = 0
        ph_bits_acc = []  # sammeln einzelner Hash-Bit Arrays
        hsv_acc = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (side, side), interpolation=cv2.INTER_AREA)
            acc += rgb.astype(np.float32)
            valid += 1
            if add_hash_bits:
                try:
                    ph = imagehash.phash(Image.fromarray(rgb), hash_size=8)
                    # Hash -> 0/1 Bits
                    bits = np.array(ph.hash, dtype=np.uint8).flatten().astype(np.float32)
                    ph_bits_acc.append(bits)
                except Exception:
                    pass
            if add_hsv_hist:
                hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
                # Einzelne Kanäle Histogramm
                h_hist = cv2.calcHist([hsv],[0],None,[hist_bins],[0,180])  # H 0..180
                s_hist = cv2.calcHist([hsv],[1],None,[hist_bins],[0,256])
                v_hist = cv2.calcHist([hsv],[2],None,[hist_bins],[0,256])
                h_hist = h_hist.reshape(-1)
                s_hist = s_hist.reshape(-1)
                v_hist = v_hist.reshape(-1)
                hv = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
                hv_sum = hv.sum() + 1e-9
                hv /= hv_sum
                hsv_acc.append(hv)
        cap.release()
        avg = acc / valid if valid else acc
        feat = avg.flatten().astype(np.float32)
        feature_list.append(feat)
        if add_hash_bits:
            if ph_bits_acc:
                hash_vec = np.mean(ph_bits_acc, axis=0)  # 0..1
            else:
                hash_vec = np.zeros(64, dtype=np.float32)
            hash_bit_list.append(hash_vec)
        if add_hsv_hist:
            if hsv_acc:
                hsv_vec = np.mean(hsv_acc, axis=0)  # schon normiert
            else:
                hsv_vec = np.zeros(hist_bins*3, dtype=np.float32)
            hsv_hist_list.append(hsv_vec)
    X = np.vstack(feature_list)
    blocks = [X]
    if add_hash_bits and hash_bit_list:
        hb = np.vstack(hash_bit_list)
        hb *= hash_weight  # simple scaling
        blocks.append(hb)
    if add_hsv_hist and hsv_hist_list:
        hv = np.vstack(hsv_hist_list)
        hv *= hist_weight
        blocks.append(hv)
    X_full = np.concatenate(blocks, axis=1)
    # Optionale Skalierung
    if do_scale:
        mean = X_full.mean(axis=0, keepdims=True)
        std = X_full.std(axis=0, keepdims=True) + 1e-9
        X_scaled = (X_full - mean) / std
    else:
        X_scaled = X_full
    if pca_components > 0 and PCA is not None and pca_components < X.shape[1]:
        try:
            pca = PCA(n_components=pca_components, random_state=rnd_state)
            X_proc = pca.fit_transform(X_scaled)
        except Exception as e:
            print(f"[grouping-kmeans] PCA Fehler: {e} -> verwende Originalfeatures")
            X_proc = X_scaled
    elif pca_variance > 0 and pca_variance < 1.0 and PCA is not None:
        try:
            pca = PCA(n_components=pca_variance, svd_solver='full', random_state=rnd_state)
            X_proc = pca.fit_transform(X_scaled)
            if log_details:
                print(f"[grouping-kmeans] PCA variance {pca_variance} -> comps={pca.n_components_}")
        except Exception as e:
            print(f"[grouping-kmeans] PCA(variance) Fehler: {e} -> verwende Originalfeatures")
            X_proc = X_scaled
    else:
        X_proc = X_scaled

    chosen_k = k
    best_sil = None
    best_labels = None
    inertias = {}
    sil_scores = {}
    if auto_k and silhouette_score is not None:
        candidate_ks = list(range(k_min, min(k_max, n_segments) + 1, k_step))
        candidate_ks = [ck for ck in candidate_ks if ck <= n_segments and ck >= 2]
        if len(candidate_ks) == 1:
            auto_k = False
        if auto_k:
            for ck in candidate_ks:
                try:
                    try:
                        km_tmp = KMeans(n_clusters=ck, random_state=rnd_state, n_init='auto')
                    except TypeError:
                        km_tmp = KMeans(n_clusters=ck, random_state=rnd_state)
                    km_tmp.fit(X_proc)
                    labels_tmp = km_tmp.labels_
                    # Silhouette nur wenn mehr als 1 Cluster und weniger als n_samples-1
                    sil = silhouette_score(X_proc, labels_tmp) if 1 < ck < n_segments else -1
                    sil_scores[ck] = sil
                    inertias[ck] = km_tmp.inertia_
                    if best_sil is None or sil > best_sil:
                        best_sil = sil
                        chosen_k = ck
                        best_labels = labels_tmp
                except Exception as e:
                    if log_details:
                        print(f"[grouping-kmeans] k={ck} Fehler: {e}")
    if best_labels is None:
        try:
            try:
                km = KMeans(n_clusters=chosen_k, random_state=rnd_state, n_init='auto')
            except TypeError:
                km = KMeans(n_clusters=chosen_k, random_state=rnd_state)
            km.fit(X_proc)
            labels = km.labels_
        except Exception as e:
            print(f"[grouping-kmeans] KMeans Fehler fallback: {e} -> jeder einzeln")
            labels = np.arange(n_segments)
    else:
        labels = best_labels
    clusters_map: Dict[int, List[str]] = {}
    for f, lbl in zip(files, labels):
        clusters_map.setdefault(int(lbl), []).append(f)
    clusters = list(clusters_map.values())
    for c in clusters:
        c.sort()
    if log_details:
        sizes = sorted([len(c) for c in clusters], reverse=True)
        msg = f"[grouping-kmeans] clusters={len(clusters)} largest={sizes[:5]} k={chosen_k}"
        if best_sil is not None:
            msg += f" silhouette={best_sil:.4f}"
        print(msg)
        if auto_k and sil_scores:
            best_pair = sorted(sil_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            print("[grouping-kmeans] top silhouette k: " + ", ".join(f"{kk}:{vv:.3f}" for kk,vv in best_pair))
    # CSV Export
    if export_csv:
        try:
            os.makedirs(merged_dir, exist_ok=True)
            csv_path = os.path.join(merged_dir, 'kmeans_clusters.csv')
            with open(csv_path, 'w', encoding='utf-8') as fh:
                fh.write('segment,cluster\n')
                for f, lbl in zip(files, labels):
                    fh.write(f"{os.path.basename(f)},{int(lbl)}\n")
            if log_details:
                print(f"[grouping-kmeans] CSV export -> {csv_path}")
        except Exception as e:
            print(f"[grouping-kmeans] CSV Export Fehler: {e}")
    return _merge_clusters(clusters, merged_dir, gcfg, 'grouping-kmeans')
