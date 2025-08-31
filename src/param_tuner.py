import os
import argparse
import numpy as np
import cv2
from PIL import Image
import imagehash
from itertools import combinations
from collections import defaultdict
from tqdm import tqdm

# Simple signature computation matching current simple grouping logic.

def sample_indices(frame_count: int, frame_num: int) -> list[int]:
    if frame_num <= 1:
        return [max(0, frame_count // 2)]
    if frame_count <= 1:
        return [0]
    positions = np.linspace(0, max(0, frame_count - 1), frame_num)
    return sorted({int(round(p)) for p in positions})


def compute_signature(path: str, side: int, frame_num: int):
    cap = cv2.VideoCapture(path)
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if fc <= 0:
        cap.release()
        return np.zeros(side*side*3, dtype=np.float32), []
    idxs = sample_indices(fc, frame_num)
    acc = np.zeros((side, side, 3), dtype=np.float32)
    valid = 0
    hashes = []
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
            hashes.append(imagehash.phash(Image.fromarray(rgb), hash_size=8))
        except Exception:
            pass
    cap.release()
    if valid > 0:
        acc /= valid
    vec = acc.flatten()
    nrm = np.linalg.norm(vec) + 1e-9
    vec = (vec / nrm).astype(np.float32)
    return vec, hashes


def hash_similarity(list_a, list_b):
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


def build_ground_truth(manual_dir: str):
    groups = {}
    for name in sorted(os.listdir(manual_dir)):
        gpath = os.path.join(manual_dir, name)
        if not os.path.isdir(gpath):
            continue
        files = [os.path.join(gpath, f) for f in sorted(os.listdir(gpath)) if f.lower().endswith('.mp4')]
        if files:
            groups[name] = files
    # Map file -> group
    file_to_group = {}
    for g, flist in groups.items():
        for f in flist:
            file_to_group[f] = g  # store full path
            # also store basename as fallback (in case of later normalization)
            base = os.path.basename(f)
            if base not in file_to_group:
                file_to_group[base] = g
    all_files = [f for flist in groups.values() for f in flist]
    return groups, file_to_group, all_files


def evaluate(all_files, file_to_group, signatures, hash_lists, hash_weight, thresholds):
    # Precompute pair similarities
    sims = {}
    intra = []
    inter = []
    for a, b in combinations(all_files, 2):
        va = signatures[a]
        vb = signatures[b]
        cos = float(np.dot(va, vb))
        h_sim = hash_similarity(hash_lists[a], hash_lists[b])
        sim = (1.0 - hash_weight) * cos + hash_weight * h_sim
        sims[(a, b)] = sim
        ga = file_to_group.get(a) or file_to_group.get(os.path.basename(a))
        gb = file_to_group.get(b) or file_to_group.get(os.path.basename(b))
        if ga is None or gb is None:
            # skip if mapping failed (should not happen)
            continue
        if ga == gb:
            intra.append(sim)
        else:
            inter.append(sim)
    intra_arr = np.array(intra) if intra else np.array([0.0])
    inter_arr = np.array(inter) if inter else np.array([0.0])

    results = []
    total_pos = len(intra)
    total_neg = len(inter)
    for thr in thresholds:
        tp = sum(1 for v in intra if v >= thr)
        fp = sum(1 for v in inter if v >= thr)
        fn = total_pos - tp
        tn = total_neg - fp
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        tpr = recall
        fpr = fp / (total_neg + 1e-9) if total_neg > 0 else 0.0
        youden = tpr - fpr
        results.append({
            'threshold': thr,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'youden': youden,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        })
    return results, intra_arr, inter_arr


def main():
    ap = argparse.ArgumentParser(description='Parameter tuner for simple grouping using manual ground truth.')
    ap.add_argument('--manual-dir', required=True, help='Path to manual-groups directory')
    ap.add_argument('--frames', default='1,3,5', help='Comma list frame counts to test')
    ap.add_argument('--hash-weights', default='0.2,0.3,0.4', help='Comma list hash weights')
    ap.add_argument('--sizes', default='48,64', help='Comma list resize side lengths')
    ap.add_argument('--thr-start', type=float, default=0.80)
    ap.add_argument('--thr-end', type=float, default=0.96)
    ap.add_argument('--thr-step', type=float, default=0.02)
    ap.add_argument('--top', type=int, default=5, help='Show top N configs by F1')
    ap.add_argument('--quiet', action='store_true', help='Minimal Ausgabe (nur Resultate)')
    ap.add_argument('--no-progress', action='store_true', help='Fortschrittsbalken unterdrücken')
    ap.add_argument('--threshold-progress', action='store_true', help='Zeigt optionalen Balken für Threshold-Scan pro Konfiguration')
    args = ap.parse_args()

    groups, file_to_group, all_files = build_ground_truth(args.manual_dir)
    if not all_files:
        print('No files found in manual groups.')
        return
    if not args.quiet:
        print(f"Loaded {len(groups)} groups with {len(all_files)} total segments.", flush=True)
        # Gruppengrößen
        g_sizes = sorted([len(v) for v in groups.values()], reverse=True)
        print(f"Group sizes: {g_sizes}", flush=True)
        pair_count = len(all_files) * (len(all_files)-1)//2
        print(f"Total pair comparisons per hash_weight set: {pair_count}", flush=True)

    frame_opts = [int(x) for x in args.frames.split(',') if x.strip()]
    hash_weights = [float(x) for x in args.hash_weights.split(',') if x.strip()]
    size_opts = [int(x) for x in args.sizes.split(',') if x.strip()]
    thresholds = np.arange(args.thr_start, args.thr_end + 1e-9, args.thr_step)

    best_rows = []
    total_configs = len(frame_opts) * len(hash_weights) * len(size_opts)
    cfg_idx = 0
    best_overall = None
    cfg_bar = None
    if not args.quiet and not args.no_progress:
        cfg_bar = tqdm(total=total_configs, desc='Configs', unit='cfg')

    for frame_num in frame_opts:
        for side in size_opts:
            signatures = {}
            hash_lists = {}
            for f in all_files:
                vec, hs = compute_signature(f, side, frame_num)
                signatures[f] = vec
                hash_lists[f] = hs
            for hw in hash_weights:
                cfg_idx += 1
                if cfg_bar:
                    cfg_bar.set_postfix(frames=frame_num, side=side, hash_w=hw)
                elif not args.quiet:
                    print(f"[{cfg_idx}/{total_configs}] frames={frame_num} side={side} hash_w={hw}", flush=True)

                if args.threshold_progress and not args.quiet:
                    thr_bar = tqdm(total=len(thresholds), desc='Thresholds', leave=False, unit='thr')
                    # manual evaluation with live update
                    # replicate evaluate logic but incremental
                    intra = []
                    inter = []
                    sims_cache = {}
                    # precompute all pair sims once
                    for a, b in combinations(all_files, 2):
                        va = signatures[a]; vb = signatures[b]
                        cos = float(np.dot(va, vb))
                        h_sim = hash_similarity(hash_lists[a], hash_lists[b])
                        sim_val = (1.0 - hw) * cos + hw * h_sim
                        sims_cache[(a, b)] = sim_val
                        if file_to_group[a] == file_to_group[b]:
                            intra.append(sim_val)
                        else:
                            inter.append(sim_val)
                    intra_arr = np.array(intra) if intra else np.array([0.0])
                    inter_arr = np.array(inter) if inter else np.array([0.0])
                    total_pos = len(intra)
                    total_neg = len(inter)
                    res = []
                    for thr in thresholds:
                        tp = sum(1 for v in intra if v >= thr)
                        fp = sum(1 for v in inter if v >= thr)
                        fn = total_pos - tp
                        tn = total_neg - fp
                        precision = tp / (tp + fp + 1e-9)
                        recall = tp / (tp + fn + 1e-9)
                        f1 = 2 * precision * recall / (precision + recall + 1e-9)
                        tpr = recall
                        fpr = fp / (total_neg + 1e-9) if total_neg > 0 else 0.0
                        youden = tpr - fpr
                        res.append({'threshold': thr,'precision': precision,'recall': recall,'f1': f1,'youden': youden})
                        thr_bar.update(1)
                    thr_bar.close()
                    best = max(res, key=lambda r: (r['f1'], r['youden']))
                    spread = float(np.mean(intra_arr) - np.mean(inter_arr))
                else:
                    res, intra_arr, inter_arr = evaluate(all_files, file_to_group, signatures, hash_lists, hw, thresholds)
                    best = max(res, key=lambda r: (r['f1'], r['youden']))
                    spread = float(np.mean(intra_arr) - np.mean(inter_arr))

                row = {
                    'frames': frame_num,
                    'side': side,
                    'hash_w': hw,
                    'best_thr': best['threshold'],
                    'f1': best['f1'],
                    'precision': best['precision'],
                    'recall': best['recall'],
                    'youden': best['youden'],
                    'intra_mean': float(np.mean(intra_arr)),
                    'inter_mean': float(np.mean(inter_arr)),
                    'mean_gap': spread,
                }
                best_rows.append(row)
                if best_overall is None or (row['f1'], row['youden'], row['mean_gap']) > (best_overall['f1'], best_overall['youden'], best_overall['mean_gap']):
                    best_overall = row
                    if not args.quiet:
                        msg = (f"New best -> frames={row['frames']} side={row['side']} hash_w={row['hash_w']} "
                               f"thr={row['best_thr']:.3f} F1={row['f1']:.3f} P={row['precision']:.3f} R={row['recall']:.3f}")
                        if cfg_bar:
                            cfg_bar.write(msg)
                        else:
                            print(msg, flush=True)
                if cfg_bar:
                    cfg_bar.update(1)
    if cfg_bar:
        cfg_bar.close()
    # sort
    best_rows.sort(key=lambda r: (r['f1'], r['youden'], r['mean_gap']), reverse=True)
    print('\nTop configurations:')
    for row in best_rows[:args.top]:
        print(f"frames={row['frames']} side={row['side']} hash_w={row['hash_w']} thr={row['best_thr']:.3f} "
              f"F1={row['f1']:.3f} P={row['precision']:.3f} R={row['recall']:.3f} gap={row['mean_gap']:.3f} "
              f"intra_mean={row['intra_mean']:.3f} inter_mean={row['inter_mean']:.3f}")

    # Recommendation (first entry)
    if best_rows:
        rec = best_rows[0]
        print('\nRecommended settings:')
        print(f"simple_frames: {rec['frames']}")
        print(f"simple_resize: {rec['side']}")
        print(f"simple_hash_weight: {rec['hash_w']}")
        print(f"simple_threshold: {rec['best_thr']:.3f}")

if __name__ == '__main__':
    main()
