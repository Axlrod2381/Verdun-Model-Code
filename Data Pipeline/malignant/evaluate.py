"""
evaluate.py
-----------
Evaluate all images in a nodule crop folder for CNN training suitability.

Metrics (all use lung background as reference — NOT annulus):
  - center_hu:        mean HU in centre 20×20 patch
  - lung_hu:          mean HU of pixels in [-980, -600] (pure airspace)
  - nodule_contrast:  center_hu - lung_hu (always positive for real nodule)
  - centroid_offset:  centre-of-mass of above-lung signal in centre 70×70 window
  - nodule_area:      pixels > lung_hu+50 in centre 70×70
  - ntype:            solid / part-solid / GGO / sub-GGO / air-centered
  - cnn_score:        weighted learnability score 0-1

Usage:
    python evaluate.py --folder /path/to/clean_malignant --csv /path/to/locations.csv
    python evaluate.py  (uses defaults)
"""

import os
import json
import csv
import argparse
import numpy as np
from dicom_utils import read_pixels_hu


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_FOLDER  = "/path/to/malignant working/clean malignant"
DEFAULT_CSV     = "/path/to/Patient Master/malignant_nodule_locations.csv"
DEFAULT_OUT     = "/tmp/eval_v2.json"


# ── Per-image analysis ────────────────────────────────────────────────────────

def analyze(hu_arr):
    """
    Compute all quality metrics for a single 150×150 HU crop.

    Key insight: use lung background (pixels -980 to -600) as contrast
    reference — NOT the surrounding annulus. Chest wall in the annulus
    gives negative contrast for GGO nodules (false 'no signal').

    Returns dict of metrics.
    """
    h, w = hu_arr.shape

    # 1. Centre HU (20×20 patch)
    c = h // 2
    center_hu_val = float(hu_arr[c-10:c+10, c-10:c+10].mean())

    # 2. Lung background — pure airspace reference
    lung_mask = (hu_arr >= -980) & (hu_arr <= -600)
    lung_hu   = float(hu_arr[lung_mask].mean()) if lung_mask.sum() > 10 else -900.0

    # 3. Contrast above lung background (always positive for any real nodule)
    nodule_contrast = center_hu_val - lung_hu

    # 4. Centroid offset — how far the nodule signal is from image centre
    #    Work in centre 70×70 to avoid chest wall edges
    pad  = (h - 70) // 2
    roi  = hu_arr[pad:pad+70, pad:pad+70]
    above = np.clip(roi - lung_hu, 0, None)
    total = above.sum()
    if total > 0:
        ys, xs   = np.mgrid[0:70, 0:70]
        cx_roi   = float((xs * above).sum() / total)
        cy_roi   = float((ys * above).sum() / total)
        offset   = float(np.sqrt((cx_roi - 35)**2 + (cy_roi - 35)**2))
    else:
        offset = 0.0   # no signal above lung = air-centred, offset undefined

    # 5. Nodule area (pixels > lung_hu+50 in centre 70×70)
    nodule_area = int((above > 50).sum())

    # 6. Nodule type
    if center_hu_val > -100:   ntype = 'solid'
    elif center_hu_val > -400: ntype = 'part-solid'
    elif center_hu_val > -700: ntype = 'GGO'
    elif center_hu_val > -850: ntype = 'sub-GGO'
    else:                      ntype = 'air-centered'

    return {
        'center_hu':       center_hu_val,
        'lung_hu':         lung_hu,
        'nodule_contrast': nodule_contrast,
        'centroid_offset': offset,
        'nodule_area':     nodule_area,
        'ntype':           ntype,
    }


def cnn_score(metrics):
    """
    CNN learnability score (0–1).
    Weights: contrast 40%, centering 25%, area 15%, type 20%.
    """
    # Contrast score: +500 HU above lung = perfect
    contrast_score = min(1.0, max(0.0, metrics['nodule_contrast'] / 500.0))

    # Centering score: < 10px = excellent, > 35px = poor
    offset = metrics['centroid_offset']
    if   offset < 10: center_score = 1.00
    elif offset < 20: center_score = 0.80
    elif offset < 35: center_score = 0.50
    else:             center_score = 0.10

    # Area score: 400+ px = full marks
    area_score = min(1.0, metrics['nodule_area'] / 400.0)

    # Type score
    type_scores = {
        'solid':       1.00,
        'part-solid':  0.85,
        'GGO':         0.70,
        'sub-GGO':     0.45,
        'air-centered':0.00,
    }
    type_score = type_scores.get(metrics['ntype'], 0.0)

    return round(0.40*contrast_score + 0.25*center_score +
                 0.15*area_score    + 0.20*type_score, 4)


# ── Lung window for display ───────────────────────────────────────────────────

def lung_window(hu_arr):
    """
    Convert HU array to uint8 using standard lung window.
    WC = -600, WW = 1500  →  display range [-1350, +150]
    """
    wmin = -1350.0
    wmax =  150.0
    clipped = np.clip(hu_arr, wmin, wmax)
    return ((clipped - wmin) / (wmax - wmin) * 255).astype(np.uint8)


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_evaluation(folder=DEFAULT_FOLDER, csv_path=DEFAULT_CSV, out_path=DEFAULT_OUT):
    """
    Evaluate every .dcm in folder/, enriched with reader count and rating from CSV.
    Writes results to out_path as JSON.
    """
    # Load CSV metadata
    meta = {}
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            key = (row['patient'], row['nodule_id'])
            meta[key] = {
                'readers': int(row.get('n_readers', 0)),
                'rating':  float(row.get('mean_rating', 0)),
            }

    results = []
    for patient in sorted(os.listdir(folder)):
        pat_dir = os.path.join(folder, patient)
        if not os.path.isdir(pat_dir):
            continue
        for fname in sorted(os.listdir(pat_dir)):
            if not fname.endswith('.dcm'):
                continue
            nodule_id = fname.replace('nodule_', '').replace('.dcm', '')
            fpath     = os.path.join(pat_dir, fname)
            try:
                hu_arr, _, _ = read_pixels_hu(fpath)
                m = analyze(hu_arr)
                m['cnn_score'] = cnn_score(m)
                m['patient']   = patient
                m['nodule']    = nodule_id
                m.update(meta.get((patient, nodule_id), {'readers': 0, 'rating': 0.0}))
                results.append(m)
            except Exception as e:
                results.append({'patient': patient, 'nodule': nodule_id,
                                'error': str(e), 'cnn_score': 0.0})

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    scores  = [r['cnn_score'] for r in results if 'error' not in r]
    harmful = [r for r in results if r.get('ntype') == 'air-centered']
    print(f"Evaluated: {len(results)} images")
    print(f"Mean CNN score: {np.mean(scores):.3f}  Median: {np.median(scores):.3f}")
    print(f"CNN score >= 0.55: {sum(s>=0.55 for s in scores)} ({100*sum(s>=0.55 for s in scores)/len(scores):.1f}%)")
    print(f"Air-centered (HARMFUL): {len(harmful)} ({100*len(harmful)/len(results):.1f}%)")
    print(f"Results saved to {out_path}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default=DEFAULT_FOLDER)
    parser.add_argument('--csv',    default=DEFAULT_CSV)
    parser.add_argument('--out',    default=DEFAULT_OUT)
    args = parser.parse_args()
    run_evaluation(args.folder, args.csv, args.out)
