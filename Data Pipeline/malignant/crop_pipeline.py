"""
crop_pipeline.py
----------------
Main centering and cropping pipeline for LIDC-IDRI malignant nodule crops.

Implements the full center-selection priority order:
  1. SEG centroid direct
  2. 7 variants of CSV x,y
  3. 7 variants of SEG centroid
  4. Known patient-specific overrides
  5. Full z-range scan (last resort for z_delta > 25mm)

All crops are saved as raw int16 DICOM files (no HU conversion).

Usage:
    python crop_pipeline.py
"""

import os
import csv
import json
import numpy as np
from dicom_utils import read_pixels_hu, crop_save_raw
from seg_extraction import seg_centroids_raw, build_z_index, get_slice_at_z


# ── Configuration ─────────────────────────────────────────────────────────────

DATASET   = "/path/to/LIDC-IDRI-lung-cancer-dataset/LIDC-IDRI"
SPLICED   = "/path/to/malignant working/final spliced"
CSV_PATH  = "/path/to/Patient Master/malignant_nodule_locations.csv"
OUT_BASE  = "/path/to/malignant working/clean malignant"

# Patients below this ID have no raw CT slices locally — use spliced copy
MIN_RAW_CT_PATIENT = 300

# Maximum z-delta (mm) before triggering full z-range scan
Z_DELTA_THRESHOLD = 25.0

# Air threshold — if center HU below this, position is in air
AIR_HU_THRESHOLD = -800.0

# Known patient-specific coordinate overrides
# Format: {patient_id: ('variant_name', 'source')}
# source = 'csv' (apply variant to CSV x,y) or 'seg' (apply variant to SEG cx,cy)
PATIENT_OVERRIDES = {
    'patient-0261': ('swapped', 'csv'),   # no SEG, empirically confirmed
    'patient-0332': ('swapped', 'seg'),   # SEG centroid in air
    'patient-0924': ('swapped', 'seg'),   # SEG centroid in air
}


# ── Tissue scoring ────────────────────────────────────────────────────────────

def tissue_score(hu):
    """Score tissue-likeness for coordinate variant selection."""
    if hu > 100:  return 0.30   # bone/vessel: present but not ideal
    if hu > -100: return 1.00   # solid nodule: peak
    if hu > -400: return 0.85   # soft tissue
    if hu > -700: return 0.70   # GGO: real nodule, accept
    if hu > -850: return 0.30   # sub-GGO: marginal
    return 0.00                 # air: reject


def tissue_score_v2(hu):
    """
    Tissue score with explicit bone penalty — use for full z-range scanning.
    Without the bone penalty, a rib (HU ~900) wins every time.
    """
    if -400 <= hu <= 200:
        return 1.0 - abs(hu - (-100)) / 300.0   # smooth peak at -100 HU
    if -700 <= hu < -400: return 0.60
    if -850 <= hu < -700: return 0.30
    if hu > 200: return max(0.10, 0.40 - (hu - 200) / 500)   # bone penalty
    return 0.00


def center_hu(hu_arr, cx, cy, r=12):
    """Mean HU in a radius-r patch around (cx, cy)."""
    cx, cy = int(round(cx)), int(round(cy))
    h, w   = hu_arr.shape
    x0, x1 = max(0, cx-r), min(w, cx+r)
    y0, y1 = max(0, cy-r), min(h, cy+r)
    patch  = hu_arr[y0:y1, x0:x1]
    return float(patch.mean()) if patch.size > 0 else -1000.0


# ── Coordinate variants ───────────────────────────────────────────────────────

def coord_variants(x, y, maxval=511):
    """
    7 possible interpretations of (x,y) in LIDC-IDRI pixel space.
    LIDC XML tools used different axis/origin conventions — test all 7.
    """
    x, y = float(x), float(y)
    return {
        'original':   (x,          y),
        'swapped':    (y,          x),
        'flip_x':     (maxval - x, y),
        'flip_y':     (x,          maxval - y),
        'flip_both':  (maxval - x, maxval - y),
        'swap_flipx': (maxval - y, x),
        'swap_flipy': (y,          maxval - x),
    }


def best_variant(hu_arr, x, y, score_fn=tissue_score):
    """Test all 7 variants of (x,y), return (best_cx, best_cy, best_score, variant_name)."""
    best_score, best_cx, best_cy, best_name = -1, x, y, 'original'
    for name, (cx, cy) in coord_variants(x, y).items():
        hu     = center_hu(hu_arr, cx, cy)
        score  = score_fn(hu)
        if score > best_score:
            best_score, best_cx, best_cy, best_name = score, cx, cy, name
    return best_cx, best_cy, best_score, best_name


# ── Full z-range scan ─────────────────────────────────────────────────────────

def full_z_scan(z_list, seg_cx, seg_cy, csv_x, csv_y):
    """
    Scan every 3rd CT slice across the entire z-range.
    For each slice, test all 7 variants of both SEG centroid and CSV x,y.
    Use tissue_score_v2 to avoid landing on ribs.

    Returns:
        (filepath, best_cx, best_cy, best_score)
    """
    best_score, best = -1, None

    candidates = []
    if seg_cx is not None and seg_cy is not None:
        candidates += list(coord_variants(seg_cx, seg_cy).values())
    if csv_x is not None and csv_y is not None:
        candidates += list(coord_variants(csv_x, csv_y).values())

    for i in range(0, len(z_list), 3):
        z, fpath = z_list[i]
        try:
            hu_arr, _, _ = read_pixels_hu(fpath)
        except Exception:
            continue
        for (cx, cy) in candidates:
            hu    = center_hu(hu_arr, cx, cy)
            score = tissue_score_v2(hu)
            if score > best_score:
                best_score = score
                best       = (fpath, cx, cy, score)

    return best if best else (None, None, None, 0.0)


# ── Main center selection ─────────────────────────────────────────────────────

def select_center(patient, nodule_id, z_center, csv_x, csv_y, pat_path):
    """
    Full priority-order center selection for one nodule.

    Priority:
      1. Patient override (known empirical fix)
      2. SEG centroid direct (if center_hu > AIR_HU_THRESHOLD)
      3. 7 variants of CSV x,y
      4. 7 variants of SEG centroid
      5. Full z-range scan (if z_delta > Z_DELTA_THRESHOLD)

    Returns:
        dict with keys: ct_path, cx, cy, method, score, center_hu_val
    """
    z_list = build_z_index(pat_path)
    seg_data = seg_centroids_raw(pat_path)
    seg = seg_data.get(nodule_id)

    # Find best CT slice by z
    ct_path, z_delta = get_slice_at_z(z_list, z_center) if z_list else (None, float('inf'))

    # ── 1. Patient-specific override ──────────────────────────────────────────
    if patient in PATIENT_OVERRIDES:
        variant_name, source = PATIENT_OVERRIDES[patient]
        if source == 'csv' and csv_x is not None:
            variants = coord_variants(csv_x, csv_y)
            cx, cy   = variants[variant_name]
            method   = f'override_csv_{variant_name}'
        elif source == 'seg' and seg:
            variants = coord_variants(seg['cx'], seg['cy'])
            cx, cy   = variants[variant_name]
            ct_path, z_delta = get_slice_at_z(z_list, seg['z'])
            method   = f'override_seg_{variant_name}'
        else:
            cx, cy, method = csv_x, csv_y, 'override_fallback'

        if ct_path:
            hu_arr, _, _ = read_pixels_hu(ct_path)
            hu_val = center_hu(hu_arr, cx, cy)
            return {'ct_path': ct_path, 'cx': cx, 'cy': cy,
                    'method': method, 'score': tissue_score(hu_val), 'center_hu_val': hu_val}

    # Load CT slice for scoring
    if ct_path:
        try:
            hu_arr, _, _ = read_pixels_hu(ct_path)
        except Exception:
            hu_arr = None
    else:
        hu_arr = None

    # ── 2. SEG centroid direct ────────────────────────────────────────────────
    if seg and z_list:
        seg_ct, seg_zdelta = get_slice_at_z(z_list, seg['z'])
        if seg_ct:
            seg_hu_arr, _, _ = read_pixels_hu(seg_ct)
            hu_val = center_hu(seg_hu_arr, seg['cx'], seg['cy'])
            if hu_val > AIR_HU_THRESHOLD:
                return {'ct_path': seg_ct, 'cx': seg['cx'], 'cy': seg['cy'],
                        'method': 'seg_direct', 'score': tissue_score(hu_val),
                        'center_hu_val': hu_val}

    # ── 3. CSV x,y variants ───────────────────────────────────────────────────
    if hu_arr is not None and csv_x is not None:
        cx, cy, score, name = best_variant(hu_arr, csv_x, csv_y, tissue_score)
        if score >= 0.30:
            hu_val = center_hu(hu_arr, cx, cy)
            return {'ct_path': ct_path, 'cx': cx, 'cy': cy,
                    'method': f'csv_{name}', 'score': score, 'center_hu_val': hu_val}

    # ── 4. SEG centroid variants ──────────────────────────────────────────────
    if seg and z_list:
        seg_ct, _ = get_slice_at_z(z_list, seg['z'])
        if seg_ct:
            seg_hu, _, _ = read_pixels_hu(seg_ct)
            cx, cy, score, name = best_variant(seg_hu, seg['cx'], seg['cy'], tissue_score_v2)
            if score >= 0.30:
                hu_val = center_hu(seg_hu, cx, cy)
                return {'ct_path': seg_ct, 'cx': cx, 'cy': cy,
                        'method': f'seg_variant_{name}', 'score': score,
                        'center_hu_val': hu_val}

    # ── 5. Full z-range scan ──────────────────────────────────────────────────
    if z_list and z_delta > Z_DELTA_THRESHOLD:
        seg_cx = seg['cx'] if seg else None
        seg_cy = seg['cy'] if seg else None
        fpath, cx, cy, score = full_z_scan(z_list, seg_cx, seg_cy, csv_x, csv_y)
        if fpath:
            scan_hu, _, _ = read_pixels_hu(fpath)
            hu_val = center_hu(scan_hu, cx, cy)
            return {'ct_path': fpath, 'cx': cx, 'cy': cy,
                    'method': 'full_z_scan', 'score': score, 'center_hu_val': hu_val}

    # ── Fallback: use whatever we have ────────────────────────────────────────
    fallback_cx = csv_x if csv_x is not None else (seg['cx'] if seg else 256)
    fallback_cy = csv_y if csv_y is not None else (seg['cy'] if seg else 256)
    hu_val = center_hu(hu_arr, fallback_cx, fallback_cy) if hu_arr is not None else -1000.0
    return {'ct_path': ct_path, 'cx': fallback_cx, 'cy': fallback_cy,
            'method': 'fallback', 'score': tissue_score(hu_val), 'center_hu_val': hu_val}


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(csv_path=CSV_PATH, out_base=OUT_BASE, dataset=DATASET, spliced=SPLICED,
                 log_path='/tmp/pipeline_log.json'):
    """
    Process all nodules in the CSV and write crops to out_base/.

    For patients < MIN_RAW_CT_PATIENT: copies from spliced/ (no raw CT available).
    For patients >= MIN_RAW_CT_PATIENT: re-crops from raw CT using select_center().
    """
    rows = []
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            rows.append(row)

    log    = []
    copied = 0
    fixed  = 0
    errors = []

    for row in rows:
        patient   = row['patient']
        nodule_id = int(row['nodule_id'])
        z_center  = float(row['z_center_mm'])
        csv_x     = float(row['x_center_px']) if row['x_center_px'] else None
        csv_y     = float(row['y_center_px']) if row['y_center_px'] else None

        out_path   = os.path.join(out_base, patient, f'nodule_{nodule_id}.dcm')
        spliced_path = os.path.join(spliced, patient, f'nodule_{nodule_id}.dcm')
        pat_num    = int(patient.replace('patient-', ''))
        pat_path   = os.path.join(dataset, patient)

        # Patients < 300: no raw CT, copy from spliced
        if pat_num < MIN_RAW_CT_PATIENT:
            import shutil
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            shutil.copy2(spliced_path, out_path)
            log.append({'patient': patient, 'nodule': nodule_id, 'method': 'spliced_copy'})
            copied += 1
            continue

        # Patients >= 300: re-crop from raw CT
        result = select_center(patient, nodule_id, z_center, csv_x, csv_y, pat_path)

        if result['ct_path'] is None:
            import shutil
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            shutil.copy2(spliced_path, out_path)
            log.append({'patient': patient, 'nodule': nodule_id, 'method': 'spliced_fallback',
                        'reason': 'no_ct_found'})
            copied += 1
            continue

        ok, msg = crop_save_raw(result['ct_path'], out_path, result['cx'], result['cy'])
        entry = {**result, 'patient': patient, 'nodule': nodule_id, 'ok': ok}
        if not ok:
            entry['error'] = msg
            errors.append(entry)
        else:
            fixed += 1
        log.append(entry)

    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)

    print(f"Done. Copied from spliced: {copied} | Re-cropped: {fixed} | Errors: {len(errors)}")
    return log


if __name__ == '__main__':
    run_pipeline()
