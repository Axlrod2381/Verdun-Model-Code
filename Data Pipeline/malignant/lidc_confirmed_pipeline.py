"""
lidc_confirmed_pipeline.py
--------------------------
Multi-slice malignant nodule pipeline for the LIDC confirmed folder.
"""

import os, sys, re, csv, struct, math
import numpy as np

LIDC_ROOT = "/sessions/optimistic-gifted-noether/mnt/LIDC confirmed/manifest-1782803489569/LIDC-IDRI"
CSV_PATH  = "/sessions/optimistic-gifted-noether/mnt/Patient Master/malignant_nodule_locations.csv"
OUT_ROOT  = "/sessions/optimistic-gifted-noether/mnt/malignant working /LIDC confirmed crops"

TAG_ROWS      = b'\x28\x00\x10\x00'
TAG_COLS      = b'\x28\x00\x11\x00'
TAG_INTERCEPT = b'\x28\x00\x52\x10'
TAG_SLOPE     = b'\x28\x00\x53\x10'
TAG_IMG_POS   = b'\x20\x00\x32\x00'
TAG_PIXEL     = b'\xe0\x7f\x10\x00'
EXTENDED_VR   = {b'OB', b'OW', b'OF', b'SQ', b'UC', b'UN', b'UR', b'UT'}


def read_ds_tag(raw, tag4):
    raw_b = bytes(raw)
    idx = raw_b.find(tag4)
    while idx >= 0:
        vr = raw_b[idx+4:idx+6]
        if len(vr) == 2 and vr[0:1].isalpha() and vr[1:2].isalpha():
            length = struct.unpack('<H', raw_b[idx+6:idx+8])[0]
            if 0 < length < 400:
                return raw_b[idx+8:idx+8+length].decode('latin-1','replace').strip('\x00 ')
        else:
            length = struct.unpack('<I', raw_b[idx+4:idx+8])[0]
            if 0 < length < 400:
                return raw_b[idx+8:idx+8+length].decode('latin-1','replace').strip('\x00 ')
        idx = raw_b.find(tag4, idx+1)
    return None

def read_uint16_tag(raw, tag4):
    raw_b = bytes(raw)
    idx = raw_b.find(tag4)
    while idx >= 0:
        vr = raw_b[idx+4:idx+6]
        if vr[0:1].isalpha() and vr[1:2].isalpha():
            length = struct.unpack('<H', raw_b[idx+6:idx+8])[0]
            if length == 2:
                return struct.unpack('<H', raw_b[idx+8:idx+10])[0]
        else:
            length = struct.unpack('<I', raw_b[idx+4:idx+8])[0]
            if length == 2:
                return struct.unpack('<H', raw_b[idx+8:idx+10])[0]
        idx = raw_b.find(tag4, idx+1)
    return None

def read_raw_px(filepath):
    with open(filepath, 'rb') as f:
        raw = f.read()
    rows = read_uint16_tag(raw, TAG_ROWS) or 512
    cols = read_uint16_tag(raw, TAG_COLS) or 512
    pix_idx = raw.rfind(TAG_PIXEL)
    if pix_idx < 0:
        raise ValueError("No pixel data tag")
    vr = raw[pix_idx+4:pix_idx+6]
    hdr_end = pix_idx + 12 if bytes(vr) in EXTENDED_VR else pix_idx + 8
    pdata = raw[hdr_end:]
    expected = rows * cols * 2
    if len(pdata) >= expected:
        pdata = pdata[:expected]
    return np.frombuffer(pdata, dtype=np.int16).reshape(rows, cols), rows, cols

def read_pixels_hu(filepath):
    arr, rows, cols = read_raw_px(filepath)
    with open(filepath, 'rb') as f:
        raw = f.read()
    intercept_s = read_ds_tag(raw, TAG_INTERCEPT)
    slope_s     = read_ds_tag(raw, TAG_SLOPE)
    intercept   = float(intercept_s) if intercept_s else -1024.0
    slope       = float(slope_s)     if slope_s     else 1.0
    return arr.astype(np.float32) * slope + intercept, rows, cols

def read_z(filepath):
    with open(filepath, 'rb') as f:
        raw = f.read(8192)
    pos_s = read_ds_tag(raw, TAG_IMG_POS)
    if pos_s:
        parts = pos_s.split('\\')
        if len(parts) >= 3:
            try:
                return float(parts[2])
            except ValueError:
                pass
    return None

def crop_save_raw(ct_path, out_path, cx, cy, size=150):
    try:
        with open(ct_path, 'rb') as f:
            raw = bytearray(f.read())
        raw_arr, rows, cols = read_raw_px(ct_path)
        cx, cy = int(round(cx)), int(round(cy))
        half = size // 2
        x0 = max(0, min(cx - half, cols - size))
        y0 = max(0, min(cy - half, rows - size))
        crop = raw_arr[y0:y0+size, x0:x0+size]
        for tag in [TAG_ROWS, TAG_COLS]:
            idx = bytes(raw).find(tag)
            if idx >= 0:
                raw[idx+8:idx+10] = struct.pack('<H', size)
        raw_b   = bytes(raw)
        pix_idx = raw_b.rfind(TAG_PIXEL)
        vr      = raw_b[pix_idx+4:pix_idx+6]
        hdr_end = pix_idx + 12 if bytes(vr) in EXTENDED_VR else pix_idx + 8
        new_pix = crop.tobytes()
        raw[hdr_end-4:hdr_end] = struct.pack('<I', len(new_pix))
        out_bytes = bytes(raw[:hdr_end]) + new_pix
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'wb') as f:
            f.write(out_bytes)
        return True, "ok"
    except Exception as e:
        return False, str(e)

SEG_FOLDER_RE = re.compile(r'[Ss]egmentation of [Nn]odule\s+(\d+)', re.IGNORECASE)

def decode_seg_pixels(pdata, rows, cols):
    n16 = rows * cols * 2
    if len(pdata) >= n16:
        a = np.frombuffer(pdata[:n16], dtype=np.uint16).reshape(rows, cols)
        if a.max() <= 1:
            return a.astype(np.uint8)
    n8 = rows * cols
    if len(pdata) >= n8:
        a = np.frombuffer(pdata[:n8], dtype=np.uint8).reshape(rows, cols)
        if a.max() <= 1:
            return a
    needed = math.ceil(rows * cols / 8)
    if len(pdata) >= needed:
        bits = np.unpackbits(np.frombuffer(pdata[:needed], dtype=np.uint8))
        return bits[:rows*cols].reshape(rows, cols).astype(np.uint8)
    return None

def extract_seg_centroids(study_path):
    result = {}
    for series_name in os.listdir(study_path):
        m = SEG_FOLDER_RE.search(series_name)
        if not m:
            continue
        nodule_n = int(m.group(1))
        seg_dir  = os.path.join(study_path, series_name)
        if not os.path.isdir(seg_dir):
            continue
        best = None
        for fname in sorted(os.listdir(seg_dir)):
            if not fname.lower().endswith('.dcm'):
                continue
            fpath = os.path.join(seg_dir, fname)
            try:
                with open(fpath, 'rb') as f:
                    raw = f.read()
                rows = read_uint16_tag(raw, TAG_ROWS) or 512
                cols = read_uint16_tag(raw, TAG_COLS) or 512
                pos_s = read_ds_tag(raw, TAG_IMG_POS)
                z = 0.0
                if pos_s and '\\' in pos_s:
                    try:
                        z = float(pos_s.split('\\')[2])
                    except (IndexError, ValueError):
                        pass
                pix_idx = raw.rfind(TAG_PIXEL)
                if pix_idx < 0:
                    continue
                vr = raw[pix_idx+4:pix_idx+6]
                data_start = pix_idx + 12 if bytes(vr) in EXTENDED_VR else pix_idx + 8
                mask = decode_seg_pixels(raw[data_start:], rows, cols)
                if mask is None or mask.sum() == 0:
                    continue
                ys, xs = np.where(mask > 0)
                n_px = int(mask.sum())
                cx_s = float(xs.mean())
                cy_s = float(ys.mean())
                if best is None or n_px > best['n_px']:
                    best = {'z': z, 'cx': cx_s, 'cy': cy_s, 'n_px': n_px}
            except Exception:
                continue
        if best:
            if nodule_n not in result or best['n_px'] > result[nodule_n]['n_px']:
                result[nodule_n] = best
    return result

def find_ct_series(study_path):
    best_path  = None
    best_count = 0
    for series_name in os.listdir(study_path):
        series_path = os.path.join(study_path, series_name)
        if not os.path.isdir(series_path):
            continue
        lower = series_name.lower()
        if 'segmentation' in lower or 'annotation' in lower or 'evaluations' in lower or 'nodule' in lower:
            continue
        dcms = [f for f in os.listdir(series_path) if f.lower().endswith('.dcm')]
        if len(dcms) > best_count:
            best_count = len(dcms)
            best_path  = series_path
    return best_path if best_count >= 10 else None

def find_ct_study(pat_path):
    best_study = None
    best_count = 0
    for study_name in os.listdir(pat_path):
        study_path = os.path.join(pat_path, study_name)
        if not os.path.isdir(study_path) or study_name.startswith('.'):
            continue
        ct = find_ct_series(study_path)
        if ct:
            n = len([f for f in os.listdir(ct) if f.lower().endswith('.dcm')])
            if n > best_count:
                best_count = n
                best_study = study_path
    return best_study

def build_z_index(ct_series_path):
    entries = []
    for fname in os.listdir(ct_series_path):
        if not fname.lower().endswith('.dcm'):
            continue
        fpath = os.path.join(ct_series_path, fname)
        z = read_z(fpath)
        if z is not None:
            entries.append((z, fpath))
    entries.sort(key=lambda x: x[0])
    return entries

def get_slice_at_z(z_list, target_z):
    if not z_list:
        return None
    diffs  = [abs(z - target_z) for z, _ in z_list]
    idx    = int(np.argmin(diffs))
    z_val, fpath = z_list[idx]
    return idx, z_val, fpath

def tissue_score_v2(hu):
    if -400 <= hu <= 200:
        return 1.0 - abs(hu - (-100)) / 300.0
    if -700 <= hu < -400:
        return 0.6
    if -850 <= hu < -700:
        return 0.3
    if hu > 200:
        return max(0.1, 0.4 - (hu - 200) / 500)
    return 0.0

def center_hu_val(hu_arr, cx, cy, radius=12):
    size = hu_arr.shape[0]
    ys, xs = np.ogrid[:size, :size]
    mask = ((xs - cx)**2 + (ys - cy)**2) <= radius**2
    vals = hu_arr[mask]
    return float(vals.mean()) if len(vals) > 0 else -1024.0

def coord_variants(x, y, maxval=511):
    return {
        'original':   (x,          y),
        'swapped':    (y,          x),
        'flip_x':     (maxval - x, y),
        'flip_y':     (x,          maxval - y),
        'flip_both':  (maxval - x, maxval - y),
        'swap_flipx': (maxval - y, x),
        'swap_flipy': (y,          maxval - x),
    }

def best_coord_variant(hu_arr, x, y, radius=12):
    variants = coord_variants(float(x), float(y))
    best_name = 'original'
    best_score = -1
    best_cx, best_cy = float(x), float(y)
    rows_arr, cols_arr = hu_arr.shape
    for name, (vx, vy) in variants.items():
        vx, vy = int(round(vx)), int(round(vy))
        if vx < 0 or vy < 0 or vx >= cols_arr or vy >= rows_arr:
            continue
        hu = center_hu_val(hu_arr, vx, vy, radius)
        sc = tissue_score_v2(hu)
        if sc > best_score:
            best_score = sc
            best_name  = name
            best_cx, best_cy = vx, vy
    return best_cx, best_cy, best_name

def qc_slice(hu_arr, cx, cy):
    size = hu_arr.shape[0]
    lung_mask = (hu_arr >= -980) & (hu_arr <= -600)
    lung_hu_val = float(hu_arr[lung_mask].mean()) if lung_mask.sum() > 10 else -950.0
    chu = center_hu_val(hu_arr, cx, cy, 12)
    contrast = chu - lung_hu_val
    half70 = 35
    cx70   = size // 2
    cy70   = size // 2
    x0 = max(0, cx70 - half70); y0 = max(0, cy70 - half70)
    x1 = min(size, cx70 + half70); y1 = min(size, cy70 + half70)
    roi    = hu_arr[y0:y1, x0:x1].astype(np.float32)
    signal = np.maximum(roi - lung_hu_val, 0)
    total  = signal.sum()
    if total > 0:
        ys_roi, xs_roi = np.where(signal > 0)
        com_x = float((xs_roi * signal[ys_roi, xs_roi]).sum() / total) + x0
        com_y = float((ys_roi * signal[ys_roi, xs_roi]).sum() / total) + y0
        offset = math.sqrt((com_x - cx70)**2 + (com_y - cy70)**2)
    else:
        offset = 999.0
    passed = (contrast >= 200) and (chu > -820) and (offset < 15)
    return passed, {
        'contrast':  round(contrast, 1),
        'center_hu': round(chu, 1),
        'lung_hu':   round(lung_hu_val, 1),
        'offset_px': round(offset, 1),
        'passed':    passed,
    }

def run_pipeline():
    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(CSV_PATH) as f:
        csv_rows = list(csv.DictReader(f))

    patient_nodules = {}
    for row in csv_rows:
        pat = row['patient']
        nid = int(row['nodule_id'])
        def safe_float(v):
            try: return float(v) if v and v.strip() not in ('', 'N/A', 'None') else None
            except: return None
        def safe_int(v):
            try: return int(float(v)) if v and v.strip() not in ('', 'N/A', 'None') else 0
            except: return 0
        patient_nodules.setdefault(pat, []).append({
            'nodule_id':   nid,
            'z_csv':       safe_float(row.get('z_center_mm')),
            'x_csv':       safe_float(row.get('x_center_px')),
            'y_csv':       safe_float(row.get('y_center_px')),
            'n_readers':   safe_int(row.get('n_readers')),
            'mean_rating': safe_float(row.get('mean_rating')) or 0.0,
        })

    all_patients = sorted(p for p in os.listdir(LIDC_ROOT) if p.startswith('LIDC-IDRI-'))
    log_lines = []
    total_kept = 0; total_dropped = 0; total_errors = 0

    for lidc_name in all_patients:
        pat_num  = lidc_name.replace('LIDC-IDRI-', '')
        pat_key  = f'patient-{pat_num}'
        pat_path = os.path.join(LIDC_ROOT, lidc_name)

        nodules = patient_nodules.get(pat_key)
        if not nodules:
            log_lines.append(f"[SKIP] {pat_key}: no malignant nodules in CSV")
            continue

        study_path = find_ct_study(pat_path)
        if not study_path:
            log_lines.append(f"[ERROR] {pat_key}: cannot find CT study")
            total_errors += 1; continue

        ct_series = find_ct_series(study_path)
        if not ct_series:
            log_lines.append(f"[ERROR] {pat_key}: no CT series in {os.path.basename(study_path)}")
            total_errors += 1; continue

        ct_count = len([f for f in os.listdir(ct_series) if f.lower().endswith('.dcm')])
        print(f"  {pat_key}: {os.path.basename(ct_series)} ({ct_count} DCMs)", flush=True)
        log_lines.append(f"\n[PAT] {pat_key}  CT={os.path.basename(ct_series)} ({ct_count} DCMs)")

        seg_centroids = extract_seg_centroids(study_path)
        log_lines.append(f"  SEGs: {sorted(seg_centroids.keys())}")

        z_list = build_z_index(ct_series)
        if not z_list:
            log_lines.append(f"  [ERROR] empty z-index"); total_errors += 1; continue
        log_lines.append(f"  z-index: {len(z_list)} slices  [{z_list[0][0]:.1f}, {z_list[-1][0]:.1f}]")

        for nod in nodules:
            nid   = nod['nodule_id']
            z_csv = nod['z_csv']
            x_csv = nod['x_csv']
            y_csv = nod['y_csv']

            seg = seg_centroids.get(nid)
            if seg:
                z_center = seg['z']; cx_seg = seg['cx']; cy_seg = seg['cy']; z_source = 'SEG'
            elif z_csv is not None:
                z_center = z_csv; cx_seg = None; cy_seg = None; z_source = 'CSV'
            else:
                log_lines.append(f"  [SKIP] n{nid}: no z"); continue

            res = get_slice_at_z(z_list, z_center)
            if res is None:
                log_lines.append(f"  [SKIP] n{nid}: z-list empty"); continue
            center_idx, center_z_actual, center_fpath = res
            # Guard: if the closest slice is >30mm away, this nodule isn't in this CT series
            if abs(center_z_actual - z_center) > 30.0:
                log_lines.append(f"  [SKIP] n{nid}: z_target={z_center:.1f} closest={center_z_actual:.1f} gap={abs(center_z_actual-z_center):.1f}mm — not in this CT series")
                continue
            log_lines.append(f"  n{nid}  z={z_center:.1f}({z_source})→idx={center_idx} z_actual={center_z_actual:.1f}")

            try:
                hu_center, _, _ = read_pixels_hu(center_fpath)
            except Exception as e:
                log_lines.append(f"  [ERROR] n{nid}: {e}"); total_errors += 1; continue

            if seg:
                chu_d = center_hu_val(hu_center, int(round(cx_seg)), int(round(cy_seg)), 12)
                if chu_d > -820:
                    final_cx, final_cy = cx_seg, cy_seg; xy_source = 'SEG_direct'
                else:
                    bx, by, vn = best_coord_variant(hu_center, cx_seg, cy_seg)
                    if center_hu_val(hu_center, bx, by, 12) > -820:
                        final_cx, final_cy = bx, by; xy_source = f'SEG_var_{vn}'
                    elif x_csv is not None:
                        bx, by, vn = best_coord_variant(hu_center, x_csv, y_csv)
                        final_cx, final_cy = bx, by; xy_source = f'CSV_var_{vn}'
                    else:
                        final_cx, final_cy = cx_seg, cy_seg; xy_source = 'SEG_fallback'
            elif x_csv is not None:
                bx, by, vn = best_coord_variant(hu_center, x_csv, y_csv)
                final_cx, final_cy = bx, by; xy_source = f'CSV_var_{vn}'
            else:
                log_lines.append(f"  [SKIP] n{nid}: no xy"); continue

            log_lines.append(f"    xy={xy_source}  cx={final_cx:.1f} cy={final_cy:.1f}")

            kept_slices = []
            for offset in range(-2, 3):
                si = center_idx + offset
                if si < 0 or si >= len(z_list):
                    continue
                z_mm, slice_path = z_list[si]
                try:
                    hu_arr, _, _ = read_pixels_hu(slice_path)
                except Exception as e:
                    log_lines.append(f"    off={offset:+d}: read error {e}"); continue

                size = 150; half = size // 2
                cx_i = int(round(final_cx)); cy_i = int(round(final_cy))
                x0 = max(0, min(cx_i - half, hu_arr.shape[1] - size))
                y0 = max(0, min(cy_i - half, hu_arr.shape[0] - size))
                crop_hu = hu_arr[y0:y0+size, x0:x0+size]

                passed, metrics = qc_slice(crop_hu, 75, 75)
                status = "PASS" if passed else "DROP"
                log_lines.append(
                    f"    off={offset:+d} z={z_mm:.1f} {status} "
                    f"c={metrics['contrast']} chu={metrics['center_hu']} off={metrics['offset_px']}px"
                )

                if passed:
                    out_name = f"nodule_{nid}_z{offset:+d}.dcm"
                    out_path = os.path.join(OUT_ROOT, pat_key, out_name)
                    ok, msg  = crop_save_raw(slice_path, out_path, final_cx, final_cy)
                    if ok:
                        kept_slices.append(offset); total_kept += 1
                    else:
                        log_lines.append(f"      SAVE ERR: {msg}"); total_errors += 1
                else:
                    total_dropped += 1

            log_lines.append(f"    → kept {len(kept_slices)}/{min(5, len(z_list)-(max(0,center_idx-2)-center_idx+2))}: {kept_slices}")

    log_lines.append(f"\n{'='*60}")
    log_lines.append(f"KEPT={total_kept}  DROPPED={total_dropped}  ERRORS={total_errors}")
    log_text = "\n".join(log_lines)
    log_path = os.path.join(OUT_ROOT, "pipeline_log.txt")
    with open(log_path, 'w') as f:
        f.write(log_text)
    print(log_text)
    return total_kept, total_dropped, total_errors

if __name__ == '__main__':
    kept, dropped, errors = run_pipeline()
    print(f"\nDone. Kept={kept}  Dropped={dropped}  Errors={errors}")
