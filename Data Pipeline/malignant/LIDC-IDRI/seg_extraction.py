"""
seg_extraction.py
-----------------
Extract nodule centroid coordinates from LIDC-IDRI SEG DICOM files.

SEG files are binary masks drawn by radiologists on CT slices.
They are stored in subfolders named "Segmentation of Nodule N".
Pixel data can be uint16, uint8, or 1-bit packed — all three are handled.

Usage:
    from seg_extraction import seg_centroids_raw, build_z_index
"""

import os
import re
import struct
import numpy as np
from dicom_utils import read_uint16_tag, read_ds_tag, TAG_ROWS, TAG_COLS, TAG_IMG_POS, TAG_PIXEL, EXTENDED_VR


# ── SEG folder detection ──────────────────────────────────────────────────────

SEG_FOLDER_PATTERN = re.compile(r'[Ss]egmentation of [Nn]odule\s+(\d+)')


def seg_centroids_raw(pat_path):
    """
    Find all SEG subfolders under pat_path and extract centroid per nodule.

    Folder naming: "Segmentation of Nodule N" (case-insensitive).
    Best frame = frame with most foreground pixels (largest cross-section).

    Returns:
        dict {nodule_int: {'z': float, 'cx': float, 'cy': float, 'n_px': int}}
        Empty dict if no SEG folders found.
    """
    result = {}

    for item in os.listdir(pat_path):
        m = SEG_FOLDER_PATTERN.match(item)
        if not m:
            continue
        nodule_n  = int(m.group(1))
        seg_dir   = os.path.join(pat_path, item)
        best      = None

        for fname in sorted(os.listdir(seg_dir)):
            if not fname.endswith('.dcm'):
                continue
            fpath = os.path.join(seg_dir, fname)

            with open(fpath, 'rb') as f:
                raw = f.read()

            rows = read_uint16_tag(raw, TAG_ROWS) or 512
            cols = read_uint16_tag(raw, TAG_COLS) or 512

            # Z position
            pos_s = read_ds_tag(raw, TAG_IMG_POS)
            z = 0.0
            if pos_s and '\\' in pos_s:
                try:
                    z = float(pos_s.split('\\')[2])
                except (IndexError, ValueError):
                    pass

            # Pixel data
            pix_idx = raw.rfind(TAG_PIXEL)
            if pix_idx < 0:
                continue
            vr         = raw[pix_idx+4:pix_idx+6]
            data_start = pix_idx + 12 if bytes(vr) in EXTENDED_VR else pix_idx + 8
            pdata      = raw[data_start:]

            mask = _decode_seg_pixels(pdata, rows, cols)
            if mask is None or mask.sum() == 0:
                continue

            ys, xs = np.where(mask > 0)
            n_px   = int(mask.sum())
            cx     = float(xs.mean())
            cy     = float(ys.mean())

            # Keep the frame with the most foreground pixels
            if best is None or n_px > best['n_px']:
                best = {'z': z, 'cx': cx, 'cy': cy, 'n_px': n_px}

        if best:
            result[nodule_n] = best

    return result


def _decode_seg_pixels(pdata, rows, cols):
    """
    Try uint16 → uint8 → 1-bit unpackbits.
    Returns binary uint8 mask or None if all formats fail.
    """
    # uint16
    n16 = rows * cols * 2
    if len(pdata) >= n16:
        candidate = np.frombuffer(pdata[:n16], dtype=np.uint16).reshape(rows, cols)
        if candidate.max() > 0:
            return (candidate > 0).astype(np.uint8)

    # uint8
    n8 = rows * cols
    if len(pdata) >= n8:
        candidate = np.frombuffer(pdata[:n8], dtype=np.uint8).reshape(rows, cols)
        if candidate.max() > 0:
            return (candidate > 0).astype(np.uint8)

    # 1-bit packed
    n_bits = (rows * cols + 7) // 8
    if len(pdata) >= n_bits:
        bits = np.unpackbits(np.frombuffer(pdata[:n_bits], dtype=np.uint8))
        if len(bits) >= rows * cols and bits[:rows*cols].max() > 0:
            return bits[:rows*cols].reshape(rows, cols)

    return None


# ── CT slice z-index ──────────────────────────────────────────────────────────

def build_z_index(pat_path):
    """
    Find the main CT folder within pat_path (the subfolder with the most DCMs,
    excluding any folder with 'segment' in its path).

    Requires >= 80 DCMs to qualify as a real CT volume (not an annotation series).

    Returns:
        [(z_mm, filepath), ...] sorted ascending by z
        Empty list if no valid CT volume found (common for patients < 0300).
    """
    best_folder, best_count = None, 0

    for root, dirs, files in os.walk(pat_path):
        if 'segment' in root.lower():
            continue
        dcms = [f for f in files if f.lower().endswith('.dcm')]
        if len(dcms) > best_count:
            best_count  = len(dcms)
            best_folder = root

    if not best_folder or best_count < 80:
        return []   # Not a full CT volume

    z_list = []
    for fname in os.listdir(best_folder):
        if not fname.lower().endswith('.dcm'):
            continue
        fpath = os.path.join(best_folder, fname)
        z = _read_z_fast(fpath)
        if z is not None:
            z_list.append((z, fpath))

    return sorted(z_list, key=lambda x: x[0])


def _read_z_fast(filepath):
    """Read only the ImagePositionPatient z from the first 8KB of a DICOM file."""
    try:
        with open(filepath, 'rb') as f:
            raw = f.read(8192)
        pos_s = read_ds_tag(raw, TAG_IMG_POS)
        if pos_s:
            parts = pos_s.split('\\')
            if len(parts) >= 3:
                return float(parts[2])
    except Exception:
        pass
    return None


def get_slice_at_z(z_list, target_z):
    """Return (filepath, delta_z) for the CT slice closest to target_z."""
    if not z_list:
        return None, float('inf')
    best = min(z_list, key=lambda x: abs(x[0] - target_z))
    return best[1], abs(best[0] - target_z)
