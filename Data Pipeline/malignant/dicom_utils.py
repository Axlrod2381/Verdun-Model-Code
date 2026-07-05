"""
dicom_utils.py
--------------
Raw DICOM tag reading and pixel I/O — no pydicom required.
Handles both Explicit VR and Implicit VR DICOM files found in LIDC-IDRI.

Usage:
    from dicom_utils import read_ds_tag, read_uint16_tag, read_raw_px, read_pixels_hu
"""

import struct
import os
import numpy as np


# ── Tag byte constants ────────────────────────────────────────────────────────
TAG_ROWS      = b'\x28\x00\x10\x00'   # (0028,0010)  uint16
TAG_COLS      = b'\x28\x00\x11\x00'   # (0028,0011)  uint16
TAG_INTERCEPT = b'\x28\x00\x52\x10'   # (0028,1052)  DS string  default -1024
TAG_SLOPE     = b'\x28\x00\x53\x10'   # (0028,1053)  DS string  default 1.0
TAG_IMG_POS   = b'\x20\x00\x32\x00'   # (0020,0032)  DS string  "x\y\z"
TAG_PIXEL     = b'\xe0\x7f\x10\x00'   # (7FE0,0010)  binary — always use rfind()

# VRs that use 4-byte extended length (explicit VR)
EXTENDED_VR = {b'OB', b'OW', b'OF', b'SQ', b'UC', b'UN', b'UR', b'UT'}


# ── Tag readers ───────────────────────────────────────────────────────────────

def read_ds_tag(raw, tag4):
    """
    Read a Decimal String (DS) DICOM tag value.
    Works for both Explicit VR and Implicit VR files.

    Args:
        raw:  bytes or bytearray of the DICOM file
        tag4: 4-byte tag e.g. TAG_INTERCEPT

    Returns:
        str value stripped of null/space padding, or None if not found
    """
    raw_b = bytes(raw)
    idx = raw_b.find(tag4)
    while idx >= 0:
        vr = raw_b[idx+4:idx+6]
        if len(vr) == 2 and vr[0:1].isalpha() and vr[1:2].isalpha():
            # Explicit VR: tag(4) + VR(2) + length(2) + value
            length = struct.unpack('<H', raw_b[idx+6:idx+8])[0]
            if 0 < length < 200:
                return raw_b[idx+8:idx+8+length].decode('latin-1', 'replace').strip('\x00 ')
        else:
            # Implicit VR: tag(4) + length(4) + value
            length = struct.unpack('<I', raw_b[idx+4:idx+8])[0]
            if 0 < length < 200:
                return raw_b[idx+8:idx+8+length].decode('latin-1', 'replace').strip('\x00 ')
        idx = raw_b.find(tag4, idx+1)
    return None


def read_uint16_tag(raw, tag4):
    """
    Read a US (unsigned short / uint16) DICOM tag value.
    Works for both Explicit VR and Implicit VR files.

    Returns:
        int value, or None if not found
    """
    raw_b = bytes(raw)
    idx = raw_b.find(tag4)
    while idx >= 0:
        vr = raw_b[idx+4:idx+6]
        if vr[0:1].isalpha() and vr[1:2].isalpha():
            # Explicit VR
            length = struct.unpack('<H', raw_b[idx+6:idx+8])[0]
            if length == 2:
                return struct.unpack('<H', raw_b[idx+8:idx+10])[0]
        else:
            # Implicit VR
            length = struct.unpack('<I', raw_b[idx+4:idx+8])[0]
            if length == 2:
                return struct.unpack('<H', raw_b[idx+8:idx+10])[0]
        idx = raw_b.find(tag4, idx+1)
    return None


# ── Pixel I/O ─────────────────────────────────────────────────────────────────

def _pixel_data_offset(raw_b):
    """Return (data_start_offset) for the pixel data in raw bytes."""
    pix_idx = raw_b.rfind(TAG_PIXEL)   # rfind: pixel data is always last
    if pix_idx < 0:
        raise ValueError("No pixel data tag (7FE0,0010) found in file")
    vr = raw_b[pix_idx+4:pix_idx+6]
    if bytes(vr) in EXTENDED_VR:
        return pix_idx + 12             # tag(4) + VR(2) + reserved(2) + length(4)
    return pix_idx + 8                  # tag(4) + length(4)


def read_raw_px(filepath):
    """
    Read raw int16 pixel array from a DICOM CT slice.
    Does NOT apply RescaleIntercept or RescaleSlope.

    Returns:
        (array np.int16 shape(rows,cols), rows, cols)

    CRITICAL: Store crops using these raw values — never convert to HU
    before storing, or you get the double-intercept bug on readback.
    """
    with open(filepath, 'rb') as f:
        raw = f.read()
    rows = read_uint16_tag(raw, TAG_ROWS) or 512
    cols = read_uint16_tag(raw, TAG_COLS) or 512
    data_start = _pixel_data_offset(bytes(raw))
    n_bytes = rows * cols * 2
    px = np.frombuffer(raw[data_start:data_start + n_bytes], dtype=np.int16)
    return px.reshape(rows, cols), rows, cols


def read_pixels_hu(filepath):
    """
    Read HU float32 array from a DICOM CT slice.
    Applies RescaleIntercept and RescaleSlope from the header.

    Returns:
        (array np.float32 shape(rows,cols), rows, cols)

    Use for ANALYSIS only — never write these values back as pixel data.
    """
    arr, rows, cols = read_raw_px(filepath)
    with open(filepath, 'rb') as f:
        raw = f.read()
    intercept_s = read_ds_tag(raw, TAG_INTERCEPT)
    slope_s     = read_ds_tag(raw, TAG_SLOPE)
    intercept   = float(intercept_s) if intercept_s else -1024.0
    slope       = float(slope_s)     if slope_s     else 1.0
    return arr.astype(np.float32) * slope + intercept, rows, cols


def read_z(filepath):
    """
    Return the ImagePositionPatient z coordinate (mm) from a DICOM slice.
    Tag (0020,0032) stores "x\\y\\z" as a backslash-separated DS string.
    """
    with open(filepath, 'rb') as f:
        raw = f.read(8192)   # z is always in the first 8KB of header
    pos_s = read_ds_tag(raw, TAG_IMG_POS)
    if pos_s:
        parts = pos_s.split('\\')
        if len(parts) >= 3:
            return float(parts[2])
    return None


# ── Crop and save ─────────────────────────────────────────────────────────────

def crop_save_raw(ct_path, out_path, cx, cy, size=150):
    """
    Crop a size×size patch centred on (cx, cy) from a CT DICOM slice.
    Saves with the ORIGINAL header intact — raw int16 pixels, no HU conversion.

    Why raw int16: if you store HU values and the RescaleIntercept zero
    fails (implicit VR), on readback HU = stored_HU + intercept ≈ -1081
    for a solid nodule. Storing raw avoids this entirely.

    Args:
        ct_path:  source CT DICOM filepath
        out_path: destination filepath
        cx, cy:   crop centre in pixel coordinates
        size:     output image size in pixels (default 150)

    Returns:
        (True, "ok") on success, (False, error_message) on failure
    """
    try:
        with open(ct_path, 'rb') as f:
            raw = bytearray(f.read())

        raw_arr, rows, cols = read_raw_px(ct_path)
        cx, cy = int(round(cx)), int(round(cy))
        half = size // 2
        x0 = max(0, min(cx - half, cols - size))
        y0 = max(0, min(cy - half, rows - size))
        crop = raw_arr[y0:y0+size, x0:x0+size]   # raw int16

        # Update Rows and Cols tags in header
        for tag in [TAG_ROWS, TAG_COLS]:
            idx = bytes(raw).find(tag)
            if idx >= 0:
                # Works for both explicit and implicit VR (value is always at +8)
                raw[idx+8:idx+10] = struct.pack('<H', size)

        # Replace pixel data
        raw_b     = bytes(raw)
        pix_idx   = raw_b.rfind(TAG_PIXEL)
        vr        = raw_b[pix_idx+4:pix_idx+6]
        hdr_end   = pix_idx + 12 if bytes(vr) in EXTENDED_VR else pix_idx + 8
        new_pix   = crop.tobytes()
        raw[hdr_end-4:hdr_end] = struct.pack('<I', len(new_pix))
        out_bytes = bytes(raw[:hdr_end]) + new_pix

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'wb') as f:
            f.write(out_bytes)
        return True, "ok"

    except Exception as e:
        return False, str(e)
