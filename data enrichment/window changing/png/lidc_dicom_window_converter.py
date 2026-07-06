"""
lidc_dicom_window_converter.py

Converts LIDC-IDRI thoracic CT DICOM series into 2D axial PNG/JPEG images
rendered under standard CT windows (lung window or soft-tissue/mediastinal
window).

Part of: Verdun-Model-Code / data enrichment / window changing / png

Methodology reference: see METHODOLOGY.md in this same folder.

Dependencies:
    pip install pydicom numpy pillow

Usage (CLI):
    python lidc_dicom_window_converter.py \
        --input /path/to/dicom_series_or_zip \
        --output /path/to/converted_output \
        --window lung \
        --format png

    python lidc_dicom_window_converter.py \
        --input /path/to/dicom_series_or_zip \
        --output /path/to/converted_output \
        --window both \
        --format both \
        --size 512x512

Usage (as a library):
    from lidc_dicom_window_converter import convert_series

    convert_series(
        input_path="path/to/series_or_zip",
        output_root="converted_output",
        windows=("lung", "soft"),
        formats=("png",),
        target_size=None,   # or (512, 512)
    )
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
from PIL import Image

# ---------------------------------------------------------------------------
# Standard CT window presets (Hounsfield Units)
# ---------------------------------------------------------------------------

WINDOW_PRESETS: Dict[str, Tuple[float, float]] = {
    # name: (level, width)
    "lung": (-600.0, 1500.0),
    "soft": (50.0, 350.0),
}

VALID_FORMATS = {"png", "jpg", "jpeg"}


@dataclass
class SliceRecord:
    """Holds a single DICOM slice plus the sort key used to order it."""
    dataset: "pydicom.dataset.FileDataset"
    sort_key: float
    source_path: str


# ---------------------------------------------------------------------------
# Step 1: Ingestion
# ---------------------------------------------------------------------------

def discover_dicom_files(input_path: str) -> List[str]:
    """
    Recursively discover DICOM files from a directory, or extract and
    discover them from a ZIP archive. Ignores non-DICOM artifacts.
    """
    if os.path.isfile(input_path) and input_path.lower().endswith(".zip"):
        extract_dir = tempfile.mkdtemp(prefix="lidc_zip_")
        with zipfile.ZipFile(input_path, "r") as zf:
            zf.extractall(extract_dir)
        input_path = extract_dir

    dicom_paths: List[str] = []
    ignore_names = {".DS_Store", "DICOMDIR", "LICENSE", "LICENSE.txt"}

    for root, _dirs, files in os.walk(input_path):
        for fname in files:
            if fname in ignore_names or fname.startswith("."):
                continue
            full_path = os.path.join(root, fname)
            if _looks_like_dicom(full_path):
                dicom_paths.append(full_path)

    return dicom_paths


def _looks_like_dicom(path: str) -> bool:
    """Fast check: try reading DICOM metadata without pixel data."""
    try:
        pydicom.dcmread(path, stop_before_pixels=True, force=True)
        return True
    except (InvalidDicomError, Exception):
        return False


def group_by_series(dicom_paths: Sequence[str]) -> Dict[str, List[str]]:
    """Group discovered DICOM file paths by SeriesInstanceUID."""
    series_map: Dict[str, List[str]] = defaultdict(list)
    for path in dicom_paths:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception:
            continue
        series_uid = getattr(ds, "SeriesInstanceUID", "UNKNOWN_SERIES")
        series_map[series_uid].append(path)
    return series_map


# ---------------------------------------------------------------------------
# Step 2: Series validation
# ---------------------------------------------------------------------------

def validate_series(paths: Sequence[str]) -> Tuple[bool, List[str]]:
    """
    Validate that a set of DICOM paths represents a readable CT series with
    pixel data. Returns (is_valid, warnings).
    """
    warnings: List[str] = []
    valid_count = 0

    for path in paths:
        try:
            ds = pydicom.dcmread(path, force=True)
        except Exception:
            continue

        modality = getattr(ds, "Modality", None)
        if modality is not None and modality != "CT":
            warnings.append(f"{path}: Modality is '{modality}', expected 'CT'.")

        has_pixels = hasattr(ds, "PixelData")
        has_ordering = any(
            hasattr(ds, attr)
            for attr in ("ImagePositionPatient", "SliceLocation", "InstanceNumber")
        )

        if has_pixels and has_ordering:
            valid_count += 1

        if not hasattr(ds, "RescaleSlope") or not hasattr(ds, "RescaleIntercept"):
            warnings.append(
                f"{path}: missing RescaleSlope/RescaleIntercept; "
                "HU conversion will default to slope=1, intercept=0 (approximate)."
            )

    is_valid = valid_count > 0
    if not is_valid:
        warnings.append("No slices with both pixel data and ordering metadata were found.")

    return is_valid, warnings


# ---------------------------------------------------------------------------
# Step 3: Slice ordering
# ---------------------------------------------------------------------------

def load_and_sort_series(paths: Sequence[str]) -> List[SliceRecord]:
    """
    Load all DICOM datasets for a series and sort them using priority:
    ImagePositionPatient[2] > SliceLocation > InstanceNumber.
    """
    records: List[SliceRecord] = []

    for path in paths:
        try:
            ds = pydicom.dcmread(path, force=True)
        except Exception:
            continue
        if not hasattr(ds, "PixelData"):
            continue

        sort_key = _extract_sort_key(ds)
        records.append(SliceRecord(dataset=ds, sort_key=sort_key, source_path=path))

    records.sort(key=lambda r: r.sort_key)
    return records


def _extract_sort_key(ds) -> float:
    if hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) == 3:
        return float(ds.ImagePositionPatient[2])
    if hasattr(ds, "SliceLocation"):
        return float(ds.SliceLocation)
    if hasattr(ds, "InstanceNumber"):
        return float(ds.InstanceNumber)
    return 0.0


# ---------------------------------------------------------------------------
# Step 4: Hounsfield Unit conversion
# ---------------------------------------------------------------------------

def to_hounsfield_units(ds) -> np.ndarray:
    """Convert a DICOM dataset's raw pixel array to Hounsfield Units."""
    pixel_array = ds.pixel_array.astype(np.float64)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = pixel_array * slope + intercept
    return hu


# ---------------------------------------------------------------------------
# Step 5: CT windowing
# ---------------------------------------------------------------------------

def apply_window(hu_array: np.ndarray, level: float, width: float) -> np.ndarray:
    """
    Apply a CT window (level, width) to an HU array and normalize to 8-bit.
    """
    lower = level - width / 2.0
    upper = level + width / 2.0

    clipped = np.clip(hu_array, lower, upper)
    normalized = (clipped - lower) / (upper - lower) * 255.0
    return normalized.astype(np.uint8)


# ---------------------------------------------------------------------------
# Step 6: Optional resizing
# ---------------------------------------------------------------------------

def resize_image(img: Image.Image, target_size: Optional[Tuple[int, int]]) -> Image.Image:
    if target_size is None:
        return img
    return img.resize(target_size, Image.LANCZOS)


# ---------------------------------------------------------------------------
# Steps 7-8: Export, organize, and package
# ---------------------------------------------------------------------------

def build_output_dirs(output_root: str, windows: Iterable[str], formats: Iterable[str]) -> Dict[Tuple[str, str], str]:
    """
    Create the axial_<window>_<format> folder structure and return a map
    from (window, format) -> directory path.
    """
    dir_map: Dict[Tuple[str, str], str] = {}
    for window in windows:
        for fmt in formats:
            fmt_norm = "jpeg" if fmt in ("jpg", "jpeg") else fmt
            dir_name = f"axial_{window}_{fmt_norm}"
            dir_path = os.path.join(output_root, dir_name)
            os.makedirs(dir_path, exist_ok=True)
            dir_map[(window, fmt)] = dir_path
    return dir_map


def safe_prefix(ds) -> str:
    """Derive a filesystem-safe filename prefix from patient/series ID."""
    patient_id = getattr(ds, "PatientID", None)
    if patient_id:
        return str(patient_id).strip().replace(" ", "_")
    series_uid = getattr(ds, "SeriesInstanceUID", None)
    if series_uid:
        return f"series_{str(series_uid)[-8:]}"
    return "scan"


def convert_series(
    input_path: str,
    output_root: str,
    windows: Sequence[str] = ("lung",),
    formats: Sequence[str] = ("png",),
    target_size: Optional[Tuple[int, int]] = None,
    custom_window_values: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, int]:
    """
    Full pipeline entry point: discover, validate, sort, convert, window,
    export, and organize LIDC-IDRI DICOM slices into PNG/JPEG images.

    Returns a summary dict of {window_format_key: slice_count}.
    """
    for w in windows:
        if w not in WINDOW_PRESETS and (not custom_window_values or w not in custom_window_values):
            raise ValueError(f"Unknown window '{w}'. Expected one of {list(WINDOW_PRESETS)} or a custom preset.")
    for f in formats:
        if f not in VALID_FORMATS:
            raise ValueError(f"Unknown format '{f}'. Expected one of {sorted(VALID_FORMATS)}.")

    window_values = dict(WINDOW_PRESETS)
    if custom_window_values:
        window_values.update(custom_window_values)

    dicom_paths = discover_dicom_files(input_path)
    if not dicom_paths:
        raise RuntimeError(f"No DICOM files found under: {input_path}")

    series_map = group_by_series(dicom_paths)

    dir_map = build_output_dirs(output_root, windows, formats)
    summary: Dict[str, int] = defaultdict(int)

    for series_uid, paths in series_map.items():
        is_valid, warnings = validate_series(paths)
        for w in warnings:
            print(f"[warn][series {series_uid}] {w}", file=sys.stderr)
        if not is_valid:
            print(f"[skip] Series {series_uid} has no usable CT slices.", file=sys.stderr)
            continue

        records = load_and_sort_series(paths)
        if not records:
            continue

        prefix = safe_prefix(records[0].dataset)

        for idx, record in enumerate(records):
            try:
                hu = to_hounsfield_units(record.dataset)
            except Exception as exc:
                print(f"[skip] Failed to read pixel data for {record.source_path}: {exc}", file=sys.stderr)
                continue

            for window in windows:
                level, width = window_values[window]
                windowed_8bit = apply_window(hu, level, width)
                img = Image.fromarray(windowed_8bit, mode="L")
                img = resize_image(img, target_size)

                for fmt in formats:
                    ext = "jpg" if fmt in ("jpg", "jpeg") else "png"
                    out_dir = dir_map[(window, fmt)]
                    filename = f"{prefix}_slice{idx:04d}.{ext}"
                    out_path = os.path.join(out_dir, filename)

                    if ext == "jpg":
                        img.save(out_path, format="JPEG", quality=95)
                    else:
                        img.save(out_path, format="PNG")

                    summary[f"{window}_{ext}"] += 1

    return dict(summary)


def package_output(output_root: str, zip_path: Optional[str] = None) -> str:
    """Compress the output folder into a ZIP for download."""
    if zip_path is None:
        zip_path = output_root.rstrip("/\\") + ".zip"
    base_name = zip_path[:-4] if zip_path.endswith(".zip") else zip_path
    shutil.make_archive(base_name, "zip", output_root)
    return base_name + ".zip"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_size(size_str: Optional[str]) -> Optional[Tuple[int, int]]:
    if not size_str or size_str.lower() == "original":
        return None
    try:
        w, h = size_str.lower().split("x")
        return (int(w), int(h))
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid size '{size_str}'. Expected format WIDTHxHEIGHT, e.g. 512x512."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LIDC-IDRI DICOM CT series into windowed axial PNG/JPEG images."
    )
    parser.add_argument("--input", required=True, help="Path to a DICOM directory or ZIP archive.")
    parser.add_argument("--output", required=True, help="Output root directory for converted images.")
    parser.add_argument(
        "--window",
        default="lung",
        choices=["lung", "soft", "both"],
        help="CT window to apply: 'lung', 'soft' (soft-tissue/mediastinal), or 'both'.",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "jpg", "jpeg", "both"],
        help="Output image format: 'png', 'jpg'/'jpeg', or 'both'.",
    )
    parser.add_argument(
        "--size",
        default=None,
        help="Target size WIDTHxHEIGHT (e.g. 512x512), or omit to keep native resolution.",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Package the output folder into a ZIP file after conversion.",
    )

    args = parser.parse_args()

    windows = ["lung", "soft"] if args.window == "both" else [args.window]
    formats = ["png", "jpg"] if args.format == "both" else [args.format]
    target_size = _parse_size(args.size)

    summary = convert_series(
        input_path=args.input,
        output_root=args.output,
        windows=windows,
        formats=formats,
        target_size=target_size,
    )

    total = sum(summary.values())
    print(f"Conversion complete. {total} images written across {len(summary)} window/format combinations:")
    for key, count in summary.items():
        print(f"  {key}: {count} images")

    if args.zip:
        zip_path = package_output(args.output)
        print(f"Packaged output as: {zip_path}")


if __name__ == "__main__":
    main()
