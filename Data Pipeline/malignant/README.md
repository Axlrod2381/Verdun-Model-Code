# LIDC-IDRI Malignant Nodule Pipeline — Code

## Files

| File | Purpose |
|---|---|
| `dicom_utils.py` | Raw DICOM tag reading + pixel I/O (no pydicom). `read_raw_px`, `read_pixels_hu`, `crop_save_raw` |
| `seg_extraction.py` | SEG mask centroid extraction + CT z-index builder |
| `crop_pipeline.py` | Full centering + cropping pipeline with 5-tier priority selection |
| `evaluate.py` | CNN learnability evaluation — contrast, centering offset, CNN score |
| `the_best_selection.py` | Data optimisation for low-probability images → builds `the_BEST/` |

## Run Order

```bash
# 1. Run the crop pipeline to build clean malignant/
python crop_pipeline.py

# 2. Evaluate all images
python evaluate.py --folder /path/to/clean_malignant --out /tmp/eval_v2.json

# 3. Select best training images
python the_best_selection.py --eval /tmp/eval_v2.json \
    --src /path/to/clean_malignant \
    --dst /path/to/the_BEST
```

## Key Paths (update before running)

All scripts have a `# ── Configuration` or `# ── Defaults` section at the top.
Update these before running:

- `DATASET`  — path to LIDC-IDRI/ folder
- `SPLICED`  — path to `final spliced/`
- `CSV_PATH` — path to `malignant_nodule_locations.csv`
- `OUT_BASE` — path to `clean malignant/`

## Critical Rules

1. **Never store HU-converted pixels** — always raw int16 (`crop_save_raw`)
2. **Always use `rfind()` for pixel data tag** — not `find()`
3. **Test 7 coordinate variants** — LIDC XML coordinate system is undocumented
4. **Use `tissue_score_v2` for z-scans** — prevents rib/bone selection
5. **Lung background as contrast reference** — not surrounding annulus
6. **Patients < 300** — no raw CT available, copy from `final spliced/`
