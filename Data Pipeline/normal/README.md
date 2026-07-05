# Data Pipeline — `normal` (no_nodule negative class)

Pipeline for building the **no_nodule (negative)** class of the lung-nodule CNN: 150×150
nodule-free lung-parenchyma crops, produced with the same spatial/intensity conventions as the
malignant/benign positive classes.

## Files
- `normal_healthy_pipeline.py` — shared core algorithm + **LIDC-IDRI confirmed-benign** driver.
  Contains the DICOM I/O (no pydicom, raw int16, no double-intercept), lung segmentation,
  center-on-parenchyma gate, inverted apparent-nodule check, farthest-point spatial-variability
  sampling, and raw-int16 crop saving.
- `unitochest_pipeline.py` — **UNITOChest** driver. Imports the shared functions from
  `normal_healthy_pipeline.py`, streams slices directly from the UNITOChest zip, and uses the
  dataset's segmentation masks as the nodule exclusion source.

> `unitochest_pipeline.py` imports from `normal_healthy_pipeline.py`, so keep both files together.

## Requirements
Python 3, NumPy, OpenCV (`opencv-python`). No pydicom.

## What it produces
| Source | Crops | Notes |
|---|---|---|
| LIDC-IDRI confirmed-benign | 1,410 | 38 patients; nodule locations used as exclusion mask |
| UNITOChest | 3,824 | 480 patients × 8; CSV `mask` column flags nodule-free slices |
| **Total** | **5,234** | 150×150 raw int16 DICOM |

Paths (input datasets, output folders) are set as constants at the top of each script — edit them
for your environment before running.

## Note on the negative-class domain shift
Negatives are ~27% LIDC / ~73% UNITOChest while the positives are 100% LIDC. Before training,
apply a single global intensity mapping across all classes and run a source-leakage audit
(LIDC-vs-UNITOChest classifier on the negatives); harmonize or add UNITOChest positives if the
scanner signature separates too easily.
