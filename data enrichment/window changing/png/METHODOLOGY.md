# LIDC-IDRI DICOM Window Conversion — Methodology

## Overview

This document describes the methodology used to convert raw LIDC-IDRI thoracic
CT DICOM series into 2D axial PNG/JPEG images rendered under standard
radiological CT windows (lung window or soft-tissue/mediastinal window). This
conversion step is part of the data enrichment pipeline that prepares LIDC-IDRI
CT scans for downstream CNN-based nodule classification (benign vs. malignant)
in the Verdun lung cancer diagnostic model.

The goal is to transform 3D DICOM volumes with arbitrary raw pixel intensities
into consistently windowed, normalized, 8-bit 2D slice images that a CNN can
consume, while preserving diagnostic-relevant contrast in the tissue of
interest (lung parenchyma or mediastinal/soft tissue).

## Pipeline Stages

### 1. Ingestion
- Accept a directory, ZIP archive, or set of individual `.dcm` files
  representing one or more DICOM series.
- Recursively discover DICOM files, ignoring non-DICOM artifacts (e.g.
  `.DS_Store`, `DICOMDIR`, license files).
- Group files by `SeriesInstanceUID` so that each series is processed as a
  self-contained 3D volume.

### 2. Series Validation
Before processing a series, verify:
- `Modality == "CT"` (when present in metadata).
- Pixel data is present and readable (`PixelData` tag populated).
- At least one slice-ordering field is available: `ImagePositionPatient`,
  `SliceLocation`, or `InstanceNumber`.
- `RescaleSlope` / `RescaleIntercept` are present for Hounsfield Unit (HU)
  conversion; if missing, default to slope = 1, intercept = 0 and flag the
  series as using an approximate HU conversion.

If a study contains multiple reconstructions (e.g. both a lung kernel and a
soft-tissue kernel), the reconstruction is matched to the requested output
window: lung-kernel series feed the lung-window output, soft-tissue-kernel
series feed the soft-tissue-window output. If only one reconstruction exists,
it is used for both windows via digital windowing of the HU values.

### 3. Slice Ordering
Slices within a series are sorted using this priority:
1. `ImagePositionPatient[2]` (z-coordinate) — most reliable, immune to
   inconsistent acquisition/instance numbering.
2. `SliceLocation` — used if `ImagePositionPatient` is unavailable.
3. `InstanceNumber` — used only if neither spatial field exists.

Filenames are never used to infer order unless no DICOM-native ordering
metadata is present, since filenames are not guaranteed to reflect anatomical
position.

### 4. Hounsfield Unit (HU) Conversion
Raw pixel data stored in a DICOM file represents scanner-specific arbitrary
units, not physically meaningful density values. Each slice's raw pixel array
is rescaled into Hounsfield Units using the linear transformation encoded in
the DICOM header:

\[ HU = \text{pixel\_array} \times \text{RescaleSlope} + \text{RescaleIntercept} \]

This step is required because CT window levels/widths (e.g. "lung window
level -600") are defined in HU space, not raw pixel space. Skipping this step
would make windowing values scanner- and protocol-dependent rather than
standardized.

### 5. CT Windowing
Radiological "windowing" selects and stretches a specific HU range to maximize
visual contrast in the tissue of interest, since the human eye (and,
practically, 8-bit image encodings) cannot represent the full dynamic range of
CT HU values (~-1000 to +3000) at once.

Given a window level \(L\) and window width \(W\):

\[ \text{lower} = L - \frac{W}{2}, \qquad \text{upper} = L + \frac{W}{2} \]

All HU values are clipped to \([\text{lower}, \text{upper}]\), then linearly
normalized to the 0–255 range for 8-bit image export:

\[ \text{pixel}_{8bit} = \frac{\text{clip}(HU, \text{lower}, \text{upper}) - \text{lower}}{W} \times 255 \]

Two standard windows are used, matching conventional chest CT reading
protocols:

| Window | Level (HU) | Width (HU) | Purpose |
|---|---|---|---|
| Lung window | -600 | 1500 | Maximizes contrast in aerated lung parenchyma; best for visualizing nodule margins, ground-glass opacity, and airway structures. |
| Soft-tissue / mediastinal window | 50 | 350 | Maximizes contrast in mediastinal and chest-wall soft tissue; best for visualizing vasculature, lymph nodes, and nodule solid components against soft tissue. |

These defaults follow standard clinical windowing conventions and are
user-overridable when the pipeline is invoked with custom level/width values.

### 6. Optional Resizing
If a target resolution is requested (e.g. 256×256 or 512×512 for a specific
CNN input size), the windowed 8-bit image is resized using high-quality
(Lanczos) interpolation, preserving aspect ratio unless square output is
explicitly requested. When no resizing is requested, the native in-plane
resolution of the scanner acquisition is preserved, since downstream
augmentation/resizing may be handled separately in the model's data loader.

### 7. Export and Organization
Each processed axial slice is saved as an independent PNG or JPEG file with a
deterministic, filesystem-safe filename encoding the patient/series ID and
zero-padded slice index, e.g.:

```
LIDC-IDRI-0001_slice0000.png
LIDC-IDRI-0001_slice0001.png
```

Outputs are grouped into window/format-specific subfolders:

```
converted_output/
├── axial_lung_png/
├── axial_lung_jpeg/
├── axial_soft_png/
└── axial_soft_jpeg/
```

This structure keeps lung-window and soft-tissue-window renderings of the
same underlying series clearly separated, which matters because they are
typically used for different downstream purposes (parenchymal nodule
detection vs. soft-tissue/lymph node characterization) and should not be
mixed in a single training set unless explicitly intended.

### 8. Quality and Error Handling
The pipeline degrades gracefully rather than failing outright:
- Corrupt or unreadable individual slices are skipped and logged; processing
  continues if enough valid slices remain in the series.
- Series with missing rescale metadata are still processed, with an explicit
  warning that HU values (and therefore windowing) are approximate.
- Non-CT or non-DICOM inputs are rejected with a clear message rather than
  silently producing garbage output.
- Studies with multiple series are split by `SeriesInstanceUID` so unrelated
  series (e.g. scout images, localizers) do not contaminate the axial output.

## Rationale for This Approach

1. **Clinical fidelity**: Using standard lung (-600/1500) and mediastinal
   (50/350) window settings ensures the visual appearance of exported images
   matches what a radiologist would see when reading the same series in a
   PACS viewer, which is important both for human QA of the dataset and for
   preserving the contrast cues that a CNN needs to distinguish nodule
   characteristics.
2. **Reproducibility**: Deriving windowing from HU (not raw pixel values)
   makes outputs comparable across scans acquired on different scanners or
   protocols, since HU is a standardized physical unit while raw pixel values
   are not.
3. **Traceability**: Sorting by spatial position (rather than filename or
   instance number alone) and preserving patient/series identifiers in
   filenames keeps every exported PNG/JPEG traceable back to its exact
   position in the original 3D volume, which is required to later align
   slices with LIDC-IDRI nodule annotation coordinates (e.g. from the XML
   annotation files) for supervised labeling.
4. **Separation of concerns**: This stage only performs windowing and image
   export — it does not perform nodule cropping, augmentation, or
   normalization for model input. Keeping this stage narrowly scoped makes it
   reusable across multiple downstream pipelines (e.g. full-slice
   classification vs. nodule-crop classification) without duplicating the
   DICOM-handling logic.

## Libraries Used
- `pydicom` — DICOM parsing and metadata/pixel access.
- `numpy` — numerical array operations (HU conversion, clipping, normalization).
- `Pillow` (PIL) — 8-bit image encoding, resizing, and PNG/JPEG export.

## Position in the Data Enrichment Pipeline

```
Raw LIDC-IDRI DICOM series
        │
        ▼
[ Data Enrichment / Window Changing / PNG ]   <-- this methodology
        │
        ▼
Windowed axial PNG/JPEG slices (lung + soft-tissue)
        │
        ▼
Nodule cropping / annotation alignment (downstream stage)
        │
        ▼
CNN training set (normal / benign / malignant)
```
