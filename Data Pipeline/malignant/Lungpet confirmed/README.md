# Lung-PET-CT-Dx — Malignant Nodule Pipeline

Builds a clean, CNN-ready set of pathology-confirmed **malignant** lung-nodule crops from the
[Lung-PET-CT-Dx](https://doi.org/10.7937/TCIA.2020.NNC20461) CT scans (TCIA), using the official
PASCAL-VOC annotation XMLs. Output crops are **150×150 raw int16 DICOM** with true Hounsfield-unit
(HU) values preserved (no enhancement, no PNGs).

## Files

| File | Purpose |
|------|---------|
| `lungpet_pipeline.py` | Combined pipeline — cropping + apparent-QC **and** best-5000 curation, with a small CLI. |
| `nodule_qc.py` | Raw-DICOM helper it imports (no pydicom): HU reading, lung reference, center refinement, HU-safe raw crop writer. Must sit next to `lungpet_pipeline.py`. |

## What it does

**Stage 1 — crop + QC.** Each annotation XML is named by a CT slice's `SOPInstanceUID`, so it maps
to an exact slice. For every annotated slice the pipeline crops that slice **and its ±2 z-neighbors**
within the same series (both reconstruction kernels are processed), centering on the annotation box.
Each cropped z level is checked by an apparent-nodule gate and only passing levels are written.

**Stage 2 — curate to the best 5,000.** Every kept crop is scored, one reconstruction is chosen per
patient, each patient's single best crop is protected, and the highest-scoring crops are kept up to
5,000 (every patient stays represented).

## QC gate (per z level)

A level is kept only if all hold, else it is dropped:

- **Not air** — center HU (12 px radius mean) > −820
- **Apparent** — contrast = center HU − lung-background HU ≥ 180 HU
- **On-frame** — center-of-mass offset within the central 70×70 < 35 px

Contrast is measured against a pure lung-airspace reference (mean HU of pixels in [−980, −600]).
Nodule type is recorded for reference only — all nodules are pathology-malignant, so type is not a gate.

## "Best" score (used for curation)

- **Centering** = `1 − offset/30` (tighter center-of-mass scores higher)
- **Conspicuity** = `0.6·min(contrast/600, 1) + 0.4·type_weight` (solid 1.0, part-solid 0.9, calcified 0.7, GGO 0.6, sub-GGO 0.4)
- **Final** = `0.5·Centering + 0.5·Conspicuity`

## HU integrity

Crops are written as **raw int16** with the DICOM header intact (read raw → crop raw → write raw).
The RescaleIntercept is never zeroed — doing so fails silently on implicit-VR files and produces a
double-intercept error (a solid nodule reads back as dense air, ~−1081 HU). The pixel-data tag is
found with `rfind()` to avoid matching stray tag bytes earlier in the file.

## Usage

```bash
python lungpet_pipeline.py crop [PID ...]     # crop + QC (resumable; optional patient filter)
python lungpet_pipeline.py select [--apply]   # curate to best 5000 (dry-run without --apply)
python lungpet_pipeline.py all  [--apply]     # crop everything, then curate
```

Requires `numpy` and `nodule_qc.py` alongside the script.

## Results (this run)

- 12,409 QC-passed crops → curated to **5,000** across **222 patients**
- Median centroid offset **6.0 px**; median contrast above lung **885 HU**
- Type mix: solid 3,626 · part-solid 1,328 · GGO 34 · calcified 12

## Notes

- The `ROOT`, `ANN`, and `OUT` paths at the top of `lungpet_pipeline.py` are the absolute paths from
  the environment this was run in; edit them to run elsewhere.
- 11 patients yielded no crops because their annotated CT reconstructions were not available locally
  at run time (they can be added later by re-running `crop` for those IDs, then `select`).
