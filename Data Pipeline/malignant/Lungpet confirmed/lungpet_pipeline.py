#!/usr/bin/env python3
"""
Lung-PET-CT-Dx malignant nodule pipeline (combined).

Two stages in one file:
  1. crop   -> for every VOC-annotated slice, crop it and its +-2 z-neighbors within the
               SAME CT series (BOTH reconstruction kernels), apparent-QC each z level, and
               write the passing ones as 150x150 raw int16 DICOM. Failing levels are not kept.
  2. select -> curate the QC-passed pool down to the BEST 5000 crops: best-centered + strongest
               malignant conspicuity, one scanner/kernel per patient, every patient kept.

Usage:
  python lungpet_pipeline.py crop [PID ...]     # crop + QC (resumable; optional patient filter)
  python lungpet_pipeline.py select [--apply]   # curate to best 5000 (dry-run without --apply)
  python lungpet_pipeline.py all  [--apply]      # crop everything, then curate

Dependencies: numpy, and nodule_qc.py (raw-DICOM parser + scoring + HU-safe crop writer),
which must sit next to this file or on PYTHONPATH.
"""
import os, sys, csv, re, glob, collections, statistics as st
import xml.etree.ElementTree as ET
import numpy as np

# nodule_qc.py provides: _read_ds, TAG_INST, TAG_POS, read_hu, lung_ref, refine_center,
# center_hu, centroid_offset, classify_type, crop_save_raw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nodule_qc as qc

# ----------------------------------------------------------------------------- config / paths
TAG_SOP  = b'\x08\x00\x18\x00'     # SOPInstanceUID
TAG_WC   = b'\x28\x00\x50\x10'     # WindowCenter
TAG_WW   = b'\x28\x00\x51\x10'     # WindowWidth

ROOT      = "/sessions/kind-exciting-volta/mnt/Lung Pet CT dx confirmed"
MANIFESTS = ["manifest-1608669183333", "manifest-1782811012258"]
ANN       = "/sessions/kind-exciting-volta/mnt/outputs/ann_inspect/Annotation"
OUT       = "/sessions/kind-exciting-volta/mnt/NO BS Folder/malignant/finished/Lung-Pet-Ct-confirmed"

# QC gates (apparent malignant nodule usable for CNN): apparent is primary, centering lenient.
AIR = -820; CONTRAST_MIN = 180; OFFSET_MAX = 35; ZWIN = 2; SIZE = 150

# Curation
TARGET  = 5000
TYPE_W  = {"solid": 1.0, "part-solid": 0.90, "calcified": 0.70, "GGO": 0.60, "sub-GGO": 0.40, "air": 0.0}

# =========================================================================== STAGE 1: crop + QC
def load_metadata():
    """patient -> list of (series_desc, abs_series_dir) for CT series only."""
    pat = {}
    for man in MANIFESTS:
        mp = os.path.join(ROOT, man, "metadata.csv")
        if not os.path.exists(mp):
            continue
        for row in csv.DictReader(open(mp)):
            if row.get("Modality") != "CT":
                continue
            pid = row["Subject ID"].replace("Lung_Dx-", "")
            absdir = os.path.join(ROOT, man, row["File Location"].lstrip("./"))
            pat.setdefault(pid, []).append((row.get("Series Description", "").strip(), absdir))
    return pat

def index_series(series_dir):
    """sop->fp, and slices sorted by z: list of (instnum, z, fp, sop)."""
    sop2fp, rows = {}, []
    if not os.path.isdir(series_dir):
        return sop2fp, []
    for fn in os.listdir(series_dir):
        if not fn.lower().endswith(".dcm"):
            continue
        fp = os.path.join(series_dir, fn)
        try:
            raw = open(fp, 'rb').read(8000)
        except Exception:
            continue                                   # cloud-only/unreadable -> skip fast
        sop  = qc._read_ds(raw, TAG_SOP)
        inst = qc._read_ds(raw, qc.TAG_INST)
        pos  = qc._read_ds(raw, qc.TAG_POS)
        z    = float(pos.split('\\')[2]) if pos and '\\' in pos else None
        inst = int(inst) if inst else 10**9
        if sop:
            sop2fp[sop] = fp
        rows.append((inst, z, fp, sop))
    rows.sort(key=lambda t: (t[1] if t[1] is not None else t[0]))   # sort by z
    return sop2fp, rows

def box_center(xmlfp):
    """VOC bounding-box center (pixel coords)."""
    b = ET.parse(xmlfp).getroot().find(".//bndbox")
    if b is None:
        return None
    x = (float(b.findtext("xmin")) + float(b.findtext("xmax"))) / 2.0
    y = (float(b.findtext("ymin")) + float(b.findtext("ymax"))) / 2.0
    return x, y

def qc_slice(hu, cx, cy):
    """Apparent-nodule check for one slice at (cx,cy). Returns keep flag + metrics."""
    lung = qc.lung_ref(hu)
    # wider search so the crop can lock onto the tumor when the box's geometric
    # center sits slightly off the nodule (large/peripheral/cavitary boxes)
    rx, ry = qc.refine_center(hu, cx, cy, lung, roi=64, cap=22)
    chu = qc.center_hu(hu, rx, ry); contrast = chu - lung
    off = qc.centroid_offset(hu, rx, ry, lung)
    ok = (chu > AIR) and (contrast >= CONTRAST_MIN) and (off < OFFSET_MAX)
    return ok, rx, ry, round(chu), round(contrast), round(off, 1), qc.classify_type(chu)

def process_patient(pid, report, PAT):
    ann_dir = os.path.join(ANN, pid)
    if not os.path.isdir(ann_dir):
        return
    xmls = {os.path.splitext(f)[0]: os.path.join(ann_dir, f)
            for f in os.listdir(ann_dir) if f.lower().endswith(".xml")}
    written = set()
    for sdesc, sdir in PAT.get(pid, []):
        sop2fp, rows = index_series(sdir)
        if not rows:
            continue
        order = [r[2] for r in rows]                    # fp in z order
        fp2idx = {fp: i for i, fp in enumerate(order)}
        ann_in = [(sop, xmls[sop]) for sop in xmls if sop in sop2fp]   # annotated slices in THIS series
        kern = re.sub(r'[^A-Za-z0-9]+', '_', sdesc) or "series"
        for sop, xmlfp in ann_in:
            ctr = box_center(xmlfp)
            if not ctr:
                continue
            cx, cy = ctr
            ci = fp2idx[sop2fp[sop]]
            for dz in range(-ZWIN, ZWIN + 1):
                j = ci + dz
                if j < 0 or j >= len(order):
                    continue
                fp = order[j]
                key = (kern, fp)
                if key in written:                      # slice already cropped for this series
                    continue
                try:
                    hu = qc.read_hu(fp)                  # may EDEADLK on an un-materialized iCloud file
                except Exception as e:
                    report.append(dict(patient=pid, kernel=sdesc, center_sop=sop, dz=dz,
                                       verdict="error", reason=f"read:{str(e)[:40]}"))
                    continue
                ok, rx, ry, chu, contrast, off, tdet = qc_slice(hu, cx, cy)
                inst, z = rows[j][0], rows[j][1]
                rec = dict(patient=pid, kernel=sdesc, center_sop=sop, dz=dz, inst=inst,
                           z=round(z, 1) if z is not None else None, center_hu=chu,
                           contrast=contrast, offset_px=off, type=tdet,
                           verdict="kept" if ok else "deleted",
                           reason="apparent" if ok else f"hu={chu},contrast={contrast},off={off}")
                if ok:
                    outfp = os.path.join(OUT, f"Lung_Dx-{pid}", kern,
                                         f"sop{sop[-8:]}_dz{dz:+d}_inst{inst}.dcm")
                    qc.crop_save_raw(fp, outfp, rx, ry, SIZE)
                    written.add(key)
                    rec["out"] = outfp
                report.append(rec)

def run_crop(only=None):
    """Resumable crop+QC over all patients (or the given PID subset)."""
    import time
    BUDGET = float(os.environ.get("BUDGET", "38"))      # seconds per chunk (stay under 45s cap)
    PAT = load_metadata()
    pids = sorted(p for p in os.listdir(ANN) if os.path.isdir(os.path.join(ANN, p)) and p in PAT)
    if only:
        pids = [p for p in pids if p in only]
    os.makedirs(OUT, exist_ok=True)
    rep_fp  = os.path.join(OUT, "qc_report.csv")
    prog_fp = os.path.join(OUT, "_done.txt")
    cols = ["patient", "kernel", "center_sop", "dz", "inst", "z", "center_hu", "contrast",
            "offset_px", "type", "verdict", "reason", "out"]
    done = set(open(prog_fp).read().split()) if os.path.exists(prog_fp) else set()
    todo = [p for p in pids if p not in done]
    new_report = not os.path.exists(rep_fp)
    t0 = time.time()
    f = open(rep_fp, "a", newline=""); w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    if new_report:
        w.writeheader()
    pf = open(prog_fp, "a"); processed = 0
    for pid in todo:
        rep = []
        try:
            process_patient(pid, rep, PAT)
        except Exception as e:
            rep.append(dict(patient=pid, verdict="ERROR", reason=str(e)[:120]))
        if not rep:
            rep = [dict(patient=pid, verdict="none", reason="no annotated slice in any local CT series")]
        for r in rep:
            w.writerow(r)
        f.flush(); pf.write(pid + "\n"); pf.flush(); processed += 1
        kept = sum(1 for r in rep if r.get("verdict") == "kept")
        print(f"{pid}: {len(rep)} rows, {kept} kept  ({time.time()-t0:.0f}s)", flush=True)
        if time.time() - t0 > BUDGET:
            break
    remaining = len(todo) - processed
    print(f"CHUNK_DONE processed={processed} remaining={remaining}", flush=True)
    if remaining == 0:
        print("ALL_DONE", flush=True)

# ============================================================================ STAGE 2: curate
def _f(x, d=0.0):
    try:
        return float(x)
    except Exception:
        return d

def score(r):
    """'Best' = tight centering + malignant conspicuity (contrast + density/type)."""
    off = _f(r["offset_px"], 99); con = _f(r["contrast"], 0)
    center     = max(0.0, 1.0 - off / 30.0)             # tightest centering -> 1.0
    contrast_s = min(con / 600.0, 1.0)                  # conspicuity above lung
    type_s     = TYPE_W.get(r["type"], 0.0)             # solid/dense = malignant-characteristic
    return round(0.5 * center + 0.5 * (0.6 * contrast_s + 0.4 * type_s), 4)

def run_select(apply=False):
    rows = [r for r in csv.DictReader(open(os.path.join(OUT, "qc_report.csv")))
            if r["verdict"] == "kept" and r.get("out")]
    rows = [r for r in rows if os.path.exists(r["out"])]     # real files only (no empty patients)
    for r in rows:
        r["_s"] = score(r)

    bypat = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        bypat[r["patient"]][r["kernel"]].append(r)

    # 1) one scanner/kernel per patient = the kernel whose crops are best on average
    pool = []
    for pid in sorted(bypat):
        ks = bypat[pid]
        bestk = max(ks, key=lambda k: sum(x["_s"] for x in ks[k]) / len(ks[k]))
        pool += ks[bestk]
    print(f"one-kernel/patient pool: {len(pool)} crops, {len(bypat)} patients")

    # 2) trim to TARGET keeping highest scores; protect each patient's single best
    anchor = {}
    for r in sorted(pool, key=lambda r: -r["_s"]):
        anchor.setdefault(r["patient"], r)
    protected = set(id(r) for r in anchor.values())
    rest = sorted([r for r in pool if id(r) not in protected], key=lambda r: -r["_s"])
    selected = list(anchor.values()) + rest[:max(0, TARGET - len(anchor))]
    print(f"selected: {len(selected)} crops | patients: {len({r['patient'] for r in selected})}")
    print(f"selected median offset={st.median([_f(r['offset_px']) for r in selected]):.1f}px "
          f"median contrast={round(st.median([_f(r['contrast']) for r in selected]))}HU")
    print("selected type mix:", dict(collections.Counter(r["type"] for r in selected)))

    keep  = set(r["out"] for r in selected)
    todel = set(r["out"] for r in rows) - keep
    assert {r["patient"] for r in selected} == set(bypat.keys()), "patient lost!"
    print(f"on disk:{len(set(r['out'] for r in rows))} keep:{len(keep)} delete:{len(todel)}")

    if apply:
        d = 0
        for p in todel:
            try:
                os.remove(p); d += 1
            except Exception:
                pass
        with open(os.path.join(OUT, "final_selection.csv"), "w", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=["patient", "kernel", "center_sop", "dz", "inst", "z",
                                               "center_hu", "contrast", "offset_px", "type", "_s", "out"],
                               extrasaction="ignore")
            w.writeheader()
            for r in sorted(selected, key=lambda r: (r["patient"], -r["_s"])):
                w.writerow(r)
        print(f"DELETED {d}. wrote final_selection.csv ({len(selected)} rows)")
    else:
        print("DRY RUN (pass --apply to delete non-selected crops and write final_selection.csv)")

# ===================================================================================== CLI
def main():
    args = sys.argv[1:]
    cmd = args[0] if args else "all"
    apply = "--apply" in args
    pids = [a for a in args[1:] if not a.startswith("-")]
    if cmd == "crop":
        run_crop(only=pids or None)
    elif cmd == "select":
        run_select(apply=apply)
    elif cmd == "all":
        run_crop(only=pids or None)
        run_select(apply=apply)
    else:
        print(__doc__)
        sys.exit(1)

if __name__ == "__main__":
    main()
