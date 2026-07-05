"""
normal_healthy_pipeline.py
--------------------------
Mine NODULE-FREE healthy-tissue crops from the confirmed-benign LIDC-IDRI CTs.
The benign nodule locations (pm2_qc.csv) are used ONLY as an exclusion mask —
we crop healthy parenchyma elsewhere in the lung.

Steps: exclusion mask -> lung seg -> valid healthy centers -> +-2 multi-slice
       -> inverted nodule-apparent check (drop/skip any crop showing a nodule)
       -> raw int16 DICOM export.

No pydicom. Raw int16 pixels, header intact (no double-intercept bug).
"""
import os, re, csv, struct, math, json, random
import numpy as np
import cv2

_K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
def bin_close(mask, it=2):
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, _K3, iterations=it).astype(bool)
def bin_erode(mask, it=3):
    return cv2.erode(mask.astype(np.uint8), _K3, iterations=it).astype(bool)
def fill_holes(mask):
    m = (mask.astype(np.uint8)) * 255
    h, w = m.shape
    ff = m.copy(); mask2 = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mask2, (0, 0), 255)
    return ((m | cv2.bitwise_not(ff)) > 0)

BASE      = "/sessions/nifty-confident-noether/mnt"
LIDC_ROOT = f"{BASE}/Confirmed benign/manifest-1781670254298/LIDC-IDRI"
QC_CSV    = f"{BASE}/NO BS Folder/benign/2nd cropped confirmed pm2/pm2_qc.csv"
OUT_ROOT  = f"{BASE}/NO BS Folder/normal/finished/LIDC-IDRI confirmed"

# ---- DICOM tag constants ----
TAG_ROWS=b'\x28\x00\x10\x00'; TAG_COLS=b'\x28\x00\x11\x00'
TAG_INTERCEPT=b'\x28\x00\x52\x10'; TAG_SLOPE=b'\x28\x00\x53\x10'
TAG_IMG_POS=b'\x20\x00\x32\x00'; TAG_PIXEL=b'\xe0\x7f\x10\x00'
EXTENDED_VR={b'OB',b'OW',b'OF',b'SQ',b'UC',b'UN',b'UR',b'UT'}

# ---- sampling params ----
SIZE=150; HALF=75
CAP_CENTERS_PER_PATIENT=8
CAP_CENTERS_PER_SLICE=3
N_REF_SLICES=7           # reference slices spread across the lung z-range
GRID_STEP=20             # candidate center grid
MIN_CENTER_DIST=40       # px between kept centers (3D)
POOL_PER_SLICE=12        # max candidate pool collected per slice (before spread selection)
LUNG_FRAC_MIN=0.82       # >=82% lung in window (interior parenchyma, not pleural edge)
INTERIOR_ERODE=13        # center must sit this deep inside lung (px)
MEAN_HU_LO=-900          # parenchyma band: reject exterior-air (too low)
MEAN_HU_HI=-650          # and reject chest-wall/soft-tissue-heavy (too high)
EXCL_RADIUS=32           # px disk around each benign nodule seed
EXCL_Z=3                 # +- slices around a nodule seed
# apparent-nodule (inverted) gate
APP_CONTRAST=200.0       # HU above lung bg counts as soft-tissue opacity
APP_MIN_BLOB=100         # px area: conservative backstop -> only flag LARGE compact masses
APP_CIRC=0.62            # circularity: round => mass-like (vessels branch/elongate)
APP_SOLID_HU=-450.0      # blob mean HU must be genuinely soft-tissue dense

def read_ds_tag(raw, tag4):
    raw_b=bytes(raw); idx=raw_b.find(tag4)
    while idx>=0:
        vr=raw_b[idx+4:idx+6]
        if len(vr)==2 and vr[0:1].isalpha() and vr[1:2].isalpha():
            length=struct.unpack('<H',raw_b[idx+6:idx+8])[0]
            if 0<length<400: return raw_b[idx+8:idx+8+length].decode('latin-1','replace').strip('\x00 ')
        else:
            length=struct.unpack('<I',raw_b[idx+4:idx+8])[0]
            if 0<length<400: return raw_b[idx+8:idx+8+length].decode('latin-1','replace').strip('\x00 ')
        idx=raw_b.find(tag4,idx+1)
    return None

def read_uint16_tag(raw, tag4):
    raw_b=bytes(raw); idx=raw_b.find(tag4)
    while idx>=0:
        vr=raw_b[idx+4:idx+6]
        if vr[0:1].isalpha() and vr[1:2].isalpha():
            length=struct.unpack('<H',raw_b[idx+6:idx+8])[0]
            if length==2: return struct.unpack('<H',raw_b[idx+8:idx+10])[0]
        else:
            length=struct.unpack('<I',raw_b[idx+4:idx+8])[0]
            if length==2: return struct.unpack('<H',raw_b[idx+8:idx+10])[0]
        idx=raw_b.find(tag4,idx+1)
    return None

def read_raw_px(filepath):
    with open(filepath,'rb') as f: raw=f.read()
    rows=read_uint16_tag(raw,TAG_ROWS) or 512
    cols=read_uint16_tag(raw,TAG_COLS) or 512
    pix_idx=raw.rfind(TAG_PIXEL)
    if pix_idx<0: raise ValueError("no pixel data")
    vr=raw[pix_idx+4:pix_idx+6]
    hdr_end=pix_idx+12 if bytes(vr) in EXTENDED_VR else pix_idx+8
    pdata=raw[hdr_end:]; exp=rows*cols*2
    if len(pdata)>=exp: pdata=pdata[:exp]
    return np.frombuffer(pdata,dtype=np.int16).reshape(rows,cols),rows,cols

def read_pixels_hu(filepath):
    arr,rows,cols=read_raw_px(filepath)
    with open(filepath,'rb') as f: raw=f.read()
    isc=read_ds_tag(raw,TAG_INTERCEPT); slp=read_ds_tag(raw,TAG_SLOPE)
    intercept=float(isc) if isc else -1024.0
    slope=float(slp) if slp else 1.0
    return arr.astype(np.float32)*slope+intercept,rows,cols

def read_z(filepath):
    with open(filepath,'rb') as f: raw=f.read(8192)
    pos=read_ds_tag(raw,TAG_IMG_POS)
    if pos:
        p=pos.split('\\')
        if len(p)>=3:
            try: return float(p[2])
            except ValueError: pass
    return None

def crop_save_raw(ct_path,out_path,cx,cy,size=SIZE):
    with open(ct_path,'rb') as f: raw=bytearray(f.read())
    raw_arr,rows,cols=read_raw_px(ct_path)
    cx,cy=int(round(cx)),int(round(cy)); half=size//2
    x0=max(0,min(cx-half,cols-size)); y0=max(0,min(cy-half,rows-size))
    crop=raw_arr[y0:y0+size,x0:x0+size]
    for tag in (TAG_ROWS,TAG_COLS):
        idx=bytes(raw).find(tag)
        if idx>=0: raw[idx+8:idx+10]=struct.pack('<H',size)
    raw_b=bytes(raw); pix_idx=raw_b.rfind(TAG_PIXEL)
    vr=raw_b[pix_idx+4:pix_idx+6]
    hdr_end=pix_idx+12 if bytes(vr) in EXTENDED_VR else pix_idx+8
    new_pix=crop.tobytes()
    raw[hdr_end-4:hdr_end]=struct.pack('<I',len(new_pix))
    out=bytes(raw[:hdr_end])+new_pix
    os.makedirs(os.path.dirname(out_path),exist_ok=True)
    with open(out_path,'wb') as f: f.write(out)

def find_ct_series(pat_path):
    best=None; bc=0
    for root,dirs,files in os.walk(pat_path):
        low=root.lower()
        if any(k in low for k in ('segmentation','annotation','evaluation','nodule')): continue
        dcms=[f for f in files if f.lower().endswith('.dcm')]
        if len(dcms)>bc: bc=len(dcms); best=root
    return best if bc>=10 else None

def build_z_index(series):
    e=[]
    for f in os.listdir(series):
        if not f.lower().endswith('.dcm'): continue
        p=os.path.join(series,f); z=read_z(p)
        if z is not None: e.append((z,p))
    e.sort(key=lambda x:x[0])
    return e

def lung_mask(hu):
    """Keep only INTERIOR air-filled components (lungs); drop border-connected
    exterior air. fill_holes then reincorporates vessels/bronchi inside lung."""
    h,w=hu.shape
    air=(hu< -320).astype(np.uint8)          # lung + exterior air + gut
    n,lbl,stats,_=cv2.connectedComponentsWithStats(air,connectivity=8)
    keep=np.zeros((h,w),bool)
    for i in range(1,n):
        area=int(stats[i,cv2.CC_STAT_AREA])
        x=stats[i,cv2.CC_STAT_LEFT]; y=stats[i,cv2.CC_STAT_TOP]
        cw=stats[i,cv2.CC_STAT_WIDTH]; ch=stats[i,cv2.CC_STAT_HEIGHT]
        touches_border=(x<=0 or y<=0 or x+cw>=w or y+ch>=h)
        if touches_border: continue           # exterior air / open gut
        if area<2000 or area>0.45*h*w: continue# plausible lung size
        keep|=(lbl==i)
    keep=bin_close(keep,3)
    keep=fill_holes(keep)                       # fill intra-lung vessel holes only
    return keep.astype(bool)

def apparent_nodule(hu):
    """INVERTED nodule-apparent check. Return True if a NODULE-LIKE opacity is
    apparent. A nodule is compact/round and nodule-sized; branching or elongated
    vessels are normal healthy-lung signal and must NOT trigger a drop."""
    H,W=hu.shape
    lm=(hu>=-980)&(hu<=-600)
    lung_hu=float(hu[lm].mean()) if lm.sum()>50 else -900.0
    lung_region=((hu>=-1000)&(hu<=-500))        # anything lung/air = "lung annulus" material
    soft=(hu>(lung_hu+APP_CONTRAST)).astype(np.uint8)   # soft-tissue opacity above lung bg
    if soft.sum()==0: return False,lung_hu,0
    n,lbl,stats,_=cv2.connectedComponentsWithStats(soft,connectivity=8)
    worst=0
    for i in range(1,n):
        area=int(stats[i,cv2.CC_STAT_AREA])
        if area<APP_MIN_BLOB or area>1400: continue     # vessel-small / hilum-huge
        x=stats[i,cv2.CC_STAT_LEFT]; y=stats[i,cv2.CC_STAT_TOP]
        w=stats[i,cv2.CC_STAT_WIDTH]; h=stats[i,cv2.CC_STAT_HEIGHT]
        if x<=1 or y<=1 or x+w>=W-1 or y+h>=H-1: continue  # touches frame = wall/mediastinum
        if max(w,h)/max(1,min(w,h))>1.7: continue          # elongated = vessel
        comp=(lbl==i).astype(np.uint8)
        cnts,_=cv2.findContours(comp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        per=cv2.arcLength(cnts[0],True)
        if per<=0 or 4*math.pi*area/(per*per)<APP_CIRC: continue  # not round => vessel/irregular
        if float(hu[comp.astype(bool)].mean())<APP_SOLID_HU: continue  # not dense enough
        ring=(cv2.dilate(comp,_K3,iterations=4).astype(bool)) & (~comp.astype(bool))
        if ring.sum()==0 or float(lung_region[ring].mean())<0.55: continue  # need lung annulus
        worst=max(worst,area)
    return (worst>0), lung_hu, worst

def parenchyma_ok(crop):
    """A good negative = CENTERED on lung parenchyma, little exterior air, and
    enough lung overall (the 150px FOV may clip vessels/wall like the positives
    do). Focal nodules are handled separately by apparent_nodule()."""
    s=crop.shape[0]; c=s//2; r=25
    cen=crop[c-r:c+r, c-r:c+r]
    cen_lung=float(((cen>=-950)&(cen<=-450)).mean())   # center sits on parenchyma
    cen_soft=float((cen> -200).mean())                 # center not solid tissue
    frac_air =float((crop< -980).mean())               # exterior air / big airspace
    frac_lung=float(((crop>=-950)&(crop<=-500)).mean())# overall lung presence
    ok=(cen_lung>=0.60) and (cen_soft<=0.12) and (frac_air<=0.12) and (frac_lung>=0.40)
    return ok,round(frac_air,3),round(cen_soft,3),round(frac_lung,3)

def in_exclusion(cx,cy,si,excl):
    for (ez,ecx,ecy) in excl:
        if abs(si-ez)<=EXCL_Z and (cx-ecx)**2+(cy-ecy)**2<=EXCL_RADIUS**2:
            return True
    return False

def run():
    os.makedirs(OUT_ROOT,exist_ok=True)
    # exclusion seeds per patient from pm2_qc
    excl_rows={}
    with open(QC_CSV) as f:
        for r in csv.DictReader(f):
            excl_rows.setdefault(r['patient'],[]).append(
                (float(r['z']),float(r['cx']),float(r['cy'])))

    patients=sorted(p for p in os.listdir(LIDC_ROOT) if p.startswith('LIDC-IDRI-'))
    prov=[]; log=[]; totals={'kept':0,'app_drop':0,'skip_center':0}
    per_patient={}

    for name in patients:
        num=name.replace('LIDC-IDRI-',''); key=f'patient-{num}'
        outdir=os.path.join(OUT_ROOT,key)
        if os.path.isdir(outdir):
            done=[f for f in os.listdir(outdir) if f.endswith('.dcm')]
            if done:
                per_patient[key]=len(done)
                print(f"{key}: resume-skip ({len(done)} crops exist)",flush=True); continue
        series=find_ct_series(os.path.join(LIDC_ROOT,name))
        if not series:
            log.append(f"[SKIP] {key}: no CT series"); continue
        zidx=build_z_index(series)
        if len(zidx)<20:
            log.append(f"[SKIP] {key}: short stack ({len(zidx)})"); continue

        # map benign-nodule z(mm) -> slice index for exclusion
        zs=np.array([z for z,_ in zidx])
        seeds=excl_rows.get(key,[])
        excl=[]
        for (zmm,cx,cy) in seeds:
            si=int(np.argmin(np.abs(zs-zmm))); excl.append((si,cx,cy))

        # lung-area profile (subsampled) to find mid-lung and max area
        prof={}
        for si in range(0,len(zidx),4):
            try:
                hu,_,_=read_pixels_hu(zidx[si][1]); prof[si]=int(lung_mask(hu).sum())
            except Exception: prof[si]=0
        if not prof or max(prof.values())==0:
            log.append(f"[SKIP] {key}: no lung"); continue
        maxarea=max(prof.values())
        valid=[si for si,a in prof.items() if a>0.30*maxarea]
        if len(valid)<3:
            log.append(f"[SKIP] {key}: too few lung slices"); continue
        lo,hi=min(valid),max(valid)
        # reference slices spread across mid 25-75% of lung z-range, spaced so +-2 stacks don't overlap
        span=hi-lo; refs=[]
        for k in range(N_REF_SLICES):
            si=int(lo+span*(0.15+0.70*k/max(1,N_REF_SLICES-1)))
            if all(abs(si-r)>5 for r in refs): refs.append(si)

        rng=random.Random(hash(key)&0xffffffff)   # deterministic per-patient shuffle
        # 1) collect a diverse candidate POOL across all reference slices (randomized order)
        pool=[]
        for ref in refs:
            try: hu,rows,cols=read_pixels_hu(zidx[ref][1])
            except Exception: continue
            lm=lung_mask(hu); interior=bin_erode(lm,INTERIOR_ERODE)
            positions=[(cx,cy) for cy in range(HALF,rows-HALF,GRID_STEP)
                                for cx in range(HALF,cols-HALF,GRID_STEP) if interior[cy,cx]]
            rng.shuffle(positions)                 # unbias scan order (was posterior-heavy)
            got=0
            for (cx,cy) in positions:
                if got>=POOL_PER_SLICE: break
                if in_exclusion(cx,cy,ref,excl): continue
                crop=hu[cy-HALF:cy+HALF,cx-HALF:cx+HALF]
                if not parenchyma_ok(crop)[0]: continue
                if apparent_nodule(crop)[0]:
                    totals['app_drop']+=1; continue
                pool.append((ref,cx,cy)); got+=1
        # 2) farthest-point selection => spread across both lungs + anterior/posterior + z
        centers=[]; pat_kept=0
        if pool:
            rng.shuffle(pool)
            centers=[pool[0]]
            def d3(a,b): return (a[1]-b[1])**2+(a[2]-b[2])**2+((a[0]-b[0])*6)**2
            while len(centers)<CAP_CENTERS_PER_PATIENT and len(centers)<len(pool):
                best=None; bestd=-1
                for c in pool:
                    if c in centers: continue
                    dmin=min(d3(c,s) for s in centers)
                    if dmin>bestd: bestd=dmin; best=c
                if best is None or bestd<MIN_CENTER_DIST**2: break
                centers.append(best)

        # multi-slice +-2 export, apparent-checked per slice
        for ci,(ref,cx,cy) in enumerate(centers,1):
            for off in range(-2,3):
                si=ref+off
                if si<0 or si>=len(zidx): continue
                if in_exclusion(cx,cy,si,excl): continue
                try: hu,_,_=read_pixels_hu(zidx[si][1])
                except Exception: continue
                crop=hu[cy-HALF:cy+HALF,cx-HALF:cx+HALF]
                if crop.shape!=(SIZE,SIZE): continue
                ok,fa,fs,fp=parenchyma_ok(crop)
                if not ok:
                    totals['app_drop']+=1; continue   # drifted off parenchyma on this slice
                app,lung_hu,blob=apparent_nodule(crop)
                if app:
                    totals['app_drop']+=1; continue
                out=os.path.join(OUT_ROOT,key,f"healthy_{ci}_z{off:+d}.dcm")
                try:
                    crop_save_raw(zidx[si][1],out,cx,cy); totals['kept']+=1; pat_kept+=1
                    prov.append(dict(patient=key,center=ci,z_off=off,
                                     z_mm=round(zidx[si][0],1),cx=cx,cy=cy,
                                     lung_hu=round(lung_hu,1),blob_px=blob,verdict='clean'))
                except Exception as e:
                    log.append(f"  [ERR] {key} c{ci} z{off}: {e}")
        per_patient[key]=pat_kept
        log.append(f"[PAT] {key}: {len(centers)} centers -> {pat_kept} crops")
        print(f"{key}: {len(centers)} centers -> {pat_kept} crops",flush=True)

    # recount all crops from disk (covers resumed patients too)
    disk_total=0; disk_pp={}
    for d in sorted(os.listdir(OUT_ROOT)):
        pdir=os.path.join(OUT_ROOT,d)
        if os.path.isdir(pdir) and d.startswith('patient-'):
            c=len([f for f in os.listdir(pdir) if f.endswith('.dcm')])
            disk_pp[d]=c; disk_total+=c
    per_patient=disk_pp; totals['kept']=disk_total
    # write/append provenance + log
    prov_path=os.path.join(OUT_ROOT,"provenance.csv")
    write_header=not os.path.exists(prov_path)
    with open(prov_path,'a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['patient','center','z_off','z_mm','cx','cy','lung_hu','blob_px','verdict'])
        if write_header: w.writeheader()
        w.writerows(prov)
    with open(os.path.join(OUT_ROOT,"pipeline_log.txt"),'w') as f:
        f.write("\n".join(log)+f"\n\nTOTAL kept={totals['kept']} apparent_drop={totals['app_drop']}")
    json.dump({'totals':totals,'per_patient':per_patient},
              open(os.path.join(OUT_ROOT,"summary.json"),'w'),indent=2)
    print(f"\nDONE kept={totals['kept']} apparent_drop={totals['app_drop']} patients={len(per_patient)}")
    return totals,per_patient

if __name__=='__main__':
    run()
