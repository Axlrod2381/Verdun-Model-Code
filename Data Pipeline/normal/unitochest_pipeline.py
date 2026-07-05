"""
unitochest_pipeline.py
----------------------
Mine nodule-free healthy 150x150 crops from UNITOChest, streaming slices directly
from the 84GB zip (no full extraction). Reuses the validated LIDC pipeline logic:
fixed lung mask, center-on-parenchyma gate, farthest-point spatial spread,
apparent-nodule backstop on EVERY crop. Exclusion source = UNITOChest nodule masks
(CSV 'mask' column + mask-PNG centroids); reference slices are nodule-free (mask=="").

Target: 8 crops/patient x ~475 patients ~= 3,790 negatives to reach 5,200 total.
"""
import os, io, csv, struct, math, json, random, zipfile, time
import numpy as np, cv2
from normal_healthy_pipeline import (
    read_ds_tag, read_uint16_tag, TAG_ROWS, TAG_COLS, TAG_INTERCEPT, TAG_SLOPE,
    TAG_IMG_POS, TAG_PIXEL, EXTENDED_VR, lung_mask, bin_erode, parenchyma_ok,
    apparent_nodule, SIZE, HALF, INTERIOR_ERODE, GRID_STEP, MIN_CENTER_DIST,
    EXCL_RADIUS, EXCL_Z)

ZIP="/sessions/nifty-confident-noether/mnt/uploads/unitochest.zip"
OUT_ROOT="/sessions/nifty-confident-noether/mnt/NO BS Folder/normal/finished/UNITOChest"
N_PATIENTS=480
CROPS_PER_PATIENT=8
POOL_PER_SLICE=10
N_REF_SLICES=8
PROFILE_STEP=6
SEED=1234

def hu_from_bytes(raw):
    rows=read_uint16_tag(raw,TAG_ROWS) or 512
    cols=read_uint16_tag(raw,TAG_COLS) or 512
    pix=raw.rfind(TAG_PIXEL); vr=raw[pix+4:pix+6]
    hdr=pix+12 if bytes(vr) in EXTENDED_VR else pix+8
    exp=rows*cols*2
    arr=np.frombuffer(raw[hdr:hdr+exp],dtype=np.int16).reshape(rows,cols)
    isc=read_ds_tag(raw,TAG_INTERCEPT); slp=read_ds_tag(raw,TAG_SLOPE)
    inter=float(isc) if isc else -1024.0; slope=float(slp) if slp else 1.0
    return arr.astype(np.float32)*slope+inter, rows, cols

def z_from_bytes(raw):
    pos=read_ds_tag(raw[:8192], TAG_IMG_POS)
    if pos and '\\' in pos:
        try: return float(pos.split('\\')[2])
        except (IndexError,ValueError): pass
    return None

def crop_save_bytes(raw, out_path, cx, cy, size=SIZE):
    raw=bytearray(raw); rb=bytes(raw)
    rows=read_uint16_tag(rb,TAG_ROWS) or 512
    cols=read_uint16_tag(rb,TAG_COLS) or 512
    pix=rb.rfind(TAG_PIXEL); vr=rb[pix+4:pix+6]
    hdr=pix+12 if bytes(vr) in EXTENDED_VR else pix+8
    exp=rows*cols*2
    arr=np.frombuffer(rb[hdr:hdr+exp],dtype=np.int16).reshape(rows,cols)
    cx,cy=int(round(cx)),int(round(cy)); half=size//2
    x0=max(0,min(cx-half,cols-size)); y0=max(0,min(cy-half,rows-size))
    crop=arr[y0:y0+size, x0:x0+size]
    for tag in (TAG_ROWS,TAG_COLS):
        idx=rb.find(tag)
        if idx>=0: raw[idx+8:idx+10]=struct.pack('<H',size)
    rb=bytes(raw); pix=rb.rfind(TAG_PIXEL); vr=rb[pix+4:pix+6]
    hdr=pix+12 if bytes(vr) in EXTENDED_VR else pix+8
    new=crop.tobytes(); raw[hdr-4:hdr]=struct.pack('<I',len(new))
    os.makedirs(os.path.dirname(out_path),exist_ok=True)
    open(out_path,'wb').write(bytes(raw[:hdr])+new)

def mask_centroids(raw):
    m=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_UNCHANGED)
    if m is None: return []
    if m.ndim==3: m=m[...,0]
    n,lbl,stats,cent=cv2.connectedComponentsWithStats((m>0).astype(np.uint8),8)
    return [(float(cent[i][0]),float(cent[i][1])) for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]>=3]

def build_index(zf):
    idx={}
    for split in ('train','val','test'):
        data=zf.read(f"unitochest/{split}/{split}_dataset.csv").decode()
        for r in csv.DictReader(io.StringIO(data)):
            key=(split,r['patientID'],r['exam'])
            mask=r['mask'].strip()
            img=f"unitochest/{split}/images/{r['image']}"
            mm=f"unitochest/{split}/masks/{mask}" if mask else None
            idx.setdefault(key,[]).append((int(r['slice']),img,mm))
    # collapse to one exam per patient (largest), key=(split,pid)
    best={}
    for (split,pid,exam),rows in idx.items():
        k=(split,pid)
        if k not in best or len(rows)>len(best[k][1]):
            best[k]=(exam,sorted(rows))
    return best

def in_excl(cx,cy,si,excl):
    for (ez,ex,ey) in excl:
        if abs(si-ez)<=EXCL_Z and (cx-ex)**2+(cy-ey)**2<=EXCL_RADIUS**2: return True
    return False

def run():
    os.makedirs(OUT_ROOT,exist_ok=True)
    zf=zipfile.ZipFile(ZIP)
    index=build_index(zf)
    keys=sorted(index.keys()); random.Random(SEED).shuffle(keys)
    prov=[]; log=[]; totals={'kept':0,'app_drop':0,'patients':0}
    t0=time.time(); TIME_BUDGET=float(os.environ.get('TIME_BUDGET','38'))
    # count already-done patient dirs toward the N_PATIENTS cap
    done_dirs=len([d for d in os.listdir(OUT_ROOT) if d.endswith(tuple('0123456789'))
                   and os.path.isdir(os.path.join(OUT_ROOT,d))]) if os.path.isdir(OUT_ROOT) else 0
    for (split,pid) in keys:
        if done_dirs+totals['patients']>=N_PATIENTS: break
        if time.time()-t0>TIME_BUDGET:
            print(f"[time budget hit] processed {totals['patients']} this call",flush=True); break
        outdir=os.path.join(OUT_ROOT,f"{split}_patient_{pid}")
        if os.path.isdir(outdir) and any(f.endswith('.dcm') for f in os.listdir(outdir)):
            continue   # already done (counted in done_dirs)
        exam,rows=index[(split,pid)]
        if len(rows)<30: continue
        names=[im for (_,im,_) in rows]                 # ordered by slice
        nodule_idx=set(); excl=[]
        for i,(sl,im,mm) in enumerate(rows):
            if mm:
                nodule_idx.add(i)
                try:
                    for (cx,cy) in mask_centroids(zf.read(mm)): excl.append((i,cx,cy))
                except Exception: pass
        rng=random.Random((hash((split,pid))&0xffffffff))
        # lung-area profile on nodule-free slices
        prof={}
        for i in range(0,len(names),PROFILE_STEP):
            if i in nodule_idx: continue
            try: hu,_,_=hu_from_bytes(zf.read(names[i])); prof[i]=int(lung_mask(hu).sum())
            except Exception: prof[i]=0
        if not prof or max(prof.values())==0: continue
        mx=max(prof.values()); valid=[i for i,a in prof.items() if a>0.30*mx]
        if len(valid)<3: continue
        lo,hi=min(valid),max(valid); span=hi-lo
        refs=[]
        for k in range(N_REF_SLICES):
            si=int(lo+span*(0.12+0.76*k/max(1,N_REF_SLICES-1)))
            if si in nodule_idx: continue
            if all(abs(si-r)>3 for r in refs): refs.append(si)
        # candidate pool (randomized), parenchyma + apparent gated
        pool=[]
        for ref in refs:
            try: hu,r,c=hu_from_bytes(zf.read(names[ref]))
            except Exception: continue
            interior=bin_erode(lung_mask(hu),INTERIOR_ERODE)
            pos=[(cx,cy) for cy in range(HALF,r-HALF,GRID_STEP)
                          for cx in range(HALF,c-HALF,GRID_STEP) if interior[cy,cx]]
            rng.shuffle(pos); got=0
            for (cx,cy) in pos:
                if got>=POOL_PER_SLICE: break
                if in_excl(cx,cy,ref,excl): continue
                crop=hu[cy-HALF:cy+HALF,cx-HALF:cx+HALF]
                if not parenchyma_ok(crop)[0]: continue
                if apparent_nodule(crop)[0]: totals['app_drop']+=1; continue
                pool.append((ref,cx,cy)); got+=1
        if not pool: continue
        # farthest-point spread -> distinct centers
        rng.shuffle(pool); centers=[pool[0]]
        def d3(a,b): return (a[1]-b[1])**2+(a[2]-b[2])**2+((a[0]-b[0])*6)**2
        while len(centers)<CROPS_PER_PATIENT and len(centers)<len(pool):
            best=None; bd=-1
            for cc in pool:
                if cc in centers: continue
                dm=min(d3(cc,s) for s in centers)
                if dm>bd: bd=dm; best=cc
            if best is None or bd<MIN_CENTER_DIST**2: break
            centers.append(best)
        # save: 1 slice per center (center slice); if <8 centers, top up with +-1,+-2
        plan=[(ref,cx,cy,0) for (ref,cx,cy) in centers]
        off_cycle=[1,-1,2,-2]; oi=0
        while len(plan)<CROPS_PER_PATIENT and centers:
            ref,cx,cy=centers[len(plan)%len(centers)]; o=off_cycle[oi%4]; oi+=1
            if (ref,cx,cy,o) not in plan: plan.append((ref,cx,cy,o))
            if oi>40: break
        kept=0
        for ci,(ref,cx,cy,o) in enumerate(plan,1):
            if kept>=CROPS_PER_PATIENT: break
            si=ref+o
            if si<0 or si>=len(names) or si in nodule_idx: continue
            if in_excl(cx,cy,si,excl): continue
            try: raw=zf.read(names[si]); hu,_,_=hu_from_bytes(raw)
            except Exception: continue
            crop=hu[cy-HALF:cy+HALF,cx-HALF:cx+HALF]
            if crop.shape!=(SIZE,SIZE): continue
            if not parenchyma_ok(crop)[0]: continue
            if apparent_nodule(crop)[0]: totals['app_drop']+=1; continue  # check EVERY crop
            out=os.path.join(outdir,f"healthy_{ci}_z{o:+d}.dcm")
            try:
                crop_save_bytes(raw,out,cx,cy); kept+=1; totals['kept']+=1
                prov.append(dict(split=split,patient=pid,ci=ci,z_off=o,z_mm=z_from_bytes(raw),
                                 cx=cx,cy=cy,verdict='clean'))
            except Exception as e: log.append(f"[ERR]{split}_{pid} c{ci}: {e}")
        totals['patients']+=1
        log.append(f"{split}_patient_{pid}: {len(centers)} centers -> {kept} crops")
        print(f"[{totals['patients']}/{N_PATIENTS}] {split}_patient_{pid}: {kept} crops (total {totals['kept']})",flush=True)
    # write provenance + summary
    pf=os.path.join(OUT_ROOT,"provenance.csv"); wh=not os.path.exists(pf)
    with open(pf,'a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['split','patient','ci','z_off','z_mm','cx','cy','verdict'])
        if wh: w.writeheader()
        w.writerows(prov)
    json.dump(totals,open(os.path.join(OUT_ROOT,"summary.json"),'w'),indent=2)
    open(os.path.join(OUT_ROOT,"pipeline_log.txt"),'a').write("\n".join(log)+f"\nTOTAL {totals}\n")
    print(f"\nDONE kept={totals['kept']} app_drop={totals['app_drop']} patients={totals['patients']}")

if __name__=='__main__':
    run()
