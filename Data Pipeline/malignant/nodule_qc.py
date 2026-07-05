#!/usr/bin/env python3
"""
nodule-apparent-qc: verify a nodule of a given type is APPARENT + CENTERED + on the right Z,
classify its type, and write an ENHANCED 150x150 crop (+ raw-int16 DICOM) for CNN training.

No pydicom (raw struct parsing). Enhancement uses OpenCV if present, else a PIL fallback.
Grounded in the malignant/benign nodule pipelines and nodule-crop-tricks.

CLI:
  python nodule_qc.py --ct <series_dir> --x 135 --y 303 --image 142 --type solid --out ./qc_out
  python nodule_qc.py --batch nodules.csv --out ./qc_out      # csv cols: ct_path,x,y,image,type
"""
import os, sys, struct, glob, csv, json, argparse
import numpy as np

# ---------------- DICOM tags ----------------
TAG_ROWS=b'\x28\x00\x10\x00'; TAG_COLS=b'\x28\x00\x11\x00'
TAG_INT =b'\x28\x00\x52\x10'; TAG_SLO =b'\x28\x00\x53\x10'
TAG_POS =b'\x20\x00\x32\x00'; TAG_INST=b'\x20\x00\x13\x00'
TAG_PIX =b'\xe0\x7f\x10\x00'; TAG_SPACING=b'\x28\x00\x30\x00'

def _read_ds(raw,tag):
    i=raw.find(tag)
    while i>=0:
        vr=raw[i+4:i+6]
        if len(vr)==2 and vr[0:1].isalpha() and vr[1:2].isalpha():
            l=struct.unpack('<H',raw[i+6:i+8])[0]
            if 0<l<200: return raw[i+8:i+8+l].decode('latin-1','replace').strip('\x00 ')
        else:
            l=struct.unpack('<I',raw[i+4:i+8])[0]
            if 0<l<200: return raw[i+8:i+8+l].decode('latin-1','replace').strip('\x00 ')
        i=raw.find(tag,i+1)
    return None

def _u16(raw,tag):
    i=raw.find(tag)
    while i>=0:
        vr=raw[i+4:i+6]
        if vr[0:1].isalpha() and vr[1:2].isalpha():
            l=struct.unpack('<H',raw[i+6:i+8])[0]
            if l==2: return struct.unpack('<H',raw[i+8:i+10])[0]
        else:
            l=struct.unpack('<I',raw[i+4:i+8])[0]
            if l==2: return struct.unpack('<H',raw[i+8:i+10])[0]
        i=raw.find(tag,i+1)
    return None

def read_hu(fp):
    raw=open(fp,'rb').read()
    R=_u16(raw,TAG_ROWS) or 512; C=_u16(raw,TAG_COLS) or 512
    p=raw.rfind(TAG_PIX); vr=raw[p+4:p+6]
    st=p+12 if bytes(vr) in (b'OB',b'OW',b'OF',b'UN') else p+8
    px=np.frombuffer(raw[st:st+R*C*2],dtype=np.int16).reshape(R,C)
    it=_read_ds(raw,TAG_INT); sl=_read_ds(raw,TAG_SLO)
    return px.astype(np.float32)*(float(sl) if sl else 1.0)+(float(it) if it else -1024.0)

def pixel_spacing(fp):
    s=_read_ds(open(fp,'rb').read(6000),TAG_SPACING)
    try: return float(s.split('\\')[0])
    except: return None

def build_index(series_dir):
    """Return sorted list of (instance_number, z, filepath) for the largest CT folder."""
    best=None; bc=0
    for root,_,files in os.walk(series_dir):
        if 'segment' in root.lower(): continue
        d=[f for f in files if f.lower().endswith('.dcm')]
        if len(d)>bc: bc=len(d); best=root
    if not best: return [],None
    out=[]
    for fn in os.listdir(best):
        if not fn.lower().endswith('.dcm'): continue
        fp=os.path.join(best,fn); raw=open(fp,'rb').read(6000)
        inst=_read_ds(raw,TAG_INST); pos=_read_ds(raw,TAG_POS)
        z=float(pos.split('\\')[2]) if pos and '\\' in pos else None
        out.append((int(inst) if inst else 10**9, z, fp))
    out.sort(key=lambda t:t[0])
    return out, best

def file_for_image(index, image_num):
    """Map a 1-based image/InstanceNumber to a file; fall back to position in the sorted stack."""
    for inst,z,fp in index:
        if inst==image_num: return fp,z
    if 1<=image_num<=len(index):
        _,z,fp=index[image_num-1]; return fp,z
    return None,None

# ---------------- scoring helpers (nodule-crop-tricks) ----------------
def variants(x,y,m=511):
    x=float(x);y=float(y)
    return {'original':(x,y),'swapped':(y,x),'flip_x':(m-x,y),'flip_y':(x,m-y),
            'flip_both':(m-x,m-y),'swap_flipx':(m-y,x),'swap_flipy':(y,m-x)}

def lung_ref(hu):
    msk=(hu>=-980)&(hu<=-600); return float(hu[msk].mean()) if msk.sum()>50 else -900.0

def center_hu(hu,cx,cy,r=10):
    cx,cy=int(round(cx)),int(round(cy)); p=hu[max(0,cy-r):cy+r,max(0,cx-r):cx+r]
    return float(p.mean()) if p.size else -1000.0

def lung_surround(hu,cx,cy,r0=20,r1=46):
    h,w=hu.shape; yy,xx=np.ogrid[:h,:w]; d=np.sqrt((xx-cx)**2+(yy-cy)**2); ring=(d>=r0)&(d<=r1)
    return float((hu[ring]<-600).mean()) if ring.sum()>=20 else 0.0

def blob_score(hu,cx,cy,lung):
    if not (15<=cx<=hu.shape[1]-15 and 15<=cy<=hu.shape[0]-15): return -1.0
    c=center_hu(hu,cx,cy)
    if c<=-820: return -1.0
    cs=min(1.0,max(0.0,(c-lung)/550.0))
    return cs*(0.35+0.65*lung_surround(hu,cx,cy))

def refine_center(hu,cx,cy,lung,roi=46,cap=12):
    cx,cy=int(round(cx)),int(round(cy)); h=roi//2
    x0=max(0,cx-h); y0=max(0,cy-h); sub=hu[y0:y0+roi,x0:x0+roi]
    if sub.size==0: return float(cx),float(cy)
    above=np.clip(sub-(lung+200),0,None)
    if above.sum()<=0: return float(cx),float(cy)
    ys,xs=np.mgrid[0:sub.shape[0],0:sub.shape[1]]
    ncx=x0+float((xs*above).sum()/above.sum()); ncy=y0+float((ys*above).sum()/above.sum())
    if (ncx-cx)**2+(ncy-cy)**2>cap**2: return float(cx),float(cy)
    return ncx,ncy

def centroid_offset(hu,cx,cy,lung,box=70):
    cx,cy=int(round(cx)),int(round(cy)); h=box//2
    x0=max(0,cx-h); y0=max(0,cy-h); roi=hu[y0:y0+box,x0:x0+box]
    above=np.clip(roi-(lung+150),0,None)
    if above.sum()<=0: return 99.0
    ys,xs=np.mgrid[0:roi.shape[0],0:roi.shape[1]]
    rcx=float((xs*above).sum()/above.sum()); rcy=float((ys*above).sum()/above.sum())
    return float(np.sqrt((rcx-(cx-x0))**2+(rcy-(cy-y0))**2))

def classify_type(hu_val):
    if hu_val>200:  return 'calcified'
    if hu_val>-100: return 'solid'
    if hu_val>-400: return 'part-solid'
    if hu_val>-700: return 'GGO'
    if hu_val>-850: return 'sub-GGO'
    return 'air'

ADJ={'calcified':['solid'],'solid':['calcified','part-solid'],'part-solid':['solid','GGO'],
     'GGO':['part-solid','sub-GGO'],'sub-GGO':['GGO'],'air':[]}

# ---------------- enhancement ----------------
def lung_u8(hu,wl=-600,ww=1500):
    lo=wl-ww/2; hi=wl+ww/2; return (np.clip((hu-lo)/(hi-lo),0,1)*255).astype(np.uint8)

def enhance(u8):
    try:
        import cv2
        den=cv2.bilateralFilter(u8,5,40,5)
        e=cv2.createCLAHE(clipLimit=1.5,tileGridSize=(8,8)).apply(den)
        b=cv2.GaussianBlur(e,(0,0),1.0)
        return cv2.addWeighted(e,1.25,b,-0.25,0)
    except Exception:
        from PIL import Image,ImageOps,ImageFilter
        im=Image.fromarray(u8)
        im=ImageOps.autocontrast(im,cutoff=1)
        im=im.filter(ImageFilter.UnsharpMask(radius=1.2,percent=80,threshold=2))
        return np.array(im)

def crop_hu(hu,cx,cy,size=150):
    cx,cy=int(round(cx)),int(round(cy)); h=size//2; R,C=hu.shape
    x0=max(0,min(cx-h,C-size)); y0=max(0,min(cy-h,R-size))
    return hu[y0:y0+size,x0:x0+size]

def crop_save_raw(ct_path,out_path,cx,cy,size=150):
    raw=bytearray(open(ct_path,'rb').read())
    R=_u16(bytes(raw),TAG_ROWS) or 512; C=_u16(bytes(raw),TAG_COLS) or 512
    p=bytes(raw).rfind(TAG_PIX); vr=raw[p+4:p+6]
    st=p+12 if bytes(vr) in (b'OB',b'OW',b'OF',b'UN') else p+8
    arr=np.frombuffer(bytes(raw[st:st+R*C*2]),dtype=np.int16).reshape(R,C)
    cx,cy=int(round(cx)),int(round(cy)); h=size//2
    x0=max(0,min(cx-h,C-size)); y0=max(0,min(cy-h,R-size))
    crop=arr[y0:y0+size,x0:x0+size].copy()
    for tag in (TAG_ROWS,TAG_COLS):
        idx=bytes(raw).find(tag)
        if idx>=0: raw[idx+8:idx+10]=struct.pack('<H',size)
    np_=crop.tobytes(); raw[st-4:st]=struct.pack('<I',len(np_))
    os.makedirs(os.path.dirname(out_path),exist_ok=True)
    open(out_path,'wb').write(bytes(raw[:st])+np_)

# ---------------- core check ----------------
def check_nodule(ct, x, y, image, ntype='any', out=None, zwin=3, variant='original', size=150,
                 slice_contrast_min=200, nodule_contrast_min=250, offset_max=15):
    index,_=build_index(ct)
    if not index: return {'verdict':'ERROR','reason':'no CT series found','ct':ct}
    N=len(index)
    cand=[]
    for inst in range(image-zwin, image+zwin+1):
        fp,z=file_for_image(index,inst)
        if not fp: continue
        hu=read_hu(fp); lung=lung_ref(hu)
        vlist=variants(x,y).items() if variant=='auto' else [(variant,variants(x,y)[variant])]
        for vname,(vx,vy) in vlist:
            rx,ry=refine_center(hu,vx,vy,lung)
            s=blob_score(hu,rx,ry,lung)
            if s>0:
                cand.append((s,inst,z,vname,rx,ry,hu,lung))
    if not cand:
        return {'verdict':'DROP','reason':'not apparent (no nodule-like center near annotation)',
                'ct':os.path.basename(ct)}
    cand.sort(key=lambda t:t[0],reverse=True)
    s,inst,z,vname,rx,ry,hu,lung=cand[0]
    chu=center_hu(hu,rx,ry); contrast=chu-lung
    maxc=max(center_hu(h_,cx_,cy_)-l_ for _,_,_,_,cx_,cy_,h_,l_ in cand)
    offset=centroid_offset(hu,rx,ry,lung)
    tdet=classify_type(chu)
    res={'ct':os.path.basename(ct),'best_slice':inst,'z':round(z,1) if z is not None else None,
         'variant':vname,'center_x':round(rx,1),'center_y':round(ry,1),'center_hu':round(chu),
         'contrast':round(contrast),'max_contrast':round(maxc),'offset_px':round(offset,1),
         'type_detected':tdet}
    # gates
    if maxc<nodule_contrast_min or contrast<slice_contrast_min:
        res.update(verdict='DROP',reason=f'not apparent (contrast {round(contrast)} < {slice_contrast_min} / max {round(maxc)} < {nodule_contrast_min})')
        return res
    if offset>offset_max:
        res.update(verdict='DROP',reason=f'not centered (offset {round(offset,1)} > {offset_max}px)')
        return res
    if ntype not in ('any',None):
        if tdet==ntype: tflag='match'
        elif tdet in ADJ.get(ntype,[]): tflag='adjacent-warn'
        else:
            res.update(verdict='DROP',reason=f'type mismatch (wanted {ntype}, got {tdet})')
            return res
        res['type_flag']=tflag
    res.update(verdict='PASS',reason='apparent, centered, valid slice'+(f', type {res.get("type_flag","")}' if ntype not in('any',None) else ''))
    # write outputs
    if out:
        case=os.path.basename(ct.rstrip('/')); base=os.path.join(out,case)
        stem=f"x{int(x)}_y{int(y)}_img{image}_s{inst}_{tdet}"
        from PIL import Image
        enh=enhance(lung_u8(crop_hu(hu,rx,ry,size)))
        os.makedirs(os.path.join(base,'png_enhanced'),exist_ok=True)
        Image.fromarray(enh).save(os.path.join(base,'png_enhanced',stem+'.png'))
        Image.fromarray(lung_u8(crop_hu(hu,rx,ry,size))).save(os.path.join(base,'png_enhanced',stem+'_plain.png'))
        fp,_=file_for_image(index,inst)
        crop_save_raw(fp,os.path.join(base,stem+'.dcm'),rx,ry,size)
        res['out']=os.path.join(base,'png_enhanced',stem+'.png')
    return res

def _append_report(out,res):
    if not out: return
    os.makedirs(out,exist_ok=True); fp=os.path.join(out,'qc_report.csv')
    cols=['ct','verdict','reason','type_detected','contrast','max_contrast','center_hu',
          'offset_px','best_slice','z','variant','center_x','center_y','out']
    new=not os.path.exists(fp)
    with open(fp,'a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore')
        if new: w.writeheader()
        w.writerow(res)

def main():
    ap=argparse.ArgumentParser(description='Nodule apparent-QC + enhance')
    ap.add_argument('--ct'); ap.add_argument('--x',type=float); ap.add_argument('--y',type=float)
    ap.add_argument('--image',type=int); ap.add_argument('--type',default='any')
    ap.add_argument('--out'); ap.add_argument('--batch')
    ap.add_argument('--zwin',type=int,default=3); ap.add_argument('--variant',default='original')
    ap.add_argument('--size',type=int,default=150)
    a=ap.parse_args()
    if a.batch:
        for row in csv.DictReader(open(a.batch)):
            r=check_nodule(row['ct_path'],float(row['x']),float(row['y']),int(row['image']),
                           row.get('type','any'),a.out,a.zwin,a.variant,a.size)
            _append_report(a.out,r); print(json.dumps(r))
    else:
        r=check_nodule(a.ct,a.x,a.y,a.image,a.type,a.out,a.zwin,a.variant,a.size)
        _append_report(a.out,r); print(json.dumps(r,indent=2))

if __name__=='__main__':
    main()
