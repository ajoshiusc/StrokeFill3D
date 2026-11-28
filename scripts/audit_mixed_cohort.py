"""Audit every prepared subject and freeze a hash-checked training-ready cohort.

Run after prepare_mixed_cohort.py. Rejected scans are recorded, never silently
accepted as normal targets. Generates one anatomical QC image per external scan.
"""
import argparse
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path


def audit(row, root):
    import nibabel as nib
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from scipy.ndimage import binary_erosion, distance_transform_edt
    from strokefill3d.mixed_rf.data import CleanDomain
    torch.set_num_threads(1)
    ds=object.__new__(CleanDomain)
    ds.rows=[row];ds.size=np.array([96]*3);ds.seed=2026;ds.margin=5;ds.cache_size=1;ds.cache=OrderedDict()
    try:
        patches=[ds[i] for i in range(4)]
        a,_,_,_,_=ds.load(0)
        lesion=np.asarray(nib.load(row['exclude_mask']).dataobj)>0
        excluded=distance_transform_edt(~lesion)<=5
        for c in patches:
            start=c['starts'].numpy(); sl=tuple(slice(int(s),int(s+96)) for s in start)
            if excluded[sl].any():raise ValueError('Lesion contamination in clean target')
        if row['domain']!='atlas':
            views=[]
            edge=lesion & ~binary_erosion(lesion)
            ys=np.argwhere(lesion)[:,1] if lesion.any() else np.array([116])
            for y in np.clip(np.quantile(ys,[.25,.5,.75]).astype(int),0,a.shape[1]-1):
                gray=np.uint8(np.clip((a[:,y,:]+1)*127.5,0,255)); rgb=np.repeat(gray[:,:,None],3,2)
                rgb[edge[:,y,:]]=[255,45,45]
                views.append(Image.fromarray(np.rot90(rgb)).resize((256,256)))
            canvas=Image.new('RGB',(768,560),'white');d=ImageDraw.Draw(canvas)
            for i,im in enumerate(views):canvas.paste(im,(i*256,24))
            for i,c in enumerate(patches[:3]):
                im=np.uint8(np.clip((c['target'][0,:,48,:].numpy()+1)*127.5,0,255))
                canvas.paste(Image.fromarray(np.rot90(im)).convert('RGB').resize((256,256)),(256*i,300))
            d.text((4,4),row['subject']+' | injury boundaries (red)',fill='black');d.text((4,282),'Four clean 96-cube target samples checked; three center slices shown',fill='black')
            qc=Path(root)/'qc';qc.mkdir(exist_ok=True);canvas.save(qc/(row['subject'].replace(':','_')+'.png'))
        return dict(subject=row['subject'],ok=True)
    except Exception as e:return dict(subject=row['subject'],ok=False,error=str(e))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    root=a.data
    meta=json.loads((root/'dataset.json').read_text())
    if meta.get('audit_complete'):
        from strokefill3d.mixed_rf.data import cohort_summary
        print('READY', json.dumps(cohort_summary(root)), flush=True)
        return
    groups={s:list(csv.DictReader((root/(s+'.csv')).open())) for s in ['train','val','test']}
    results=[]
    with ProcessPoolExecutor(a.workers) as pool:
        jobs=[pool.submit(audit,r,str(root)) for rs in groups.values() for r in rs]
        for i,j in enumerate(as_completed(jobs),1):
            result=j.result();results.append(result)
            with (root/'audit_progress.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
            if i%20==0 or not result['ok']: print(i,result,flush=True)
    failed={r['subject'] for r in results if not r['ok']}
    (root/'audit_failures.json').write_text(json.dumps([r for r in results if not r['ok']],indent=2))
    for split,rows in groups.items():
        rows=[r for r in rows if r['subject'] not in failed];groups[split]=rows
        path=root/(split+'.csv');tmp=path.with_suffix('.tmp')
        with tmp.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        tmp.replace(path)
    from strokefill3d.mixed_rf.data import validate_splits,cohort_summary
    validate_splits(*(root/(s+'.csv') for s in groups))
    meta.update(audit_complete=True,audited_subjects=len(results),audit_rejected=len(failed),counts={s:dict(Counter(r['domain'] for r in rs)) for s,rs in groups.items()},manifest_sha256={s:hashlib.sha256((root/(s+'.csv')).read_bytes()).hexdigest() for s in groups})
    # Validate counts before publishing readiness.
    for domain in ['atlas','episurg','epibios','arc']:
        if meta['counts']['train'].get(domain,0)<10 or min(meta['counts'][s].get(domain,0) for s in ['val','test'])<2:raise ValueError(f'Insufficient audited subjects in {domain}')
    tmp=root/'dataset.tmp';tmp.write_text(json.dumps(meta,indent=2));tmp.replace(root/'dataset.json')
    print('READY',json.dumps(cohort_summary(root)),flush=True)


if __name__=='__main__':main()
