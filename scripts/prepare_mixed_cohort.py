"""Discover every labeled external participant, prepare in Python, freeze splits.

No subject cap. Missing images, missing injury labels, and failed QC are reported.
A manifest is published only after all discovered participants are processed.
"""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import traceback

FIELDS = ['healthy_t1', 'exclude_mask', 'brain_mask', 'subject', 'domain', 'site']


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def discover(args):
    cases, excluded = [], []
    for folder in sorted((args.episurg / 'subjects').glob('sub-*')):
        t1s = sorted((folder / 'postop').glob('*t1mri*.nii.gz'))
        masks = sorted((folder / 'postop').glob('*seg*.nii.gz'))
        if not t1s or not masks:
            excluded.append(dict(domain='episurg', subject=folder.name, reason='missing_postop_t1_or_injury_label'))
            continue
        cases.append(dict(domain='episurg', subject=folder.name, image=str(t1s[0]), labels=list(map(str, masks)), label_image=str(t1s[0]), provenance='union_of_supplied_resection_raters'))
    # SecondReview has priority over earlier reviews; group repeated scans by person.
    masks = sorted((args.epibios / 'lesion_segmentations_box').rglob('*SecondReview.nii.gz'))
    grouped = collections.defaultdict(list)
    for mask in masks:
        scan = mask.name.split('_segmentation')[0]
        person = '_'.join(scan.split('_')[:3])
        t1 = args.epibios / 't1mprage' / (scan + '_t1mprage.nii.gz')
        flair = args.epibios / 't2flair' / (scan + '_t2flair.nii.gz')
        if t1.exists() and flair.exists():
            grouped[person].append(dict(domain='epibios', subject=person, image=str(t1), labels=[str(mask)], label_image=str(flair), provenance='supplied_second_review_FLAIR_labels'))
        else:
            excluded.append(dict(domain='epibios', subject=person, reason='missing_same_scan_t1_or_flair', mask=str(mask)))
    for rows in grouped.values():
        cases.append(sorted(rows, key=lambda r:r['image'])[0])
    labeled = set(grouped)
    for t1 in sorted((args.epibios / 't1mprage').glob('*.nii.gz')):
        person = '_'.join(t1.name.split('_')[:3])
        if person not in labeled:
            excluded.append(dict(domain='epibios', subject=person, reason='no_paired_second_review_injury_label'))
    grouped = collections.defaultdict(list)
    for mask in sorted((args.arc / 'derivatives/lesion_masks').rglob('*desc-lesion_mask.nii.gz')):
        sub, session = mask.relative_to(args.arc / 'derivatives/lesion_masks').parts[:2]
        meta = json.loads(mask.with_name(mask.name.replace('.nii.gz','.json')).read_text())
        source = args.arc / meta['SpatialReference'].removeprefix('bids:raw:')
        t1s = sorted((args.arc / sub / session / 'anat').glob('*T1w.nii.gz'))
        if source.exists() and t1s:
            grouped[sub].append(dict(domain='arc', subject=sub, image=str(t1s[0]), labels=[str(mask)], label_image=str(source), provenance='supplied_ARC_manual_lesion_mask'))
        else:
            excluded.append(dict(domain='arc', subject=sub, reason='missing_same_session_t1_or_label_reference'))
    for rows in grouped.values():
        cases.append(sorted(rows, key=lambda r:r['image'])[0])
    for folder in sorted(args.arc.glob('sub-*')):
        if folder.name not in grouped:
            excluded.append(dict(domain='arc', subject=folder.name, reason='no_paired_manual_label'))
    return cases, excluded


def align_labels(fixed, moving, masks, threads):
    """Rigid image registration with MI; never align a label using anatomy headers alone."""
    import nibabel as nib
    import numpy as np
    from strokefill3d.normal_prior.medical import to_sitk, resample, sitk_module
    from strokefill3d.normal_prior.preprocess import aligned
    sitk = sitk_module()
    union = np.zeros(moving.shape, bool)
    for path in masks:
        m = nib.as_closest_canonical(nib.load(path)); aligned(moving, m)
        union |= np.asarray(m.dataobj) > 0
    label = nib.Nifti1Image(union.astype(np.uint8), moving.affine)
    if fixed.shape == moving.shape and np.allclose(fixed.affine, moving.affine, atol=1e-4) and masks and str(moving.get_filename()) == str(fixed.get_filename()):
        return union, None, {}
    f = to_sitk(fixed, np.asarray(fixed.dataobj, np.float32))
    m = to_sitk(moving, np.asarray(moving.dataobj, np.float32))
    transform = sitk.CenteredTransformInitializer(f, m, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    reg = sitk.ImageRegistrationMethod(); reg.SetNumberOfThreads(threads)
    reg.SetMetricAsMattesMutualInformation(48); reg.SetMetricSamplingStrategy(reg.REGULAR)
    reg.SetMetricSamplingPercentage(.15, 2026); reg.SetInterpolator(sitk.sitkLinear)
    reg.SetShrinkFactorsPerLevel([4,2,1]); reg.SetSmoothingSigmasPerLevel([2,1,0]); reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetOptimizerAsRegularStepGradientDescent(2., .001, 200)
    reg.SetOptimizerScalesFromPhysicalShift(); reg.SetInitialTransform(transform, inPlace=True)
    reg.Execute(f,m)
    result = resample(label, fixed, transform, True) > 0
    ratio = result.sum() * abs(np.linalg.det(fixed.affine[:3,:3])) / max(1., union.sum()*abs(np.linalg.det(moving.affine[:3,:3])))
    if union.any() and not .6 < ratio < 1.5:
        raise ValueError(f'Cross-modality label volume retention failed: {ratio:.3f}')
    return result, transform, dict(metric=reg.GetMetricValue(), stop=reg.GetOptimizerStopConditionDescription(), volume_retention=ratio)


def prepare_case(case, output, threads, weights, template, template_mask):
    import contextlib
    import nibabel as nib
    import numpy as np
    import torch
    from nibabel.processing import resample_to_output
    from scipy.ndimage import binary_dilation
    from strokefill3d.normal_prior.preprocess import canonical, save
    from strokefill3d.normal_prior.medical import sitk_module, n4_correct, register_affine, resample
    from strokefill3d.normal_prior.skullstrip import extract_brain
    folder = Path(output) / 'volumes' / (case['domain'] + '_' + case['subject'])
    folder.mkdir(parents=True, exist_ok=True)
    try:
        if (folder/'prepared.json').exists():
            previous = json.loads((folder/'prepared.json').read_text())
            if previous['source'] != case:
                raise ValueError('Cached source identity changed')
            return previous
        with (folder/'prepare.log').open('a', buffering=1) as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            sitk = sitk_module(); sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(threads); torch.set_num_threads(threads)
            native = canonical(case['image'])
            # ITK requires orthogonal voxel axes. Preserve physical coordinates when removing shear.
            direction = native.affine[:3,:3] / nib.affines.voxel_sizes(native.affine)
            if not np.allclose(direction.T@direction,np.eye(3),atol=1e-4):
                native = resample_to_output(native, nib.affines.voxel_sizes(native.affine), order=1)
            if case['label_image'] == case['image']:
                from nibabel.processing import resample_from_to
                lesion = np.zeros(native.shape,bool)
                for path in case['labels']:
                    label = canonical(path)
                    original = canonical(case['image'])
                    if label.shape != original.shape or not np.allclose(label.affine,original.affine,atol=1e-4):
                        raise ValueError('Supplied T1 label grid mismatch')
                    lesion |= np.asarray(resample_from_to(label,native,order=0).dataobj)>0
                report = {'method':'supplied_T1_grid_union'}
            else:
                lesion, tx, report = align_labels(native,canonical(case['label_image']),case['labels'],threads)
                if tx: sitk.WriteTransform(tx,str(folder/'t1_to_label_source.tfm'))
            save(lesion,native,folder/'lesion_native.nii.gz',np.uint8)
            # Full precision is required: half precision can invalidate SynthStrip masks.
            brain, extraction = extract_brain(native,weights,None,'cpu',threads)
            print('Brain extraction complete',flush=True)
            observed = brain & ~binary_dilation(lesion,iterations=5) & (np.asarray(native.dataobj)>0)
            corrected, _, n4report = n4_correct(native,observed,threads)
            fixed, bm = canonical(template), canonical(template_mask)
            fixedbrain = np.asarray(bm.dataobj)>0
            moving = nib.Nifti1Image(corrected,native.affine)
            tx, stages = register_affine(fixed,moving,fixedbrain,brain,observed,threads)
            singular = np.linalg.svd(np.asarray(tx.GetMatrix()).reshape(3,3),compute_uv=False)
            if singular.min()<.5 or singular.max()>2 or np.linalg.det(np.asarray(tx.GetMatrix()).reshape(3,3))<=0:
                raise ValueError('Invalid affine scales/reflection')
            a = resample(moving,fixed,tx)
            b = resample(nib.Nifti1Image(brain.astype(np.uint8),native.affine),fixed,tx,True)>0
            l = resample(nib.Nifti1Image(lesion.astype(np.uint8),native.affine),fixed,tx,True)>0
            dice = 2*(b&fixedbrain).sum()/max(1,b.sum()+fixedbrain.sum())
            inside = (l&b).sum()/max(1,l.sum())
            if dice<.8 or (l.any() and inside<.8):
                raise ValueError(f'Registration QC failed: brain Dice={dice:.3f}, injury inside brain={inside:.3f}')
            save(np.where(b,a,0),fixed,folder/'t1_brain_training.nii.gz')
            save(b,fixed,folder/'brain_mask_training.nii.gz',np.uint8)
            save(l,fixed,folder/'lesion_training.nii.gz',np.uint8)
            sitk.WriteTransform(tx,str(folder/'training_to_native.tfm'))
            row = dict(healthy_t1=str(folder/'t1_brain_training.nii.gz'),exclude_mask=str(folder/'lesion_training.nii.gz'),brain_mask=str(folder/'brain_mask_training.nii.gz'),subject=case['domain']+':'+case['subject'],domain=case['domain'],site=case['domain'])
            result = dict(source=case,row=row,qc=dict(brain_dice=float(dice),injury_inside_brain=float(inside)),label_registration=report,brain_extraction=extraction,n4=n4report,registration=stages)
            write_json(folder/'prepared.json',result)
            return result
    except Exception:
        return dict(source=case,error=traceback.format_exc())


def publish(args, results, excluded):
    # Keep original ATLAS splits unchanged. All external participants have a stable hash split.
    groups={}
    for split,name in [('train','atlas3_train.csv'),('val','anatomical_v1/atlas3_val.csv'),('test','anatomical_v1/atlas3_test.csv')]:
        rows=list(csv.DictReader((args.atlas_manifests/name).open()))
        groups[split]=[dict(healthy_t1=r['healthy_t1'],exclude_mask=r['exclude_mask'],brain_mask='',subject='atlas:'+Path(r['healthy_t1']).parent.parent.parent.name,domain='atlas',site=r.get('site','')) for r in rows]
    for domain in ['episurg','epibios','arc']:
        rows=[r['row'] for r in results if 'error' not in r and r['row']['domain']==domain]
        rows.sort(key=lambda r:hashlib.sha256(('20261128:'+r['subject']).encode()).hexdigest())
        if len(rows)<10: raise ValueError(f'Only {len(rows)} eligible {domain} subjects; refusing long-run manifest')
        n=max(2,round(len(rows)*.1))
        for i,row in enumerate(rows):
            groups['val' if i<n else 'test' if i<2*n else 'train'].append(row)
    for split,rows in groups.items():
        with (args.output/(split+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    from strokefill3d.mixed_rf.data import validate_splits
    validate_splits(*(args.output/(s+'.csv') for s in ['train','val','test']))
    summary=dict(version=2,complete=True,counts={s:dict(collections.Counter(r['domain'] for r in rs)) for s,rs in groups.items()},failed=sum('error'in r for r in results),exclusion_records=len(excluded),split='subject-disjoint 80/10/10 per external cohort; existing ATLAS split',notes='All discovered labeled participants attempted. One same-session scan per person. EPISURG masks union all raters. ARC supplied manual masks, not predictions. No generated MRI targets. Cross-modality mask registration requires visual QC; automatic QC is not a manual review.')
    write_json(args.output/'dataset.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episurg',type=Path,default=Path('/deneb_disk/auto_resection/EPISURG/EPISURG'))
    p.add_argument('--epibios',type=Path,default=Path('/deneb_disk/ifs/loni/faculty/dduncan/rgarner/shared/epibios/human/data'))
    p.add_argument('--arc',type=Path,default=Path('/deneb_disk/ARC'))
    p.add_argument('--atlas-manifests',type=Path,default=Path('/deneb_disk/data/stroke_inpainting/manifests'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=6);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--synthstrip-weights',required=True);p.add_argument('--template',required=True);p.add_argument('--template-mask',required=True)
    p.add_argument('--inventory-only',action='store_true')
    args=p.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'dataset.json').exists(): raise ValueError('Published data is immutable; choose a new output directory')
    cases,excluded=discover(args)
    # Interleave cohorts so cross-modality errors surface early.
    buckets = [[c for c in cases if c['domain']==d] for d in ['episurg','epibios','arc']]
    cases = [bucket[i] for i in range(max(map(len,buckets))) for bucket in buckets if i<len(bucket)]
    write_json(args.output/'inventory.json',dict(cases=cases,excluded=excluded))
    print('Discovered:',dict(collections.Counter(c['domain'] for c in cases)),flush=True)
    if args.inventory_only:return
    results=[]
    with ProcessPoolExecutor(args.workers) as pool:
        jobs=[pool.submit(prepare_case,c,str(args.output),args.threads,args.synthstrip_weights,args.template,args.template_mask) for c in cases]
        for job in as_completed(jobs):
            result=job.result();results.append(result)
            with (args.output/'progress.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
            print(f"{len(results)}/{len(cases)} {result['source']['domain']} {result['source']['subject']} {'FAILED' if 'error' in result else 'OK'}",flush=True)
    write_json(args.output/'failures.json',[r for r in results if 'error'in r])
    publish(args,results,excluded)


if __name__=='__main__':main()
