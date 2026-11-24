# Data staging

MRI data are not tracked in Git. The local ATLAS3 data at `/deneb_disk/ATLAS3/ATLAS3_Training_Preprocessed` can supply nonlesional target patches and donor masks. Run `scripts/build_atlas3_training_manifests.py` to pair T1/mask files, reserve whole sites for holdout, and write local manifests under `/deneb_disk/data/stroke_inpainting/manifests/`.

## Candidate sources

- **ATLAS v2.0** contains chronic stroke T1 scans and manually drawn lesion masks; access is through ICPSR with data-use requirements. Masks can inform or supply registered synthetic mask shapes, but cross-sectional lesion scans do not include lesion-free paired targets.
- **ATLAS v3.0** has a newer, larger release path, but access requires accepting project terms and requesting an access code. Do not automate around that gate.
- **OASIS / IXI or local healthy control cohorts** may be suitable healthy-image targets, subject to their current access and use terms. Deface before any public transfer and check each dataset's license.

The training CLI accepts an `--manifest` with `healthy_t1` and `exclude_mask` columns plus a separate `--mask-manifest` with one `lesion_mask` column. For ATLAS3 targets, the loader rejects patches overlapping the matched source lesion or a conservative surrounding margin. It then transfers donor mask morphology into the nonlesional patch. Donor placement is randomized, not subject-specific anatomical registration. Keep manifests local and do not commit subject data or identifying paths.

Record dataset version, source URL, access date, license/DUA, subject identifiers, and preprocessing provenance in a local `DATA_PROVENANCE.md`. Never commit scans, masks, manifests with identifying paths, or credentials.

## Suggested layout

```text
/deneb_disk/data/stroke_inpainting/
  healthy_t1/       # authorized, defaced T1 NIfTI files
  masks_atlas/      # authorized masks, optional
  manifests/        # train/validation/test CSVs
  checkpoints/      # model weights
```

Use subject-level and preferably site-level splits. Keep a final test set untouched until model selection is finished.
