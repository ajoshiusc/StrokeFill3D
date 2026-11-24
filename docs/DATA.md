# Data staging

No MRI data is included. Keep scans outside Git in `/deneb_disk/data/stroke_inpainting/` and maintain a `healthy_t1` CSV manifest.

## Candidate sources

- **ATLAS v2.0** contains chronic stroke T1 scans and manually drawn lesion masks; public release access is through ICPSR with data-use requirements. The masks can inform synthetic mask shape sampling, but there is no lesion-free paired target in the ordinary cross-sectional data.
- **ATLAS v3.0** has a newer larger release path, but access requires agreeing to the project terms and requesting the encryption code. Do not automate around that gate.
- **OASIS / IXI or local healthy control cohorts** may be suitable healthy-image targets, subject to their current access and use terms. Deface before any public transfer and check the dataset-specific license.

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
