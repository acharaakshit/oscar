## Overview
- Data root: set `PREFIX=/path/to/datasets`. All datasets live under `PREFIX/<DATASET>_<FS>/` (FS = field strength, e.g., `3T` or `1.5T`).
- Processed NIfTI volumes are written to `PREFIX/<DATASET>_<FS>/processed/` by `process/create_data_objects.py`.
- Metadata CSVs are written to `PREFIX/METADATA/<DATASET>_ATTRIBUTES.csv` by `process/metadata.py` and are consumed by training/eval scripts.
- Folder names come from `config/folder.yaml` (`processed`, `preprocessed`, `metadata`); adjust there if your layout differs.

## Common steps
1) Place preprocessed brain-extracted NIfTIs under `PREFIX/<DATASET>_<FS>/<preprocessed>/` (defaults to `preprocessed/`).
2) Ensure raw metadata files listed below are present under each dataset folder (e.g., `PREFIX/ADNI_3T/WORKING.csv`).
3) Run metadata aggregation:
```
PYTHONPATH=$PROJECTDIR python -u process/metadata.py --dataset <DATASET>
```
4) Create processed, padded Niftis:
```
PYTHONPATH=$PROJECTDIR python -u process/create_data_objects.py --dataset <DATASET> --FS 3T
```
Repeat for `--FS 1.5T` where applicable.

## Dataset-specific notes

### ADNI (multifield: 3T, 1.5T)
- Preprocessed scans: `PREFIX/ADNI_<FS>/preprocessed/**/<IMAGE_ID>_MNI_Brain.nii.gz`.
- Required metadata: `WORKING.csv`, `race.csv` (from LONI), field-strength CSV (`ADNI_<FS>.csv`) under `PREFIX/ADNI_<FS>/`.
- `process/metadata.py` filters to White/Black, maps `gender/race/age/disease_group`, and enforces field strength per scan.

### OASIS (multifield: 3T, 1.5T)
- Preprocessed scans + JSON sidecars with `MagneticFieldStrength` in `PREFIX/OASIS_<FS>/preprocessed/` (filenames like `<SUBJECT>_MNI_Brain.nii.gz`).
- Metadata: `metadata.csv`, `cognition.csv`, `healthy.csv` under `PREFIX/OASIS_<FS>/`.
- Filters to stable cognitive status, uses JSON to keep matching field strength.

### IXI (multifield: 3T, 1.5T)
- Preprocessed scans `PREFIX/IXI_<FS>/preprocessed/*Brain.nii.gz`.
- Metadata: `IXI.xls` under `PREFIX/IXI_<FS>/` (gender/race/age). Keeps White/Asian subjects.

### HCP (3T)
- Raw structure: `PREFIX/HCP/<SUBJECT>/T1w/T1w_acpc_dc_restore_brain.nii.gz`.
- Metadata: `metadata.csv` (public) and `metadata_restricted.csv` (restricted) under `PREFIX/HCP_3T/`.
- Keeps White/Black subjects; uses restricted file for age and race.

### A4 (3T)
- Preprocessed scans + JSON sidecars in `PREFIX/A4_3T/preprocessed/` (filenames include visit code, end with `MNI_Brain.nii.gz`).
- Metadata: `metadata.csv` and `visits_datadic.csv` under `PREFIX/A4_3T/` for age adjustments across visits.
- Field strength validated from JSON; visit timing added to age when available.

### UKBB (3T)
- Preprocessed scans `PREFIX/UKBB_3T/preprocessed/<SUBJECT>_Brain.nii.gz`.
- Metadata: `FILTERED.csv` from `process/ukbb_filter.py` under `PREFIX/UKBB_3T/` (contains subject/gender/race/age for allowed subjects).
- Keeps specified ethnicity codes (White/Black/Asian/Chinese); uses N4 bias correction.

### ABIDE (3T)
- Preprocessed scans `PREFIX/ABIDE_3T/preprocessed/<IMAGE_ID>_Brain.nii.gz`.
- Metadata: `metadata.csv` under `PREFIX/ABIDE_3T/` (includes `Image Data ID`, `Subject`, `Sex`, `Age`, `Group`).
- Maps Group to `CN` vs `AD` label for consistency with other scripts.

### 2D datasets (CelebA, CheXpert)
- Place standard image folders/CSVs under `PREFIX/<dataset>/` following the original dataset structure. Training scripts expect dataset-specific loaders defined in `src/data.py`; ensure paths there point to your layout.

## Sanity checks
- After `process/metadata.py`, inspect `PREFIX/METADATA/<DATASET>_ATTRIBUTES.csv` for expected columns (e.g., `image_id,subject_id,scan,gender,race,age[,disease_group,...]`).
- `process/create_data_objects.py` skips already-processed scans to avoid reprocessing.
- If scans are skipped, check field-strength mismatches or missing JSON/metadata entries as printed by the scripts.

Note: These dataset steps correspond to the MRI experiments from “Invisible attributes, visible biases: Exploring demographic shortcuts in MRI-based Alzheimer’s disease classification” (MICCAI FAIMI 2025).
