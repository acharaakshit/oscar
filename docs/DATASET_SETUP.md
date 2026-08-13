## Overview
- Data root: set `PREFIX=/path/to/datasets`. Most MRI datasets live under `PREFIX/<DATASET>_<FS>/` (FS = field strength, e.g., `3T` or `1.5T`).
- Processed NIfTI volumes are written to `PREFIX/<DATASET>_<FS>/processed/` by `process/create_data_objects.py`.
- Metadata CSVs are written to `PREFIX/METADATA/<DATASET>_ATTRIBUTES.csv` by `process/metadata.py` and are consumed by training/eval scripts.
- Folder names come from `config/folder.yaml` (`processed`, `preprocessed`, `metadata`); adjust there if your layout differs.
- HCP is the main exception: raw scans are read from `PREFIX/HCP/<SUBJECT>/...`, while metadata is expected under `PREFIX/HCP_3T/`.

## Common steps
1) Place preprocessed brain-extracted NIfTIs under `PREFIX/<DATASET>_<FS>/<preprocessed>/` (defaults to `preprocessed/`).
2) Ensure the dataset-specific metadata files described below are present under each dataset folder.
3) Run metadata aggregation:
```
PYTHONPATH=$PROJECTDIR python -u process/metadata.py --dataset <DATASET>
```
4) Create processed Niftis:
```
PYTHONPATH=$PROJECTDIR python -u process/create_data_objects.py --dataset <DATASET> --FS 3T
```
Repeat for `--FS 1.5T` where applicable.

## 3D Dataset-Specific Notes

### [ADNI](https://ida.loni.usc.edu/home/projectPage.jsp?project=ADNI) (multifield: 3T, 1.5T)
- Preprocessed scans: `PREFIX/ADNI_<FS>/preprocessed/**/<IMAGE_ID>_MNI_Brain.nii.gz`.
- Required metadata: the main ADNI report CSV, `race.csv` (from LONI), and the field-strength CSV (`ADNI_<FS>.csv`) under `PREFIX/ADNI_<FS>/`.
- `process/metadata.py` filters to White/Black, maps `gender/race/age/disease_group`, and enforces field strength per scan.

### [OASIS](https://sites.wustl.edu/oasisbrains/home/oasis-3/) (multifield: 3T, 1.5T)
- Preprocessed scans + JSON with `MagneticFieldStrength` in `PREFIX/OASIS_<FS>/preprocessed/` (filenames like `<SUBJECT>_MNI_Brain.nii.gz`).
- Metadata: `metadata.csv`, `cognition.csv`, `healthy.csv` under `PREFIX/OASIS_<FS>/`.
- Filters to stable cognitive status, uses JSON to keep matching field strength.

### [IXI](https://brain-development.org/ixi-dataset/) (multifield: 3T, 1.5T)
- Preprocessed scans `PREFIX/IXI_<FS>/preprocessed/*Brain.nii.gz`.
- Metadata: `IXI.xls` under `PREFIX/IXI_<FS>/` (gender/race/age). Keeps White/Asian subjects.

### [HCP](https://www.humanconnectome.org/study/hcp-young-adult/data-releases) (3T)
- Raw structure: `PREFIX/HCP/<SUBJECT>/T1w/T1w_acpc_dc_restore_brain.nii.gz`.
- Metadata: `metadata.csv` (public) and `metadata_restricted.csv` (restricted) under `PREFIX/HCP_3T/`.
- Keeps White/Black subjects; uses restricted file for age and race.

### [A4](https://www.a4studydata.org/) (3T)
- Preprocessed scans + JSON in `PREFIX/A4_3T/preprocessed/` (filenames include visit code, end with `MNI_Brain.nii.gz`).
- Metadata: `metadata.csv` and `visits_datadic.csv` under `PREFIX/A4_3T/` for age adjustments across visits.
- Field strength validated from JSON; visit timing added to age when available.

### [UKBB](https://www.ukbiobank.ac.uk/) (3T)
- Preprocessed scans `PREFIX/UKBB_3T/preprocessed/<SUBJECT>_Brain.nii.gz`.
- Metadata: `FILTERED.csv` from `process/ukbb_filter.py` under `PREFIX/UKBB_3T/` (contains subject/gender/race/age for allowed subjects).
- Keeps specified ethnicity codes (White/Black/Asian/Chinese); uses N4 bias correction.

### [ABIDE](https://ida.loni.usc.edu/home/projectPage.jsp?project=ABIDE) (3T)
- Preprocessed scans `PREFIX/ABIDE_3T/preprocessed/<IMAGE_ID>_Brain.nii.gz`.
- Metadata: `metadata.csv` under `PREFIX/ABIDE_3T/` (includes `Image Data ID`, `Subject`, `Sex`, `Age`, `Group`).
- Maps Group to `CN` vs `AD` label for consistency with other scripts.

## 2D Dataset Downloads

### CelebA
CelebA is downloaded through `torchvision.datasets.CelebA(..., download=True)`.
Set `PREFIX` as the dataset root before running the 2D scripts.

### CheXpert
Go to Stanford AIMI, get the CheXpert dataset URL and token, then download with AzCopy:
```
azcopy copy '<CHEXPERT_DATASET_URL_WITH_TOKEN>' \
  "$PREFIX/chexpert/data/" \
  --recursive
```

### Waterbirds
Waterbirds is generated from CUB-200-2011 and Places365.
Download [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) and [Places365](http://places2.csail.mit.edu/download.html).
Generate the dataset with:
```
python process/generate_waterbirds_balanced_pool.py \
  --cub-dir "$PREFIX/CUB/CUB_200_2011" \
  --places-dir "$PREFIX/CUB/places365" \
  --output-dir "$PREFIX/CUB/waterbirds/waterbird_stratified_pool_forest2water2"
```

### [ISIC 2019](https://challenge.isic-archive.com/data/#2019)
Download the ISIC 2019 images, ground-truth CSVs, and metadata CSVs.

### [Camelyon17](https://wilds.stanford.edu/datasets/#camelyon17)
Download Camelyon17 from the WILDS benchmark distribution.

## Sanity checks
- After `process/metadata.py`, inspect `PREFIX/METADATA/<DATASET>_ATTRIBUTES.csv` for expected columns (e.g., `image_id,subject_id,scan,gender,race,age[,disease_group,...]`).
- `process/create_data_objects.py` skips already-processed scans to avoid reprocessing.
- If scans are skipped, check field-strength mismatches or missing JSON/metadata entries as printed by the scripts.

Note: The 3D dataset steps correspond to the MRI experiments from “Invisible attributes, visible biases: Exploring demographic shortcuts in MRI-based Alzheimer’s disease classification” (MICCAI FAIMI 2025).
