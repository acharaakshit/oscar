## Overview
- Data root: set `PREFIX=/path/to/datasets`. Most MRI datasets should be stored in `PREFIX/<DATASET>_<FS>/` (FS = field strength, e.g., `3T` or `1.5T`).
- Processed NIfTI volumes are stored in `PREFIX/<DATASET>_<FS>/processed/` by `process/create_data_objects.py`.
- Metadata CSVs are written to `PREFIX/METADATA/<DATASET>_ATTRIBUTES.csv` by `process/metadata.py` and are used for creating the training, validation and testing datasets.
- For HCP, raw scans are read from `PREFIX/HCP/<SUBJECT>/...`, while metadata is expected in `PREFIX/HCP_3T/`.

## Preprocessing steps
1) Place preprocessed brain-extracted NIfTIs in `PREFIX/<DATASET>_<FS>/<preprocessed>/`.
2) Ensure the dataset-specific metadata files described below are present in each dataset folder.
3) Run metadata aggregation:
```
PYTHONPATH=$PROJECTDIR python -u process/metadata.py --dataset <DATASET>
```
4) Create processed Niftis:
```
PYTHONPATH=$PROJECTDIR python -u process/create_data_objects.py --dataset <DATASET> --FS 3T
```
Repeat for `--FS 1.5T` where applicable.

## 3D Datasets

### [ADNI](https://ida.loni.usc.edu/home/projectPage.jsp?project=ADNI) (multifield: 3T, 1.5T)
- Preprocessed scans: `PREFIX/ADNI_<FS>/preprocessed/**/<IMAGE_ID>_MNI_Brain.nii.gz`.
- Required metadata: the main ADNI report CSV, `race.csv` (from LONI), and the field-strength CSV (`ADNI_<FS>.csv`) in `PREFIX/ADNI_<FS>/`.

### [OASIS](https://sites.wustl.edu/oasisbrains/home/oasis-3/) (multifield: 3T, 1.5T)
- Preprocessed scans + JSON with `MagneticFieldStrength` in `PREFIX/OASIS_<FS>/preprocessed/` (filenames like `<SUBJECT>_MNI_Brain.nii.gz`).
- Metadata: `metadata.csv`, `cognition.csv`, `healthy.csv` in `PREFIX/OASIS_<FS>/`.

### [IXI](https://brain-development.org/ixi-dataset/) (multifield: 3T, 1.5T)
- Preprocessed scans `PREFIX/IXI_<FS>/preprocessed/*Brain.nii.gz`.
- Metadata: `IXI.xls` in `PREFIX/IXI_<FS>/` (gender/race/age).

### [HCP](https://www.humanconnectome.org/study/hcp-young-adult/data-releases) (3T)
- Raw structure: `PREFIX/HCP/<SUBJECT>/T1w/T1w_acpc_dc_restore_brain.nii.gz`.
- Metadata: `metadata.csv` (public) and `metadata_restricted.csv` (restricted and used for age and race) in `PREFIX/HCP_3T/`.

### [A4](https://www.a4studydata.org/) (3T)
- Preprocessed scans + JSON in `PREFIX/A4_3T/preprocessed/` (filenames include visit code, end with `MNI_Brain.nii.gz`).
- Metadata: `metadata.csv` and `visits_datadic.csv` in `PREFIX/A4_3T/` for age adjustments across visits.

### [UKBB](https://www.ukbiobank.ac.uk/) (3T)
- Preprocessed scans `PREFIX/UKBB_3T/preprocessed/<SUBJECT>_Brain.nii.gz`.
- Metadata: `FILTERED.csv` from `process/ukbb_filter.py` in `PREFIX/UKBB_3T/` (contains subject/gender/race/age for allowed subjects).
- Scans from multiple ethnicities are provided (White/Black/Asian/Chinese) which are relatively higher in number as compared to the other datasets. We don't provide multi-race classification in the research papers but this dataset is suitable for multi-race classification. The scans are preprocessed using N4 bias correction.

### [ABIDE](https://ida.loni.usc.edu/home/projectPage.jsp?project=ABIDE) (3T)
- Preprocessed scans `PREFIX/ABIDE_3T/preprocessed/<IMAGE_ID>_Brain.nii.gz`.
- Metadata: `metadata.csv` in `PREFIX/ABIDE_3T/` (includes `Image Data ID`, `Subject`, `Sex`, `Age`, `Group`).

## Sanity checks
- After `process/metadata.py`, inspect `PREFIX/METADATA/<DATASET>_ATTRIBUTES.csv` for expected columns (e.g., `image_id,subject_id,scan,gender,race,age[,disease_group,...]`).
- `process/create_data_objects.py` skips already-processed scans to avoid reprocessing.
- If scans are skipped, check field-strength mismatches or missing JSON/metadata entries as printed by the scripts.

Note: The 3D dataset steps correspond to the MRI experiments from “Invisible attributes, visible biases: Exploring demographic shortcuts in MRI-based Alzheimer’s disease classification” (MICCAI FAIMI 2025).

## 2D Datasets

### CelebA
CelebA is downloaded through `torchvision.datasets.CelebA(..., download=True)`.
Set `PREFIX` as the dataset root before running the 2D scripts.

### CheXpert
Go to [Stanford AIMI](https://aimi.stanford.edu/data), get the [CheXpert dataset URL](https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2) and token, then download with [AzCopy](https://github.com/Azure/azure-storage-azcopy):
```
azcopy copy '<CHEXPERT_DATASET_URL_WITH_TOKEN>' \
  "$PREFIX/chexpert/data/" \
  --recursive
```

The following datasets are only used in https://github.com/acharaakshit/shortcut-groups.

### Waterbirds
Download [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) and [Places365](http://places2.csail.mit.edu/download.html).
Create the dataset with:
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
