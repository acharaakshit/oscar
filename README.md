# Localising Shortcut Learning via OSCAR

Pipelines for demographic shortcut analysis and mitigation in image classifiers. Includes the MICCAI workshop components and the OSCAR attribution workflow.

## Layout
```
scripts/        # train / eval / explain wrappers
src/            # models, data, trainers, eval
explain[_2d]/   # attribution + stability (3D/2D)
process/        # preprocessing + metadata (3D MRI)
config/         # tasks, models, folders
requirements.txt
setup.sh
```

## Setup
```
git clone https://github.com/acharaakshit/oscar.git
cd oscar
conda create -n suioscar python=3.10 -y
conda activate suioscar
pip install -r requirements.txt
export PROJECTDIR=$PWD
export PREFIX=/path/to/datasets
export PYTHONPATH=$PROJECTDIR
```

## Data
- Raw data: `PREFIX/<dataset>/...`
- Metadata CSVs: `PREFIX/METADATA/` (see `config/folder.yaml`)
- Tasks/columns defined in `config/tasks.yaml`

## Run
- Train 3D: `scripts/train_scut.sh`
- Train 2D: `scripts/train_scut_2d.sh`
- Bias sweep (2D): `scripts/train_scut_2d_samples_exp.sh`
- Evaluate: `scripts/evaluate*.sh`
- Explain: `scripts/explain*.sh` (2D/3D attribution maps)
- Aggregate stats: `scripts/attribution_statistics*.sh`

## Correlation workflow (2D)
- Attribution maps: run `scripts/attribution_statistics_2d*.sh` with the same seed/bias counts/partition+regions you trained and explained; the filenames produced must match `__partition_{partition}_regions_{region}_seed_{seed}[...].pkl`.
- Partial correlations: `python3 -u src/vismethods_2d.py --outfile '...' --seeds N` treats `--seeds` as “number of seeds” (loops `i=0..N-1`). Ensure those seeds were trained/explained or reduce `N`.

## Attribution
- 2D maps via `explain_2d/vismethods.py`; inspect in `notebooks/vismethods_2d.ipynb`.
- RCS mitigation: `use_rcs_consistency`, `rcs_mask`, `up`, `down`, `shuffle_seed`.
- Stability tooling: `explain_2d/` (2D), `explain/` (3D).

## Citation
```bibtex
@inproceedings{achara2025invisible,
  title={Invisible attributes, visible biases: Exploring demographic shortcuts in mri-based alzheimer’s disease classification},
  author={Achara, Akshit and Anton, Esther Puyol and Hammers, Alexander and King, Andrew P and Alzheimers Disease Neuroimaging Initiative},
  booktitle={MICCAI Workshop on Fairness of AI in Medical Imaging},
  pages={156--166},
  year={2025},
  organization={Springer}
}
```

License: MIT  
Contact: @acharaakshit
