import numpy as np
import os
import yaml
import ants
import create_splits
import argparse
from tqdm import tqdm
from scipy.stats import rankdata
import pickle
from scipy.ndimage import mean as region_mean
import logging
from joblib import Parallel, delayed
from explain_2d.utils import saliency_score_percent
import nibabel as nib


logging.basicConfig(level=logging.INFO)

def main(args):
    PROJECT_ROOT = os.getenv('PROJECTDIR')
    PREFIX = os.getenv('PREFIX')
    dataset = args.dataset
    model_name = args.model
    baseline = args.baseline
    method = args.method
    partition_method = args.partition
    attribute = args.attribute
    regions = args.regions
    seed = args.seed
    spearman = args.spearman
    saliency = args.saliency

    assert not (spearman and saliency), "both spearman and saliency can't be true"

    bias_samples_percent = args.bias_samples_percent

    if bias_samples_percent:
        biased_string = f'_biased_{bias_samples_percent}'
        assert bias_samples_percent in [10, 20, 30, 40]
        bias_samples_percent = bias_samples_percent/100.0
    else:
        biased_string = ''

    # read folders yaml file
    with open(f'{PROJECT_ROOT}/config/folder.yaml') as f:
        folders = yaml.safe_load(f)

    # read the tasks config
    with open(f'{PROJECT_ROOT}/config/tasks.yaml') as f:
        tasks = yaml.safe_load(f)

    task_col = 'disease_group'
    print(task_col)
    
    METADATA_DIR = folders['metadata']
    EXPLAIN_DIR = folders['explain']
    RESULTS_DIR = folders['results']
    MAPS_DIR = os.path.join(PREFIX, dataset, EXPLAIN_DIR)

    assert partition_method == "atlas", "only atlas experiments for 3D"

    map_paths = os.listdir(MAPS_DIR)
    # use registered
    map_paths = [m for m in map_paths if m.endswith(f'{model_name}_{baseline}_{attribute}_GradCAM_seed_{seed}{biased_string}.nii.gz')]


    _, _, test_data = create_splits.split_data(
                                    metadata_path=os.path.join(PREFIX, METADATA_DIR, f'{dataset}_ATTRIBUTES.csv'), 
                                    tasks=tasks,
                                    baseline=baseline,
                                )
    
    image_ids = test_data['image_id'].tolist()
    image_paths = test_data['scan'].tolist()

    map_path_idx = {}
    for map_path in map_paths:
        map_path_idx[map_path.split('_')[0]] = map_path
    map_paths = [map_path for map_path in map_paths if map_path.split('_')[0] in image_ids]

    print(len(map_paths), len(image_paths), len(image_ids))

    assert len(image_ids) == len(map_paths), "Unequal number of images and maps, running subject registration only!"
    # maintain correctness in order
    map_paths = [map_path_idx[image_id] for image_id in image_ids]

    print(len(map_paths), len(image_paths), len(image_ids))
    assert len(image_ids) == len(map_paths) == len(image_paths)

    # maintain correctness in order
    map_paths = [map_path_idx[image_id] for image_id in image_ids]

    template_img = ants.image_read(os.path.join(PREFIX, 'fsl/data/standard/MNI152_T1_1mm_brain.nii.gz')) # MNI Atlas
    atlas_img = ants.image_read(os.path.join(PREFIX, 'Hammers_mith-n30r95-maxprob-MNI152-SPM12/Hammers_mith-n30r95-MaxProbMap-full-MNI152-SPM12.nii.gz'))
    atlas_img = ants.resample_image_to_target(atlas_img, template_img, interp_type='nearestNeighbor')

    assert ants.image_physical_space_consistency(template_img, atlas_img)

    label = f'{dataset}_{model_name}_{baseline}_{attribute}_{method}__partition_{partition_method}_seed_{seed}_regions_{regions}{biased_string}'

    SAVEDIR = os.path.join(PREFIX, dataset, RESULTS_DIR)
    os.makedirs(SAVEDIR, exist_ok=True)

    if spearman:
            save_path = os.path.join(SAVEDIR, f'{label}_spearman.pkl')
    else:
        if saliency:
            save_path = os.path.join(SAVEDIR, f'{label}_saliency.pkl')
        else:
            save_path = os.path.join(SAVEDIR, f'{label}.pkl')

    if os.path.isfile(save_path):
        print(f"attribution file exists")

    assert regions in [64, 216, 512, 4096, 96], "regions can only be 64, 216, 512 or 4096 for spatial partitioning and 96 for atlas"

    partition_path = os.path.join(PREFIX, dataset, f"{partition_method}_{model_name}_regions_{regions}.npy")

    if not os.path.exists(partition_path):
        logging.info(f"Computing the partitions!")
        numpy_atlas = atlas_img.numpy()

        np.save(partition_path, numpy_atlas)
    else:
        logging.info("partition already available, reusing!")
        numpy_atlas = np.load(partition_path)

    labels = np.unique(numpy_atlas)
    atlas_label_map =[]

    def process_subject(map_path, original_path):
        image_id = map_path.split('_')[0]
        assert image_id in original_path and image_id == original_path.split('/')[-1].split('.')[0]
        attribution_path = os.path.join(MAPS_DIR, map_path)
        if partition_method == 'atlas':
            subject_attribution = ants.image_read(attribution_path).numpy()
        else:
            subject_attribution = np.load(attribution_path)['array'][0][0]

        subject_attribution = subject_attribution.astype(np.float32)
        subject_attribution = subject_attribution / (subject_attribution.sum() + 1e-16)

        if not os.path.isfile(original_path):
            print(f"{original_path} doesn't exist")
            return

        if saliency:
            _, scores = saliency_score_percent(sal_map=subject_attribution, atlas=numpy_atlas)
        else:
            scores = region_mean(subject_attribution, labels=numpy_atlas, index=labels)
        if not spearman:
            inv = -scores
            ranks = rankdata(inv, method='average')
        else:
            ranks = scores

        return ranks

    atlas_label_map = Parallel(n_jobs=4)(
        delayed(process_subject)(m, o) for m, o in tqdm(zip(map_paths, image_paths))
    )

    with open(save_path, "wb") as handle:
        pickle.dump(atlas_label_map, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="This is script to compute attribution ranks based on the provided partition and attribution maps.")
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    parser.add_argument("--model", type=str, default="efficientnetb0")
    parser.add_argument("--method", type=str, default='GradCAM')
    parser.add_argument("--regions", type=int, default=512)
    parser.add_argument("--baseline", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--attribute", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--partition", type=str, default='grid')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--spearman", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--bias-samples-percent", type=float, default=None)
    parser.add_argument("--saliency", type=bool, action=argparse.BooleanOptionalAction)
    # Parse the arguments
    args = parser.parse_args()
    main(args)