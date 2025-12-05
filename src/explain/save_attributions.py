import numpy as np
import os
import yaml
import ants
import create_splits
import argparse
from tqdm import tqdm
import hashlib
import logging

logging.basicConfig(level=logging.INFO)

maps = {
        'gender': {0: 'F', 1: 'M'},
        'disease_group': {'CN': 'CN', 'AD':'AD'} # since disease labels are already in this form
    }

def main(args):
    PROJECT_ROOT = os.getenv('PROJECTDIR')
    PREFIX = os.getenv('PREFIX')
    dataset = args.dataset
    model_name = args.model
    baseline = args.baseline
    attribute = args.attribute
    bias_samples_percent = args.bias_samples_percent
    seed = args.seed

    print(bias_samples_percent)

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
    rev_map = maps[task_col]
    logging.info(task_col)
    logging.info(rev_map)
    
    METADATA_DIR = folders['metadata']
    EXPLAIN_DIR = folders['explain']
    MAPS_DIR = os.path.join(PREFIX, dataset, EXPLAIN_DIR)

    map_paths = os.listdir(MAPS_DIR)
    map_paths = [m for m in map_paths if m.endswith(f'{model_name}_{baseline}_{attribute}_GradCAM_seed_{seed}{biased_string}.npz')]

    _, _, test_data = create_splits.split_data(
                                    metadata_path=os.path.join(PREFIX, METADATA_DIR, f'{dataset}_ATTRIBUTES.csv'), 
                                    tasks=tasks,
                                    baseline=baseline,
                                )
    
    image_ids = test_data['image_id'].tolist()
    image_paths = test_data['scan'].tolist()
    # task_map = {row['image_id']:row[task_col] for idx, row in test_data.iterrows() if rev_map[row[task_col]] in rev_map.values()}

    map_path_idx = {}
    for map_path in map_paths:
        map_path_idx[map_path.split('_')[0]] = map_path
    map_paths = [map_path for map_path in map_paths if map_path.split('_')[0] in image_ids]

    assert len(image_ids) == len(map_paths), "Unequal number of images and maps!"
    # maintain correctness in order
    map_paths = [map_path_idx[image_id] for image_id in image_ids]


    template_img = ants.image_read(os.path.join(PREFIX, 'fsl/data/standard/MNI152_T1_1mm_brain.nii.gz')) # MNI Atlas
    atlas_img = ants.image_read(os.path.join(PREFIX, 'Hammers_mith-n30r95-maxprob-MNI152-SPM12/Hammers_mith-n30r95-MaxProbMap-full-MNI152-SPM12.nii.gz'))
    assert ants.get_orientation(template_img) == ants.get_orientation(atlas_img)
    atlas_img = ants.resample_image_to_target(atlas_img, template_img, interp_type='nearestNeighbor')

    assert ants.image_physical_space_consistency(template_img, atlas_img)

    prev_digest = None
    for map_path, original_path in tqdm(zip(map_paths, image_paths)):
        image_id = map_path.split('_')[0]
        
        if map_path != original_path: # only done when explainability map is applied
            assert image_id in original_path and image_id == original_path.split('/')[-1].split('.')[0]


        if not os.path.isfile(original_path): # making sure input path exists
            logging.info(f"{original_path} doesn't exist")
            continue

        subject_img = ants.image_read(original_path, pixeltype='float')

        if ants.get_orientation(subject_img) != ants.get_orientation(template_img):
            subject_img = ants.reorient_image.reorient_image2(subject_img, orientation="".join(ants.get_orientation(template_img)))
        
        assert ants.get_orientation(subject_img) == ants.get_orientation(template_img)

        outprefix = os.path.join(PREFIX, 'ants/', original_path.split('/')[-1].split('.')[0])
        os.makedirs(os.path.dirname(outprefix), exist_ok=True)
        fwd_affine = outprefix + "0GenericAffine.mat"
        fwd_warp   = outprefix + "1Warp.nii.gz"
        if os.path.exists(fwd_affine) and os.path.exists(fwd_warp):
            logging.info(f"Reusing existing transform {fwd_affine}!")
            # Reuse saved transforms
            transform = {
                "fwdtransforms": [fwd_warp, fwd_affine],
                "invtransforms": [
                    fwd_affine,
                    outprefix + "1InverseWarp.nii.gz"
                ],
            }
        else:
            logging.info(f"{fwd_affine} doesn't exist, running transform!")
            logging.info(f"Save map to {outprefix}!")
            # transform subject image to the MNI template
            transform = ants.registration(fixed=template_img, moving=subject_img, type_of_transform='SyN', outprefix=outprefix, random_seed=42)
            continue
        subject_img_registered = ants.apply_transforms(fixed=template_img, moving=subject_img, transformlist=transform['fwdtransforms'])
        assert ants.image_physical_space_consistency(subject_img_registered, atlas_img)       
        
        attribution_path = os.path.join(MAPS_DIR, map_path)
        with np.load(attribution_path, mmap_mode=None) as npz:
            attr_map = np.array(npz['array'][0, 0], copy=True)
        
        attr_map_ants = ants.from_numpy(attr_map, origin=subject_img.origin, spacing=subject_img.spacing, direction=subject_img.direction)
        assert ants.image_physical_space_consistency(attr_map_ants, subject_img)
        
        if ants.get_orientation(subject_img) != ants.get_orientation(template_img):
            attr_map_ants = ants.reorient_image.reorient_image2(attr_map_ants, orientation="".join(ants.get_orientation(template_img)))
        
         # confirm orientation correction again
        assert ants.get_orientation(subject_img) == ants.get_orientation(template_img) == ants.get_orientation(attr_map_ants)
        
        attr_map_registered = ants.apply_transforms(fixed=template_img, moving=attr_map_ants, transformlist=transform['fwdtransforms'])
        assert ants.image_physical_space_consistency(attr_map_registered, atlas_img)

        digest = hashlib.md5(attr_map_registered.numpy().tobytes()).hexdigest()
        if prev_digest is not None and digest == prev_digest:
            logging.info(f"[WARN] subject {image_id} output is IDENTICAL to previous one!")
        prev_digest = digest

        # instead of saving label stats, save the image and attributions
        ants.image_write(attr_map_registered, filename=os.path.join(MAPS_DIR, f"{image_id}_{model_name}_{baseline}_{attribute}_GradCAM_seed_{seed}{biased_string}.nii.gz"))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="This is a script to compute transforms of attributions to the registered MRIs")
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    parser.add_argument("--model", type=str, default="efficientnetb0")
    parser.add_argument("--baseline", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--attribute", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bias-samples-percent", type=float, default=None)
    # Parse the arguments
    args = parser.parse_args()
    main(args)