import numpy as np
import os
import yaml
import argparse
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from scipy.stats import rankdata
import pickle
from scipy.ndimage import mean as region_mean
from datasets2d import get_biased_celeba_splits, get_biased_chexpert_splits
from partition import square_atlas_grid, kmeans_partition, build_ref_edge_image, superpixel_partition
from sklearn.model_selection import StratifiedKFold
from utils import saliency_score_percent
import logging

logging.basicConfig(level=logging.INFO)

def main(args):
    PROJECT_ROOT = os.getenv('PROJECTDIR')
    PREFIX = os.getenv('PREFIX')
    dataset = args.dataset
    model_name = args.model
    baseline = args.baseline
    method = args.method
    attribute = args.attribute
    seed = args.seed
    partition_method = args.partition_method
    regions = args.regions
    spearman = args.spearman
    saliency = args.saliency
    attenuation = args.attenuation


    assert not (spearman and saliency), "both spearman and saliency can't be true"
    

    # read folders yaml file
    with open(f'{PROJECT_ROOT}/config/folder.yaml') as f:
        folders = yaml.safe_load(f)

    EXPLAIN_DIR = folders['explain']
    RESULTS_DIR = folders['results']

    attr_labs = False
    if baseline:
        if attribute:
            attr_labs = True
    
    if dataset == 'celeba_gender':
        _, _, test_dataset = get_biased_celeba_splits(
                                            root=PREFIX,
                                            label="Blond_Hair",
                                            attribute="Male",
                                            balanced=baseline, # if true, give a balanced dataset,
                                            train_samples=10000,
                                            test_samples=1000,
                                            val_samples=500,
                                            attr_labs=attr_labs,
                                        )
    elif dataset == 'chexpert_pleuraleffusiongender':
        _, _, test_dataset = get_biased_chexpert_splits(
                                            root=os.path.join(PREFIX, "chexpert/data/chexpertchestxrays-u20210408/"),
                                            label="Pleural Effusion",
                                            attribute="Sex",
                                            balanced=baseline, # if true, give a balanced dataset,
                                            train_samples=10000,
                                            test_samples=1000,
                                            val_samples=500,
                                            attr_labs=attr_labs,
                                        )
    else:
        raise ValueError("Incorrect dataset passed")

    
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    if attenuation:
        assert attenuation in [1,2,3,4] # test set 1 or 2
        attenuation_string = f'_attenuation_{attenuation}'
        N = len(test_dataset)
        y   = test_dataset.labels.cpu().numpy().astype(int)
        attr = test_dataset.attributes.cpu().numpy().astype(int)
        joint = y * 10 + attr # just a unique code as 1,0 and 0,1 will be same
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

        folds = []
        for train_idx, test_idx in skf.split(np.arange(N), joint):
            folds.append((train_idx, test_idx))

        # Pick one of the three folds by attenuation (1..3)
        test_idx, _ = folds[attenuation - 1]

        subset_test = Subset(test_dataset, test_idx.tolist())
        test_dataloader = DataLoader(subset_test, batch_size=1, shuffle=False)
    else:
        attenuation_string = ''

    label = f'{dataset}_{model_name}_{baseline}_{attribute}_{method}__partition_{partition_method}_regions_{regions}_seed_{seed}{attenuation_string}'

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
        logging.info(f"attribution file exists")
        # return
    
    OUTPUT_DIR = os.path.join(PREFIX, dataset, EXPLAIN_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    partition_path = os.path.join(PREFIX, dataset, f"{partition_method}_{model_name}_regions_{regions}.npy")
    assert regions in [64,256], "regions can only be 64 or 256 for 2D"
    if not os.path.exists(partition_path):
        logging.info(f"Computing the partitions!")
        if partition_method == 'grid':
            if regions == 64:
                grid = (8, 8)
            else:
                grid = (16, 16)
            atlas = square_atlas_grid(size=(224,224), grid=grid)
        elif partition_method == 'kmeans':
            atlas = kmeans_partition(size=(224,224), parts=regions)
        elif partition_method == 'superpixel':
            # images: list or generator of (H, W, C) arrays, all same size
            images = []
            for _, batch in tqdm(enumerate(test_dataloader)):
                inputs = batch[0].squeeze(0).permute(1, 2, 0).cpu().numpy()
                images += [inputs]

            ref_img = build_ref_edge_image(images)
            atlas = superpixel_partition(ref_img, parts=regions, compactness=10.0)
            # `partition` is your global (H, W) label map, reusable for the entire dataset.

        else:
            raise ValueError('incorrect partition method!')

        np.save(partition_path, atlas)
    else:
        logging.info("partition already available, reusing!")
        atlas = np.load(partition_path)
    

    atlas_label_map =[]
    for idx, _ in tqdm(enumerate(test_dataloader)):
        gidx = test_dataloader.dataset.indices[idx] if isinstance(test_dataloader.dataset, Subset) else idx


        image_id = gidx
        savep = os.path.join(OUTPUT_DIR, f"{image_id}_{model_name}_{baseline}_{attribute}_{method}_seed_{seed}.npz")

        if not os.path.isfile(savep):
            logging.info(f"{savep} doesn't exist")
            continue

        att = np.load(savep)['array'][0][0]

        # l1 normalise as we are only using positive attributions
        att = att / (np.sum(att) + 1e-16)

        if saliency:
            _, scores = saliency_score_percent(sal_map=att, atlas=atlas)
        else:
            scores = region_mean(att, labels=atlas, index=np.unique(atlas))

        if not spearman:
            inv = -scores
            ranks = rankdata(inv, method='average')
            atlas_label_map += [ranks] # no float
        else:
            atlas_label_map += [scores]
        
        assert len(atlas_label_map[-1]) == regions, f"incorrect map generated with regions: {len(ranks)}"

    with open(save_path, "wb") as handle:
        pickle.dump(atlas_label_map, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="This is a script to compute aggregted attribution ranks for 2D datasets using provided partition and attribution maps.")
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    parser.add_argument("--model", type=str, default="efficientnetb0")
    parser.add_argument("--method", type=str, default='GradCAM')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--partition_method", type=str, default='grid')
    parser.add_argument("--regions", type=int, default=64)
    parser.add_argument("--baseline", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--attribute", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--spearman", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--saliency", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--attenuation", type=int, default=0)
    # Parse the arguments
    args = parser.parse_args()
    main(args)