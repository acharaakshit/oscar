import torch
from torch.utils.data import DataLoader, Subset
import argparse
import yaml
from datasets2d import get_biased_celeba_splits, get_biased_chexpert_splits
from models import Classifier2D
import lightning as L
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import json
import os
import numpy as np
import logging
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from utils import per_group_acc, build_rcs_mask_from_npy

logging.basicConfig(level=logging.INFO)

def compute_classification_metrics(model, test_dataset, attribute):
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=16)

    trainer = L.Trainer(accelerator="auto", precision="bf16-mixed")
    test_results = trainer.predict(model, dataloaders=test_dataloader)


    predictions = []
    labels = []
    for preds, y in test_results:
        predictions.append(preds)
        labels.append(y)

    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    if attribute:
        gacc = per_group_acc(labels, test_dataset.labels, predictions)
    else:
        gacc = per_group_acc(labels, test_dataset.attributes, predictions)

    logging.info(gacc)

    creport = classification_report(labels, predictions, output_dict=True, zero_division=0.0)

    logging.info("Classification Report:")
    logging.info(creport)
    logging.info(f"Accuracy: {accuracy_score(labels, predictions):.4f}")
    cm = confusion_matrix(labels, predictions, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()

    acc_class_0 = tn / (tn + fp) if (tn + fp) else float("nan")  # correct 0's / all 0's
    acc_class_1 = tp / (tp + fn) if (tp + fn) else float("nan")  # correct 1's / all 1's

    logging.info(f"Confusion Matrix: {cm}")
    logging.info(f"Class-0 accuracy: {acc_class_0:.3f}")
    logging.info(f"Class-1 accuracy: {acc_class_1:.3f}")
    creport['0']['C'] = float(tn)
    creport['0']['N'] = float(fp)
    creport['0']['ACC'] = acc_class_0
    creport['1']['C'] = float(tp)
    creport['1']['N'] = float(fn)
    creport['1']['ACC'] = acc_class_1
    logging.info(creport)

    return gacc, cm, creport

class AttrLabelSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices.tolist())
        self.attribute_name = dataset.attribute_name
        self.label_name = dataset.label_name
        self.labels = dataset.labels[indices]
        self.attributes = dataset.attributes[indices]


def main(args):
    PROJECT_ROOT = os.getenv('PROJECTDIR')
    PREFIX = os.getenv('PREFIX')
    dataset = args.dataset
    model_name = args.model
    mode = args.mode
    in_channels = args.in_channels
    baseline = args.baseline
    attribute = args.attribute
    seed = args.seed
    masked = args.masked
    partition_method = args.partition
    regions = args.regions
    bias_samples_train = args.bias_samples_train
    bias_samples_val = args.bias_samples_val
    threshold = args.threshold
    shuffle = args.shuffle
    shuffle_seed = args.shuffle_seed
    attenuation = args.attenuation

    # read folders yaml file
    with open(f'{PROJECT_ROOT}/config/folder.yaml') as f:
        folders = yaml.safe_load(f)
    
    RESULTS_DIR = folders['results']
    CKPT_DIR = folders['checkpoints']

    json_file = os.path.join(PROJECT_ROOT, RESULTS_DIR, args.json) # to save results

    if not os.path.exists(json_file):
        with open(json_file, "w") as f:
            json.dump({}, f)

    # read models yaml file
    with open(f'{PROJECT_ROOT}/config/models.yaml') as f:
        models = yaml.safe_load(f)

    logging.info(f'The model used here is {model_name}')
    model_alias = models[model_name]

    # normal evaluation
    if bias_samples_train == 25 and bias_samples_val == 10:
        ckpt = f'MODEL_{model_name}_{in_channels}_' + f"SEED2D_{seed}_BASELINE_{baseline}_{attribute}_F1.ckpt" 
    else:
        ckpt = f'MODEL_{model_name}_{in_channels}_' + f"SEED2D_{seed}_BASELINE_{baseline}_{attribute}_{bias_samples_train}_{bias_samples_val}_F1.ckpt" 

    checkpoint_name = os.path.join(PREFIX, dataset, CKPT_DIR, ckpt)

    attr_labs = False
    if baseline:
        if attribute:
            attr_labs = True

    img_size = 224

    if dataset == 'celeba_gender':
        train_dataset, val_dataset, test_dataset = get_biased_celeba_splits(
                                            root=PREFIX,
                                            label="Blond_Hair",
                                            attribute="Male",
                                            balanced=baseline, # if true, give a balanced dataset,
                                            train_samples=10000,
                                            test_samples=1000,
                                            val_samples=500,
                                            bias_samples_train=bias_samples_train,
                                            bias_samples_val=bias_samples_val,
                                            attr_labs=attr_labs,
                                        )
    elif dataset == 'chexpert_pleuraleffusiongender':
        train_dataset, val_dataset, test_dataset = get_biased_chexpert_splits(
                                            root=os.path.join(PREFIX, "chexpert/data/chexpertchestxrays-u20210408/"),
                                            label="Pleural Effusion",
                                            attribute="Sex",
                                            balanced=baseline, # if true, give a balanced dataset,
                                            train_samples=10000,
                                            test_samples=1000,
                                            val_samples=500,
                                            bias_samples_train=bias_samples_train,
                                            bias_samples_val=bias_samples_val,
                                            attr_labs=attr_labs,
                                        )
    else:
        raise ValueError("Incorrect dataset passed")

    if mode != 'eval': # test on train split
        test_dataset = train_dataset
    
    corr = (test_dataset.labels == test_dataset.attributes).sum().div(len(test_dataset))
    

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

        # Pick one of the four folds by attenuation (1..4)
        val_idx, test_idx = folds[attenuation - 1] # opp of chosen for RCS

        held_test_dataset = AttrLabelSubset(test_dataset, test_idx)
        val_dataset = AttrLabelSubset(test_dataset, val_idx)
        test_dataset = held_test_dataset
    else:
        attenuation_string = ''
    
    if masked:
        masked_path = os.path.join(PREFIX, dataset, f"RCS_{model_name}_{partition_method}_{regions}_{bias_samples_train}{attenuation_string}.npy")
        rcs = build_rcs_mask_from_npy(masked_path)
    else:
        rcs=None
        
    # logging.info(dataset, attribute)

    logging.info(f"Dataset-level correlation: {corr:.3f}")

    model = Classifier2D.load_from_checkpoint(checkpoint_path=checkpoint_name,
                            model_alias=model_alias,
                            num_classes=2,
                            in_channels=in_channels,
                            lr=1e-4,
                            img_size=img_size,
                            use_rcs_consistency=masked,
                            rcs_mask=rcs,
                            strict=False
                        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    if baseline and not attribute:
        run_type = "balanced"
    elif attribute:
        run_type = "attribute"
    else:
        run_type = "biased"

    # now, we can have a placeholder threshold routine
    if threshold:
        assert masked, "This mode is specifically for computing thresholds"
        
        ups = [1, 5, 10, 15, 20, 25, 30, 40, 50]
        downs = [1, 5, 10, 15, 20, 25, 30, 40, 50]
        best_bacc = -1
        best_wgacc = -1
        best_ups = [] # in case no thresholds are found, run the same thing
        best_downs = [] # in case no thresholds are found, run the same thing

        # compute initial validation metrics
        model.use_rcs_consistency = False

        gacc, cm, creport = compute_classification_metrics(model=model, test_dataset=val_dataset, attribute=attribute)
        init_glevel = list(gacc[0].values())
        init_min_wg = min(init_glevel)
        init_bacc = sum(gacc[0].values())/len(gacc[0].values())

        model.use_rcs_consistency = True
        model.rcs_img_mask = rcs
        model.shuffle = shuffle # needed if shuffle is true
        model.shuffle_seed = shuffle_seed
        
        for up in ups:
            for down in downs:
                model.up  = up
                model.down = down

                gacc, cm, creport = compute_classification_metrics(model=model, test_dataset=val_dataset, attribute=attribute)
                glevel = list(gacc[0].values())
                bacc = sum(gacc[0].values())/len(gacc[0].values())
                # first check: improvement in worst group acc
                if min(glevel) >= best_wgacc: # compare only with best worst group accuracy in case the search fails (min(glevel) >= init_min_wg) and 
                    # second check: improvement in balanced accuracy
                    if (bacc - best_bacc  >= -0.005) and (bacc - init_bacc >= -0.005): # 0.05% limit on balanced accuracy for both best and init
                        logging.info(f"The best balanced accuracy has been updated to {bacc}")
                        # if worst group accuracy = initial worst group acc and the current worst group acc is lower
                        if min(glevel) == best_wgacc:
                            best_ups += [up]
                            best_downs += [down]
                        else: # worst group accuracy has improved while being in the acceptable balanced accuracy limit
                            best_ups = [up]
                            best_downs = [down]  

                        best_bacc = bacc
                        best_wgacc = min(glevel)
 
        logging.info(f"The best up threshold for dataset {dataset} and model {model_name} is {best_ups}.")
        logging.info(f"The best down threshold for dataset {dataset} and model {model_name} is {best_downs}.")
        logging.info(f"The best bacc for dataset {dataset} and model {model_name} is {best_bacc}.")

        json_file = 'results/thresholds.json'
        baccs = []
        wgs = []
        glevels = []
        for best_up, best_down in zip(best_ups, best_downs):
            model.up  = best_up
            model.down = best_down

            gacc, cm, creport = compute_classification_metrics(model=model, test_dataset=test_dataset, attribute=attribute) # compute on the actual test set
            bacc = sum(gacc[0].values())/len(gacc[0].values())
            glevel = list(gacc[0].values())

            baccs += [bacc]
            wgs += [min(glevel)]
            glevels += [glevel]

        run_record = {
            "dataset": dataset,
            "label": test_dataset.label_name,
            "attribute": test_dataset.attribute_name,
            "run_type": run_type,
            "balanced_acc": sum(baccs)/len(baccs),
            "worst_group_acc": sum(wgs)/len(wgs),
            "up": best_ups,
            "down": best_downs,
            "mean_group": pd.DataFrame(glevels).mean().to_dict(),
        }

    else:                    
        gacc, cm, creport = compute_classification_metrics(model=model, test_dataset=test_dataset, attribute=attribute)

        run_record = {
            "dataset": dataset,
            "label": test_dataset.label_name,
            "attribute": test_dataset.attribute_name,
            "run_type": run_type,
            "group_accuracy": gacc,
            "confusion_matrix": cm.tolist(),
            "classification_report": creport
        }

    if not os.path.exists(json_file):
        with open(json_file, "a") as f:
            json.dump([], f)

    if masked:
        if shuffle:
            key = f"{dataset}_{model_name}_{run_record['label']}_{run_record['attribute']}_{run_type}_seed{seed}_{bias_samples_train}_{bias_samples_val}_masked_{partition_method}_{regions}_shuffled_{shuffle_seed}{attenuation_string}"
        else:
            key = f"{dataset}_{model_name}_{run_record['label']}_{run_record['attribute']}_{run_type}_seed{seed}_{bias_samples_train}_{bias_samples_val}_masked_{partition_method}_{regions}{attenuation_string}"
    else:
        key = f"{dataset}_{model_name}_{run_record['label']}_{run_record['attribute']}_{run_type}_seed{seed}_{bias_samples_train}_{bias_samples_val}{attenuation_string}"

    try:
        with open(json_file, "r") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            payload = {}
    except (FileNotFoundError, json.JSONDecodeError):
        payload = {}

    payload[key] = run_record

    with open(json_file, "w") as f:
        json.dump(payload, f, indent=2)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is a test script for 2D classification, mitigation, with varying samples")
    parser.add_argument("--PREFIX", type=str, help='PREFIX for the main data folders.')
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    parser.add_argument("--model", type=str, default="efficientnetb0")
    parser.add_argument("--checkpoint", type=str, default="None")
    parser.add_argument("--mode", default='eval')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--bias-samples-train", type=int, default=25)
    parser.add_argument("--bias-samples-val", type=int, default=10)
    parser.add_argument("--json", type=str, default="mitig.json")
    parser.add_argument("--baseline", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--attribute", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--masked", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--partition", type=str, default=None)
    parser.add_argument("--regions", type=int, default=0)
    parser.add_argument("--threshold", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--shuffle", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--shuffle-seed", type=int, default=1)
    parser.add_argument("--attenuation", type=int, default=0)
    # Parse the arguments
    args = parser.parse_args()
    # Call main with parsed arguments
    main(args)