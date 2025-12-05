import torch
from torch.utils.data import DataLoader
import argparse
import yaml
from datasets2d import get_biased_celeba_splits, get_biased_chexpert_splits
from models import Classifier2D
import lightning as L
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import json
import os
import numpy as np
from utils import per_group_acc
import logging

logging.basicConfig(level=logging.INFO)

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
    ckpt = f'MODEL_{model_name}_{in_channels}_' + f"SEED2D_{seed}_BASELINE_{baseline}_{attribute}_F1.ckpt" 

    checkpoint_name = os.path.join(PREFIX, dataset, CKPT_DIR, ckpt)

    attr_labs = False
    if baseline:
        if attribute:
            attr_labs = True

    img_size = 224

    if dataset == 'celeba_gender':
        train_dataset, _, test_dataset = get_biased_celeba_splits(
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
        train_dataset, _, test_dataset = get_biased_chexpert_splits(
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

    if mode != 'eval': # test on train split
        test_dataset = train_dataset
        
    # logging.info(dataset, attribute)
    corr = (test_dataset.labels == test_dataset.attributes).sum().div(len(test_dataset))

    logging.info(f"Dataset-level correlation: {corr:.3f}")

    model = Classifier2D.load_from_checkpoint(checkpoint_path=checkpoint_name,
                            model_alias=model_alias,
                            num_classes=2,
                            in_channels=in_channels,
                            lr=1e-4,
                            img_size=img_size,
                            strict=False
                        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()
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
    
    if not os.path.exists(json_file):
        with open(json_file, "a") as f:
            json.dump([], f)
        
    if baseline and not attribute:
        run_type = "balanced"
    elif attribute:
        run_type = "attribute"
    else:
        run_type = "biased"

    run_record = {
        "dataset": dataset,
        "label": test_dataset.label_name,
        "attribute": test_dataset.attribute_name,
        "run_type": run_type,
        "group_accuracy": gacc,
        "confusion_matrix": cm.tolist(),
        "classification_report": creport
    }

    key = f"{dataset}_{model_name}_{run_record['label']}_{run_record['attribute']}_{run_type}_seed{seed}"

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
    parser = argparse.ArgumentParser(description="This is a test script for 2D classification")
    parser.add_argument("--PREFIX", type=str, help='PREFIX for the main data folders.')
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    parser.add_argument("--model", type=str, default="efficientnetb0")
    parser.add_argument("--checkpoint", type=str, default="None")
    parser.add_argument("--mode", default='eval')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--json", type=str, default="report2D.json")
    parser.add_argument("--baseline", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--attribute", type=bool, action=argparse.BooleanOptionalAction)
    # Parse the arguments
    args = parser.parse_args()
    # Call main with parsed arguments
    main(args)