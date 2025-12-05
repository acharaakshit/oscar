import torch
from torch.utils.data import DataLoader
import argparse
import yaml
import create_splits
from data import SingleClassDataset
from models import CNN3DClassifier, SwinTransformer3DClassifier, VisionTransformer3DClassifier
import lightning as L
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import json
import os
import numpy as np
from utils import per_group_acc

def main(args):
    L.seed_everything(42, workers=True) # this is to ensure compatibility
    PROJECT_ROOT = os.getenv('PROJECTDIR')
    PREFIX = os.getenv('PREFIX')
    dataset = args.dataset
    model_name = args.model
    in_channels = args.in_channels
    baseline = args.baseline
    attribute = args.attribute
    seed = args.seed

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
    
    METADATA_DIR = folders['metadata']
    CKPT_DIR = folders['checkpoints']

    ckpt = f'MODEL_{model_name}_{in_channels}_' + f"SEED3D_{seed}_BASELINE_{baseline}_{attribute}_F1{biased_string}.ckpt" 

    checkpoint_name = os.path.join(PREFIX, dataset, CKPT_DIR, ckpt)
    
    JSON_DIR = os.path.join(PREFIX, dataset, 'ood')
    os.makedirs(JSON_DIR, exist_ok=True)

    json_file = os.path.join(JSON_DIR, args.json) # to save results

    if not os.path.exists(json_file):
        with open(json_file, "w") as f:
            json.dump({}, f)

    # read the tasks config
    with open(f'{PROJECT_ROOT}/config/tasks.yaml') as f:
        tasks = yaml.safe_load(f)

    # read models yaml file
    with open(f'{PROJECT_ROOT}/config/models.yaml') as f:
        models = yaml.safe_load(f)

    print(f'The model used here is {model_name}')
    model_alias = models[model_name]
    

    _, _, test_data = create_splits.split_data(
                                                metadata_path=os.path.join(PREFIX, METADATA_DIR, f'{dataset}_ATTRIBUTES.csv'), 
                                                tasks=tasks,
                                                baseline=baseline,
                                            )
    X_test = test_data['scan'].tolist()
    if attribute:
        y_class_test = torch.tensor(test_data['gender'].tolist(), dtype=torch.long)
    else:
        y_class_test = torch.tensor(test_data['disease_group'].tolist(), dtype=torch.long)

    test_dataset = SingleClassDataset(X_test, y_class_test, channels=in_channels)
    assert len(torch.unique(y_class_test)) == 2
    num_classes = 2
    
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    result_key = f"{dataset}_{model_name}_{baseline}_{attribute}_seed{seed}{biased_string}"

    if result_key in results:
        print("Result already exists, clear the JSON entry and run again!")
        # return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_alias == 'swintransformer':
        use_v2 = False if 'v1' in model_name else True
        model = SwinTransformer3DClassifier.load_from_checkpoint(
                                checkpoint_path=checkpoint_name,
                                img_size=(256, 256, 256), 
                                in_channels=in_channels,
                                num_classes=num_classes,
                                use_v2=use_v2)
    elif model_alias == 'vit':
        model = VisionTransformer3DClassifier.load_from_checkpoint(
                                checkpoint_path=checkpoint_name,
                                in_channels=in_channels,
                                num_classes=num_classes,
                                )
    else:
        model = CNN3DClassifier.load_from_checkpoint(
                                checkpoint_path=checkpoint_name, 
                                model_alias=model_alias, 
                                num_classes=num_classes, 
                                in_channels=in_channels,
                                strict=False
                                ) # weights are not required

    model.to(device)
    model.eval()
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16)

    trainer = L.Trainer(accelerator="auto", precision="bf16-mixed")
    test_results = trainer.predict(model, dataloaders=test_dataloader)

    predictions = []
    labels = []
    groups = []
    if attribute:
        for (preds, y), y_gt, group in zip(test_results, test_data['gender'].tolist(), test_data['disease_group'].tolist()):
            pred = int(preds.squeeze().item())
            lab  = int(y.squeeze().item())
            assert lab == int(y_gt)
            predictions.append(pred)
            labels.append(lab)
            groups.append(group)
        gacc = per_group_acc(labels, groups, predictions)
    else:
        for (preds, y), y_gt, group in zip(test_results, test_data['disease_group'].tolist(), test_data['gender'].tolist()):
            pred = int(preds.squeeze().item())
            lab  = int(y.squeeze().item())
            assert lab == int(y_gt)
            predictions.append(pred)
            labels.append(lab)
            groups.append(group)
        
        gacc = per_group_acc(labels, groups, predictions)


    predictions = np.array(predictions)
    labels = np.array(labels)
    group_labels_list = np.array(groups)

    creport = classification_report(labels, predictions, output_dict=True)
    print("Classification Report:")
    print(creport)
    print(f"Accuracy: {accuracy_score(labels, predictions):.4f}")
    print(confusion_matrix(labels, predictions, labels=[0,1]))

    cm = confusion_matrix(labels, predictions, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()

    acc_class_0 = tn / (tn + fp) if (tn + fp) else float("nan")  # correct 0's / all 0's
    acc_class_1 = tp / (tp + fn) if (tp + fn) else float("nan")  # correct 1's / all 1's

    print(f"Confusion Matrix: {cm}")
    print(f"Class-0 accuracy: {acc_class_0:.3f}")
    print(f"Class-1 accuracy: {acc_class_1:.3f}")
    creport['0']['C'] = int(tn)
    creport['0']['N'] = int(fp)
    creport['0']['ACC'] = acc_class_0
    creport['1']['C'] = int(tp)
    creport['1']['N'] = int(fn)
    creport['1']['ACC'] = acc_class_1

    print("\n=== Group-wise Classification Reports ===")

    groupwise_results = {}
    unique_groups = np.unique(group_labels_list)

    for group in unique_groups:
        mask = group_labels_list == group
        group_labels = labels[mask]
        group_preds = predictions[mask]

        # the code below is for two class setup, change manually if multiclass is used
        cm = confusion_matrix(group_labels, group_preds, labels=[0,1])
        tn, fp, fn, tp = cm.ravel()

        acc_class_0 = tn / (tn + fp) if (tn + fp) else float("nan")  # correct 0's / all 0's
        acc_class_1 = tp / (tp + fn) if (tp + fn) else float("nan")  # correct 1's / all 1's

        print(f"Class-0 accuracy: {acc_class_0:.3f}")
        print(f"Class-1 accuracy: {acc_class_1:.3f}")

        groupwise_results[f"group-{group}"] = {
                                            "0" : {"ACC": acc_class_0, "C": int(tn), "N": int(fp)},
                                            "1" : {"ACC": acc_class_1, "C": int(tp), "N": int(fn)}
                                        }

    results[result_key] = {}
    results[result_key]['disease'] = creport
    results[result_key]['gender'] = groupwise_results
    results[result_key]['group_accuracy'] = gacc

    with open(json_file, 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is a test script for 3D classification")
    parser.add_argument("--PREFIX", type=str, help='PREFIX for the main data folders.')
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    parser.add_argument("--model", type=str, default="efficientnetb0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--json", type=str, default="pamishortadni.json")
    parser.add_argument("--baseline", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--attribute", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--bias-samples-percent", type=float, default=None)
    # Parse the arguments
    args = parser.parse_args()
    # Call main with parsed arguments
    main(args)