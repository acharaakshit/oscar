import torch
import argparse
import yaml
from data import normalise
import nibabel as nib
from models import CNN3DClassifier
import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import create_splits
from tqdm import tqdm
from utils import per_group_acc
import logging

logging.basicConfig(level=logging.INFO)

def main(args):
    PROJECT_ROOT = os.getenv('PROJECTDIR')
    PREFIX = os.getenv('PREFIX')
    dataset = args.dataset
    model_name = args.model
    in_channels = args.in_channels
    seed = args.seed
    baseline = args.baseline

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
    
    RESULTS_DIR = folders['results']
    CKPT_DIR = folders['checkpoints']

    METADATA_DIR = folders['metadata']
    CKPT_DIR = folders['checkpoints']
    print(biased_string)

    if biased_string == '':
        if baseline:
            ckpt = f'MODEL_{model_name}_{in_channels}_' + f"SEED3D_{seed}_BASELINE_True_None_F1.ckpt" 
        else:
            ckpt = f'MODEL_{model_name}_{in_channels}_' + f"SEED3D_{seed}_BASELINE_None_None_F1.ckpt"
    else:
        ckpt = f'MODEL_{model_name}_{in_channels}_' + f"SEED3D_{seed}_BASELINE_None_None_F1{biased_string}.ckpt" 

    json_file = os.path.join(PROJECT_ROOT, RESULTS_DIR, args.json) # to save results

    if not os.path.exists(json_file):
        with open(json_file, "w") as f:
            json.dump({}, f)

    # read the tasks config
    with open(f'{PROJECT_ROOT}/config/tasks.yaml') as f:
        tasks = yaml.safe_load(f)

    # read models yaml file
    with open(f'{PROJECT_ROOT}/config/models.yaml') as f:
        models = yaml.safe_load(f)

    logging.info(f'The model used here is {model_name}')
    model_alias = models[model_name]
    

    checkpoint_name = os.path.join(PREFIX, dataset, CKPT_DIR, ckpt)

    img_size = 224

    train_data, _, test_data = create_splits.split_data(
                                metadata_path=os.path.join(PREFIX, METADATA_DIR, f'{dataset}_ATTRIBUTES.csv'), 
                                tasks=tasks,
                                baseline=baseline,
                                biased_samples_percent=bias_samples_percent
                            )

    X_train = train_data['scan'].tolist() 
    X_test = test_data['scan'].tolist()
    y_class_attributes_train = torch.tensor(train_data['gender'].tolist(), dtype=torch.long)
    y_class_attributes_test = torch.tensor(test_data['gender'].tolist(), dtype=torch.long)


    model = CNN3DClassifier.load_from_checkpoint(
                                checkpoint_path=checkpoint_name, 
                                model_alias=model_alias, 
                                num_classes=2, 
                                in_channels=in_channels,
                                strict=False
                                ) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    all_feats = []

    with torch.no_grad():
        for train_scan in tqdm(X_train):
            x = torch.from_numpy(normalise(nib.load(train_scan).get_fdata(), channels=1)).unsqueeze(0).to(device).to(device=device, dtype=torch.float32)
            feats = model.model.forward_features(x)
            penult = model.model.forward_head(feats, pre_logits=True)
            all_feats.append(penult.cpu())

    X_train = torch.cat(all_feats, dim=0).numpy()

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_class_attributes_train)

    w = clf.coef_[0]
    
    print(f"Shape of log reg coef", w.shape)

    w_norm_sq = np.dot(w, w)

    def remove_A_direction(z):
        alpha = np.dot(w, z) / w_norm_sq
        return z - alpha * w

    all_feats = []
    preds = []
    preds_clean = []
    with torch.no_grad():
        for test_scan in tqdm(X_test):
            x = torch.from_numpy(normalise(nib.load(test_scan).get_fdata(), channels=1)).unsqueeze(0).to(device=device, dtype=torch.float32)
            feats = model.model.forward_features(x)
            penult = model.model.forward_head(feats, pre_logits=True)
            all_feats.append(penult.cpu())
            z = penult.squeeze(0).cpu().numpy()
            z_clean = remove_A_direction(z)
            yhat_clean = model.model.fc(torch.from_numpy(z_clean).to(device=device, dtype=torch.float32)).cpu().numpy()
            yhat = model.model.fc(torch.from_numpy(z).to(device=device, dtype=torch.float32)).cpu().numpy()
            preds_clean += [np.argmax(yhat_clean, axis=-1)]
            preds += [np.argmax(yhat, axis=-1)]

    X_test = torch.cat(all_feats, dim=0).numpy()

    a_pred = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_class_attributes_test, a_pred)
    logging.info(f"Leakage AUC (predicting A from features): {auc:.3f}")

    print("CLEAN:", per_group_acc(y=test_data['disease_group'].to_numpy(), a=test_data['gender'], yhat=np.array(preds_clean)))
    print("UNCHANGED:", per_group_acc(y=test_data['disease_group'].to_numpy(), a=test_data['gender'], yhat=np.array(preds)))

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is a test script for probing sensitive attribute encoding for 3D")
    parser.add_argument("--PREFIX", type=str, help='PREFIX for the main data folders.')
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    parser.add_argument("--model", type=str, default="efficientnetb0")
    parser.add_argument("--checkpoint", type=str, default="None")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--baseline", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--bias-samples-percent", type=float, default=None)
    # Parse the arguments
    args = parser.parse_args()
    # Call main with parsed arguments
    main(args)