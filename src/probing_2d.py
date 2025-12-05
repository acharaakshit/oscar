import torch
from torch.utils.data import DataLoader
import argparse
import yaml
from datasets2d import get_biased_celeba_splits, get_biased_chexpert_splits
from train_2d import split_summaries
from models import Classifier2D
import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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
    bias_samples_train = args.bias_samples_train
    bias_samples_val = args.bias_samples_val
    baseline = args.baseline

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
        if baseline:
            ckpt = f'MODEL_{model_name}_{in_channels}_' + f"SEED2D_{seed}_BASELINE_True_None_F1.ckpt" 
        else:
            ckpt = f'MODEL_{model_name}_{in_channels}_' + f"SEED2D_{seed}_BASELINE_None_None_F1.ckpt" 
    else:
        ckpt = f'MODEL_{model_name}_{in_channels}_' + f"SEED2D_{seed}_BASELINE_None_None_{bias_samples_train}_{bias_samples_val}_F1.ckpt" 

    checkpoint_name = os.path.join(PREFIX, dataset, CKPT_DIR, ckpt)

    img_size = 224

    if dataset == 'celeba_gender':
        train_dataset, _, test_dataset = get_biased_celeba_splits(
                                            root=PREFIX,
                                            label="Blond_Hair",
                                            attribute="Male",
                                            balanced=baseline,
                                            train_samples=10000,
                                            test_samples=1000,
                                            val_samples=500,
                                            bias_samples_train=bias_samples_train,
                                            bias_samples_val=bias_samples_val,
                                            attr_labs=False,
                                        )
        
        _, _, test_dataset1 = get_biased_celeba_splits(
                                            root=PREFIX,
                                            label="Blond_Hair",
                                            attribute="Male",
                                            balanced=baseline,
                                            train_samples=10000,
                                            test_samples=1000,
                                            val_samples=500,
                                            attr_labs=False,
                                        )
    elif dataset == 'chexpert_pleuraleffusiongender':
        train_dataset, _, test_dataset = get_biased_chexpert_splits(
                                            root=os.path.join(PREFIX, "chexpert/data/chexpertchestxrays-u20210408/"),
                                            label="Pleural Effusion",
                                            attribute="Sex",
                                            balanced=baseline,
                                            train_samples=10000,
                                            test_samples=1000,
                                            val_samples=500,
                                            bias_samples_train=bias_samples_train,
                                            bias_samples_val=bias_samples_val,
                                            attr_labs=False,
                                        )
        
        _, _, test_dataset1 = get_biased_chexpert_splits(
                                            root=os.path.join(PREFIX, "chexpert/data/chexpertchestxrays-u20210408/"),
                                            label="Pleural Effusion",
                                            attribute="Sex",
                                            balanced=baseline,
                                            train_samples=10000,
                                            test_samples=1000,
                                            val_samples=500,
                                            attr_labs=False,
                                        )
    else:
        raise ValueError("Incorrect dataset passed")
    
    # Correct usage
    assert torch.equal(test_dataset.labels, test_dataset1.labels), "Labels do not match!"
    assert torch.equal(test_dataset.attributes, test_dataset1.attributes), "Attributes do not match!"
        
    # logging.info(dataset, attribute)
    corr = (test_dataset.labels == test_dataset.attributes).sum().div(len(test_dataset))

    split_summaries(splits=[("train", train_dataset), ("test", test_dataset)], model_name=f"{dataset}_baseline")

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
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=16)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16)

    all_feats = []

    with torch.no_grad():
        for batch in tqdm(train_dataloader):
            x = batch[0].to(device)
            feats = model.model.forward_features(x)
            penult = model.model.forward_head(feats, pre_logits=True)
            all_feats.append(penult.cpu())

    X_train = torch.cat(all_feats, dim=0).numpy()
    a_train = train_dataset.attributes.numpy()

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, a_train)

    w = clf.coef_[0]

    w_norm_sq = np.dot(w, w)

    def remove_A_direction(z):
        alpha = np.dot(w, z) / w_norm_sq
        return z - alpha * w


    all_feats = []
    preds = []
    preds_clean = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            x = batch[0].to(device)
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
    a_test = test_dataset.attributes.numpy()

    a_pred = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(a_test, a_pred)
    logging.info(f"Leakage AUC (predicting A from features): {auc:.3f}")
    print(a_test.shape, test_dataset.labels.cpu().numpy().shape, np.array(preds).shape)

    print("CLEAN:", per_group_acc(y=test_dataset.labels.cpu().numpy(), a=a_test, yhat=np.array(preds_clean)))
    print("UNCHANGED:", per_group_acc(y=test_dataset.labels.cpu().numpy(), a=a_test, yhat=np.array(preds)))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is a test script for probing sensitive attribute encoding for 2D")
    parser.add_argument("--PREFIX", type=str, help='PREFIX for the main data folders.')
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    parser.add_argument("--model", type=str, default="efficientnetb0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--bias-samples-train", type=int, default=25)
    parser.add_argument("--bias-samples-val", type=int, default=10)
    parser.add_argument("--baseline", type=bool, action=argparse.BooleanOptionalAction)
    # Parse the arguments
    args = parser.parse_args()
    # Call main with parsed arguments
    main(args)