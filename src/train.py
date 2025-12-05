import torch
from torch.utils.data import DataLoader
import argparse
import yaml
import create_splits
from data import SingleClassDataset
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import wandb
from lightning.pytorch.loggers import WandbLogger
from models import CNN3DClassifier, SwinTransformer3DClassifier, VisionTransformer3DClassifier
import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def subject_level_counts(df, scol='subject_id', ycol='disease_group', acol='gender'):
    # One row per subject with a single label and attribute
    subj_df = (
        df.groupby(scol)
          .agg({
              ycol: lambda x: x.mode()[0], # most common disease label for that subject
              acol: 'first' # gender is constant per subject in your pipeline
          })
          .reset_index()
    )

    # Count subjects by (Y, A) pair
    counts = (
        subj_df
        .groupby([ycol, acol])[scol]
        .nunique()
        .reset_index()
        .rename(columns={scol: 'n_subjects'})
    )

    return counts

    

def main(args):
    PROJECT_ROOT = os.getenv('PROJECTDIR')
    PREFIX = os.getenv('PREFIX')
    dataset = args.dataset
    assert args.in_channels in [1,2,3]
    in_channels = args.in_channels
    BATCHSIZE=args.batch_size
    EPOCHS = args.max_epochs
    model_name = args.model
    baseline=args.baseline
    attribute = args.attribute
    seed = args.seed
    bias_samples_percent = args.bias_samples_percent
    # set seed for pytorch lightning
    L.seed_everything(seed, workers=True)

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

    # read the tasks config
    with open(f'{PROJECT_ROOT}/config/tasks.yaml') as f:
        tasks = yaml.safe_load(f)

    # read models yaml file
    with open(f'{PROJECT_ROOT}/config/models.yaml') as f:
        models = yaml.safe_load(f)

    logging.info(f'The model used here is {model_name}')
    model_alias = models[model_name]

    metadata_path=os.path.join(PREFIX, METADATA_DIR, f'{dataset}_ATTRIBUTES.csv')
    df = pd.read_csv(metadata_path)

    train_data, val_data, test_data = create_splits.split_data(
                                                        metadata_path=os.path.join(PREFIX, METADATA_DIR, f'{dataset}_ATTRIBUTES.csv'),
                                                        tasks=tasks,
                                                        baseline=baseline,
                                                        biased_samples_percent=bias_samples_percent
                                                    )

    X_train = train_data['scan'].tolist()
    X_val = val_data['scan'].tolist()
    X_test = test_data['scan'].tolist()

    if attribute:
        # use attribute column
        y_class_train = torch.tensor(train_data['gender'].tolist(), dtype=torch.long)
        y_class_val = torch.tensor(val_data['gender'].tolist(), dtype=torch.long)
        y_class_test = torch.tensor(test_data['gender'].tolist(), dtype=torch.long)
    else:
        y_class_train = torch.tensor(train_data['disease_group'].tolist(), dtype=torch.long)
        y_class_val = torch.tensor(val_data['disease_group'].tolist(), dtype=torch.long)
        y_class_test = torch.tensor(test_data['disease_group'].tolist(), dtype=torch.long)

    logging.info(f"Train: {len(y_class_train)}, Val: {len(y_class_val)}, Test: {len(y_class_test)}")
    check_x = [x.split('/')[-1].split('.')[0] for x in X_train]
    logging.info(f"""Train:
        {df[df['image_id'].isin(check_x)]
        .groupby(['disease_group', 'gender'])
        .size()
        .reset_index(name='count')}"""
    )
    check_x = [x.split('/')[-1].split('.')[0] for x in X_val]
    logging.info(f"""Val:
        {df[df['image_id'].isin(check_x)]
        .groupby(['disease_group', 'gender'])
        .size()
        .reset_index(name='count')}"""
    )
    check_x = [x.split('/')[-1].split('.')[0] for x in X_test]
    logging.info(f"""Test:
        {df[df['image_id'].isin(check_x)]
        .groupby(['disease_group', 'gender'])
        .size()
        .reset_index(name='count')}"""
    )

    print("Train (subject-level) counts:")
    print(subject_level_counts(train_data))

    print("\nVal (subject-level) counts:")
    print(subject_level_counts(val_data))

    print("\nTest (subject-level) counts:")
    print(subject_level_counts(test_data))

    # this is the only script where augment will be used
    train_dataset = SingleClassDataset(X_train, y_class_train, augment=True, channels=in_channels)
    val_dataset = SingleClassDataset(X_val, y_class_val, augment=False, channels=in_channels)

    assert len(torch.unique(y_class_train)) == 2
    num_classes = 2

    if model_alias == 'swintransformer':
        use_v2 = False if 'v1' in model_name else True
        model = SwinTransformer3DClassifier(in_channels=in_channels,
                                            num_classes=num_classes, 
                                            lr=1e-4,
                                            use_checkpoint=True,
                                            use_v2=use_v2)
    elif model_alias == 'vit':
        model = VisionTransformer3DClassifier(in_channels=in_channels,
                                              num_classes=num_classes,
                                              drop_rate=0.1)
    else:
        model = CNN3DClassifier(model_alias=model_alias,
                                num_classes=num_classes,
                                in_channels=in_channels,
                                lr=5e-5, # finetuning lr should be low
                                drop_rate=0.2,
                                pretrained=True)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=8, pin_memory=True) # no shuffle here

    for batch_idx, batch in enumerate(train_dataloader):
        logging.info(f"Batch {batch_idx + 1}:")
        for elem in batch:
            logging.info(f"Shape: {elem.shape}")
        break

    SAVEDIR = os.path.join(PREFIX, dataset, 'saved_data/models/')
    os.makedirs(SAVEDIR, exist_ok=True)
    loss_checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1, # final runs
            filename=f'MODEL_{model_name}_{in_channels}_' + f"SEED3D_{seed}_BASELINE_{baseline}_{attribute}_LOSS{biased_string}" ,
            save_weights_only=False,
            dirpath=SAVEDIR,
        )
    

    f1_checkpoint_callback = ModelCheckpoint(
            monitor="val_fscore",
            mode="max",
            save_top_k=1, # final runs
            filename=f'MODEL_{model_name}_{in_channels}_' + f"SEED3D_{seed}_BASELINE_{baseline}_{attribute}_F1{biased_string}" ,
            save_weights_only=False,
            dirpath=SAVEDIR,
        )

    checkpoint_callbacks = [loss_checkpoint_callback, f1_checkpoint_callback]

    if model_name in models['pretrained']:
        gradient_clip_val = 0.5
    else:
        gradient_clip_val = None
    
    patience = models['early_epochs'][f'{model_name}']
    assert type(patience) == int and patience <= EPOCHS

    early_stopping_callback = EarlyStopping(
            monitor="val_fscore",
            mode="max",
            patience=patience,
            verbose=True,
            min_delta=0.01
        )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    wandb_logger = WandbLogger(project="attribute-classification-LASTDANCE-" + f"BASELINE-{baseline}-{attribute}", name=f"{dataset}-{model_name}-{in_channels}")

    trainer = L.Trainer(max_epochs=EPOCHS, 
                        accelerator="auto",
                        callbacks=[early_stopping_callback, lr_monitor] + checkpoint_callbacks,
                        logger=wandb_logger,
                        default_root_dir=SAVEDIR,
                        accumulate_grad_batches=4,
                        precision='bf16-mixed',
                        gradient_clip_val=gradient_clip_val,
                        )

    final_path = os.path.join(SAVEDIR, f"MODEL_{model_name}_{in_channels}_final_model_{seed}_BASELINE_{baseline}_{attribute}{biased_string}.ckpt")
    if os.path.exists(final_path):
        logging.info(f"Model {final_path} already exists!")
        return
    trainer.fit(model, train_dataloader, val_dataloader)
    # save model at the end
    trainer.save_checkpoint(final_path)
    wandb.finish()
    logging.info("Training Complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is a training script for 3D MRI based Alzheimer's disease classification")
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    parser.add_argument("--model", type=str, default="efficientnetb0")
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--baseline", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--attribute", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--bias-samples-percent", type=float, default=None)
    # Parse the arguments
    args = parser.parse_args()

    # Call main with parsed arguments
    main(args)
