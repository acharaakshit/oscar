from torch.utils.data import DataLoader
import argparse
import yaml
from datasets2d import get_biased_celeba_splits, get_biased_chexpert_splits
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import wandb
from lightning.pytorch.loggers import WandbLogger
from models import Classifier2D
import os
from utils import split_summaries
import logging

logging.basicConfig(level=logging.INFO)

def main(args):
    # set seed for pytorch lightning
    seed = int(args.seed)
    assert seed in list(range(150))
    L.seed_everything(seed, workers=True)

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
    bias_samples_train = args.bias_samples_train
    bias_samples_val = args.bias_samples_val

    # read models yaml file
    with open(f'{PROJECT_ROOT}/config/models.yaml') as f:
        models = yaml.safe_load(f)

    logging.info(f'The model used here is {model_name}')
    model_alias = models[model_name]

    attr_labs = False
    if baseline:
        if attribute:
            attr_labs = True

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
    
    logging.info("Train label counts")
    if not baseline and not attribute:
        split_summaries(splits=[("train", train_dataset), ("val", val_dataset), ("test", test_dataset)], model_name=f"{dataset}_biased")
    elif baseline and not attribute:
        split_summaries(splits=[("train", train_dataset), ("val", val_dataset), ("test", test_dataset)], model_name=f"{dataset}_baseline")
    else:
        split_summaries(splits=[("train", train_dataset), ("val", val_dataset), ("test", test_dataset)], model_name=f"{dataset}_attribute")

    train_dataloader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=8, pin_memory=True) # no shuffle here

    img_size = None
    for batch_idx, batch in enumerate(train_dataloader):
        logging.info(f"Batch {batch_idx + 1}:")
        for elem in batch:
            logging.info(f"Shape: {elem.shape}")
        img_size = batch[0].shape[-1]
        break

    model = Classifier2D(model_alias=model_alias,
                            num_classes=2,
                            in_channels=in_channels,
                            lr=1e-5 if 'chexpert' in dataset else 1e-4,
                            img_size=img_size
                        )


    SAVEDIR = os.path.join(PREFIX, dataset, 'saved_data/models/')
    os.makedirs(SAVEDIR, exist_ok=True)
    loss_checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1, # final runs
            filename=f'MODEL_{model_name}_{in_channels}_' + f"SEED2D_{seed}_BASELINE_{baseline}_{attribute}_{bias_samples_train}_{bias_samples_val}_LOSS",
            save_weights_only=False,
            dirpath=SAVEDIR,
        )

    f1_checkpoint_callback = ModelCheckpoint(
            monitor="val_fscore",
            mode="max",
            save_top_k=1, # final runs
            filename=f'MODEL_{model_name}_{in_channels}_' + f"SEED2D_{seed}_BASELINE_{baseline}_{attribute}_{bias_samples_train}_{bias_samples_val}_F1" ,
            save_weights_only=False,
            dirpath=SAVEDIR,
    )
    
    patience = 10 # fixed patience
    assert type(patience) == int and patience <= EPOCHS

    early_stopping_callback = EarlyStopping(
            monitor="val_fscore",
            mode="max",
            patience=patience,
            verbose=True,
            min_delta=0.0005
        )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    wandb_logger = WandbLogger(project="attribute-classification" + f"-SEED2D-{seed}-BASELINE-{baseline}-{attribute}", name=f"{dataset}-{model_name}-{in_channels}")

    trainer = L.Trainer(max_epochs=EPOCHS, 
                        accelerator="auto",
                        callbacks=[early_stopping_callback, lr_monitor, loss_checkpoint_callback, f1_checkpoint_callback],
                        logger=wandb_logger,
                        default_root_dir=SAVEDIR,
                        precision='bf16-mixed',
                        deterministic=True,
                        benchmark=False,
                        gradient_clip_val=0.5,
                        )
    
    final_path = os.path.join(SAVEDIR, f"MODEL_{model_name}_{in_channels}_final_model_SEED2D_{seed}_BASELINE_{baseline}_{attribute}_{bias_samples_train}_{bias_samples_val}.ckpt")
    if os.path.exists(final_path):
        logging.info(f"Model {final_path} already exists!")
        return

    trainer.fit(model, train_dataloader, val_dataloader)
    # save model at the end
    trainer.save_checkpoint(final_path)
    wandb.finish()
    logging.info("Training Complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is a training script for 2D classification")
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    parser.add_argument("--model", type=str, default="efficientnetb0")
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--bias-samples-train", type=int, default=25)
    parser.add_argument("--bias-samples-val", type=int, default=10)
    parser.add_argument("--baseline", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--attribute", type=bool, action=argparse.BooleanOptionalAction)
    # Parse the arguments
    args = parser.parse_args()


    main(args)
