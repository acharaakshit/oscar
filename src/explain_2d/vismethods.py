from datasets2d import get_biased_celeba_splits, get_biased_chexpert_splits
from tqdm import tqdm
import numpy as np
import torch
import yaml
import argparse
from torch.utils.data import DataLoader
import os
from models import Classifier2D
from captum.attr import LayerAttribution, LayerGradCam, Saliency
import torchvision.transforms.functional as TF
import lightning as L
import logging
from zennit.attribution import Gradient
from zennit.torchvision import ResNetCanonizer, VGGCanonizer
from zennit.composites import EpsilonPlusFlat
from lxt.efficient import monkey_patch, monkey_patch_zennit
import importlib

logging.basicConfig(level=logging.INFO)

def main(args):
    L.seed_everything(42, workers=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    PROJECT_ROOT = os.getenv('PROJECTDIR')
    PREFIX = os.getenv('PREFIX')
    dataset = args.dataset
    model_name = args.model
    method = args.method
    in_channels = 3
    baseline = args.baseline
    attribute = args.attribute
    seed = args.seed
    bias_samples_train = args.bias_samples_train
    bias_samples_val = args.bias_samples_val

    # read folders yaml file
    with open(f'{PROJECT_ROOT}/config/folder.yaml') as f:
        folders = yaml.safe_load(f)

    EXPLAIN_DIR = folders['explain']
    CKPT_DIR = folders['checkpoints']

    attr_labs = False
    if baseline:
        if attribute:
            attr_labs = True

    with open(f'{PROJECT_ROOT}/config/models.yaml') as f:
        models = yaml.safe_load(f)

    model_alias = models[model_name]
    
    if bias_samples_train and bias_samples_val:
        checkpoint_name = f'MODEL_{model_name}_{in_channels}_' + f"SEED2D_{seed}_BASELINE_{baseline}_{attribute}_{bias_samples_train}_{bias_samples_val}_F1.ckpt" 
    else:    
        checkpoint_name = f'MODEL_{model_name}_{in_channels}_' + f"SEED2D_{seed}_BASELINE_{baseline}_{attribute}_F1.ckpt" 
    # checkpoints[checkpoint][0] # model with highest F1-Score
    checkpoint_name = os.path.join(PREFIX, dataset, CKPT_DIR, checkpoint_name)

    img_size = 224
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

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Classifier2D.load_from_checkpoint(checkpoint_path=checkpoint_name,
                            model_alias=model_alias,
                            num_classes=2,
                            lr=1e-4,
                            img_size=img_size
                        )
    
    model.eval()
    model.to(device)

    canonizers = []

    if model_name == 'resnet':
        target_layer = model.model.layer4[-1] # verified
        canonizers = [ResNetCanonizer()]
    elif model_name == 'mobilenet':
        target_layer = model.model.blocks[5][2] # verified
    elif model_name == 'vgg':
        target_layer = model.model.features[29] # verified
        canonizers = [VGGCanonizer()]
    elif model_name == 'vggbn':
        target_layer = model.model.features[40]
        canonizers = [VGGCanonizer()]
    elif model_name == 'densenet':
        target_layer = model.model.features.denseblock4.denselayer16
    elif model_name == 'inception':
        target_layer = model.model.Mixed_7c # verified
    elif model_name == 'convnext':
        target_layer = model.model.stages[-1].blocks[-1]
    elif 'swin2d' in model_name:
        if method == "GradCAM":
            raise ValueError("GradCAM is not implemented for ViT")
        elif method == "LRP":
            raise ValueError("LRP is not implemented for Swin, currently!")
    elif 'vit' in model_name:
        if method == "GradCAM":
            raise ValueError("GradCAM is not implemented for ViT")
        vit_mod = importlib.import_module(model.model.__class__.__module__)
        monkey_patch(vit_mod, verbose=False)
        monkey_patch_zennit(verbose=False)
    else:
        raise ValueError("Model not supported yet!")

    if method == 'GradCAM':
        logging.info(f"Using target layer: {target_layer}")

    # biased_model.eval()
    OUTPUT_DIR = os.path.join(PREFIX, dataset, EXPLAIN_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.info(model_name)

    for idx, batch in tqdm(enumerate(test_dataloader)):
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        assert len(batch) == 2
        image_id = idx
        if bias_samples_train and bias_samples_val:
            savep = os.path.join(OUTPUT_DIR, f"{image_id}_{model_name}_{baseline}_{attribute}_{method}_seed_{seed}_{bias_samples_train}_{bias_samples_val}.npz")
        else:
            savep = os.path.join(OUTPUT_DIR, f"{image_id}_{model_name}_{baseline}_{attribute}_{method}_seed_{seed}.npz")
        
        if os.path.exists(savep):
            continue

        with torch.no_grad():
            outputs = model(inputs).argmax(dim=1).item()


        if method == 'GradCAM':
            attribution_handle = LayerGradCam(model, target_layer)
            attribution_map = attribution_handle.attribute(inputs, target=outputs, relu_attributions=True)
        elif method == "LRP":
            x = inputs.detach().requires_grad_(True)
            if 'vit' in model_name:

                vit_backbone = model.model
                patch_acts = None

                def save_patch_embeds(module, inp, out):
                    nonlocal patch_acts
                    patch_acts = out
                    patch_acts.retain_grad()

                handle = vit_backbone.conv_proj.register_forward_hook(save_patch_embeds)

                logits = model(x)
                handle.remove()

                logit = logits[0, outputs]
                model.zero_grad(set_to_none=True)
                logit.backward()

                relevance_map = (patch_acts * patch_acts.grad).sum(1, keepdim=True)
                attribution_map = torch.relu(relevance_map)
            else:
                composite = EpsilonPlusFlat(canonizers=canonizers)
                with Gradient(model, composite) as attr:
                    logits = model(x)
                    one_hot = torch.zeros_like(logits).scatter_(1, torch.tensor([[outputs]], device=device), 1.0)
                    _, relevance = attr(x, one_hot)
                attribution_map = torch.relu(relevance.sum(1, keepdim=True))
        elif method == 'Saliency':
            attribution_handle = Saliency(model)
            attribution_map = attribution_handle.attribute(inputs, target=outputs)
        else:
            raise ValueError('Method not supported yet!')
        
        attribution = LayerAttribution.interpolate(attribution_map.to(device), (inputs.shape[-1],inputs.shape[-1]), interpolate_mode='bilinear')
        np.savez_compressed(savep, array=attribution.detach().cpu().numpy())


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="This is a script to compute attribution maps for 2D classification models")
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    parser.add_argument("--model", type=str, default="efficientnetb0")
    parser.add_argument("--mode", type=str, default='eval')
    parser.add_argument("--method", type=str, default='GradCAM')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--baseline", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--attribute", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--bias-samples-train", type=int, default=None)
    parser.add_argument("--bias-samples-val", type=int, default=None)
    # Parse the arguments
    args = parser.parse_args()
    main(args)
