from datasets2d import (
    get_biased_celeba_splits,
    get_biased_chexpert_splits,
    get_biased_multiattribute_celeba_splits,
)
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

import sys
from pathlib import Path


# do: git clone https://github.com/vaynexie/CausalX-ViT.git and adjust the path below
repoCX = Path("~/fairness/CausalX-ViT")

sys.path.insert(0, str(repoCX / "ViT_CX"))
sys.path.insert(0, str(repoCX / "ViT_CX" / "py_cam"))

# do: git clone https://github.com/aenglebert/Transformer_Input_Sampling.git and adjust the path below
repoCX = Path("~/fairness/Transformer_Input_Sampling")

sys.path.insert(0, str(repoCX))

from ViT_CX import ViT_CX, reshape_function_vit
from tis import TIS

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
    multilabel_sa = (
        dataset == 'celeba_gender_multiattr'
        and baseline
        and attribute
    )

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
    
    if dataset == 'celeba_gender_multiattr':
        if not baseline:
            checkpoint_role = output_role = "ts"
        elif multilabel_sa:
            checkpoint_role = "sa_multilabel"
            output_role = f"sa_multilabel_{args.sa_attribute.lower()}"
        else:
            checkpoint_role = output_role = "ba"
        checkpoint_name = (
            f"MODEL_{model_name}_{in_channels}_SEED2D_{seed}_"
            f"MULTIATTR_{checkpoint_role}_F1.ckpt"
        )
    elif bias_samples_train and bias_samples_val:
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
    elif dataset == 'celeba_gender_multiattr':
        _, _, test_dataset = get_biased_multiattribute_celeba_splits(
                                            root=PREFIX,
                                            balanced=baseline,
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
                            img_size=img_size,
                            multilabel=multilabel_sa,
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
        if dataset == 'celeba_gender_multiattr':
            savep = os.path.join(
                OUTPUT_DIR,
                f"{image_id}_{model_name}_MULTIATTR_{output_role}_{method}_"
                f"seed_{seed}.npz",
            )
        elif bias_samples_train and bias_samples_val:
            savep = os.path.join(OUTPUT_DIR, f"{image_id}_{model_name}_{baseline}_{attribute}_{method}_seed_{seed}_{bias_samples_train}_{bias_samples_val}.npz")
        else:
            savep = os.path.join(OUTPUT_DIR, f"{image_id}_{model_name}_{baseline}_{attribute}_{method}_seed_{seed}.npz")

        if os.path.exists(savep) and method not in {"CX", "TiS"}:
            continue

        with torch.no_grad():
            logits = model(inputs)
            if multilabel_sa:
                target_attribute = 0 if args.sa_attribute == "Male" else 1
                target_class = logits[0, target_attribute].argmax().item()
                captum_target = (target_attribute, target_class)
            else:
                target_class = logits.argmax(dim=1).item()
                captum_target = target_class


        if method == 'GradCAM':
            attribution_handle = LayerGradCam(model, target_layer)
            attribution_map = attribution_handle.attribute(
                inputs, target=captum_target, relu_attributions=True
            )
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

                if multilabel_sa:
                    logit = logits[0, target_attribute, target_class]
                else:
                    logit = logits[0, target_class]
                model.zero_grad(set_to_none=True)
                logit.backward()

                relevance_map = (patch_acts * patch_acts.grad).sum(1, keepdim=True)
                attribution_map = torch.relu(relevance_map)
            else:
                composite = EpsilonPlusFlat(canonizers=canonizers)
                with Gradient(model, composite) as attr:
                    logits = model(x)
                    one_hot = torch.zeros_like(logits)
                    if multilabel_sa:
                        one_hot[0, target_attribute, target_class] = 1.0
                    else:
                        one_hot[0, target_class] = 1.0
                    _, relevance = attr(x, one_hot)
                attribution_map = torch.relu(relevance.sum(1, keepdim=True))
        elif method == 'Saliency':
            attribution_handle = Saliency(model)
            attribution_map = attribution_handle.attribute(
                inputs, target=captum_target
            )
        elif method == 'CX':
            if multilabel_sa:
                raise ValueError("We don't use CX for the multi-attribute SA model")
            assert model_name == 'vit', "only works for vit"
            attribution_map = ViT_CX(
                model=model,
                image=inputs,                  # [1, 3, H, W]
                target_layer=model.model.encoder.layers[-1].ln_1,
                target_category=target_class, #None,                # None = top-1 class
                reshape_function=reshape_function_vit,
                gpu_batch=5000,
            )
            attribution_map = np.maximum(attribution_map,0.0)
            np.savez_compressed(savep, array=attribution_map)
            continue
        elif method == 'TiS':
            if multilabel_sa:
                raise ValueError("We don't use TiS for the multi-attribute SA model")
            # already non-negative
            saliency_method = TIS(model.model, batch_size=512)
            attribution_map = saliency_method(inputs, 
                    class_idx=target_class
                    ).cpu()
            attribution_map = torch.relu(attribution_map)
            np.savez_compressed(savep, array=attribution_map)
            continue
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
    parser.add_argument("--sa-attribute", choices=("Male", "Smiling"), default="Male")
    parser.add_argument("--bias-samples-train", type=int, default=None)
    parser.add_argument("--bias-samples-val", type=int, default=None)
    # Parse the arguments
    args = parser.parse_args()
    main(args)
