
# https://github.com/helenjin/soft_stability/blob/main/tutorial.ipynb
from datasets2d import get_biased_celeba_splits, get_biased_chexpert_splits
from tqdm import tqdm
import torch
import math
import yaml
import argparse
from torch.utils.data import DataLoader
import os
from models import Classifier2D
from captum.attr import LayerGradCam, Saliency
import torchvision.transforms.functional as TF
import json

def mask_image(image, selected_features):
    """
    Args:
        image (torch.FloatTensor): of shape (3,224,224).
        selected_features (torch.Tensor): a 0/1-valued tensor of shape (7,7).

    Returns:
        torch.FloatTensor: the alpha-masked image.
    """
    # print(selected_features.shape)
    big_mask = torch.nn.functional.interpolate(selected_features.float().view(1, 1, 14, 14), size=(224,224)) # (1,1,224,224)
    masked_image = image.view(1,3,224,224) * big_mask # (1,3,224,224)
    return masked_image.view(3,224,224)

def sample_alpha_pertbs(alpha: torch.LongTensor, radius: int, num_samples: int):
    """
    Sample uniformly from:
        Delta_r = {alpha' : alpha' >= alpha, |alpha' - alpha| <= r}

    Args:
        alpha (torch.LongTensor): The 0/1-valued tensor of shape (n,).
        radius (int): The radius within which we sample.
        num_samples (int): The number of perturbed samples to generate.

    Returns:
        torch.LongTensor: Sampled perturbations of shape (num_samples, n).
    """
    original_shape = alpha.shape
    alpha = alpha.view(-1)

    samples = alpha.view(1, -1).repeat(num_samples, 1)

    # Find indices where alpha is 0 (these can potentially be flipped to 1)
    zero_indices = torch.nonzero(alpha == 0, as_tuple=False).squeeze()
    num_zeros = zero_indices.numel()
    if radius > num_zeros:
        raise ValueError(f"Radius {radius} > num zeros {num_zeros}")

    # Compute log-binomial coefficients in log-space, because we have massive blow-up otherwise.
    log_flip_probs = torch.tensor(
        [
            math.lgamma(num_zeros + 1) - math.lgamma(i + 1) - math.lgamma(num_zeros - i + 1)
            for i in range(radius + 1)
        ],
        dtype = torch.float32,
        device = alpha.device
    )

    # Convert log-probs to sampling-friendly format via Gumbel-max trick
    gumbel_noise = -torch.log(-torch.log(torch.rand(num_samples, radius + 1, device=alpha.device)))
    log_probs_with_noise = log_flip_probs.view(1, -1) + gumbel_noise
    num_flips = torch.argmax(log_probs_with_noise, dim=-1)

    # Select random indices to flip in each sample
    for i, flips in enumerate(num_flips):
        flip_inds = torch.randperm(num_zeros)[:flips]
        samples[i, zero_indices[flip_inds]] = 1 # Flip selected indices from 0 to 1

    return samples.view(num_samples, *original_shape).long()

@torch.no_grad()
def estimate_stability_rate(
    f,
    x: torch.FloatTensor,
    alpha: torch.LongTensor,
    radius: int,
    epsilon: float = 0.1,
    delta: float = 0.1,
    batch_size: int = 16,
):
    """
    Measure the soft stability rate for a classifier of form y = f(x, alpha), where:
        soft_stability_rate = Pr_{alpha' ~ Delta_r} [f(x, alpha') == f(x, alpha)]

    Args:
        f: Any function (ideally nn.Module) that takes as input x, alpha.
        x: The input to f of some shape. NOT batched.
        alpha: The 0/1 attribution of some shape. NOT batched.
        radius: The radius to which we give the guarantee.
        epsilon: The error tolerance.
        delta: The admissible failure probability
        batch_size: The batch size in case we run out-of-memory.

    Returns:
        soft_stability_rate: A value between 0 and 1.
    """
    C, H, W = x.shape
    num_samples = int(math.log(2/delta) / (2 * (epsilon**2))) + 1
    y = f(mask_image(x, alpha).unsqueeze(0)) # Reference prediction
    all_y_pertbs = []

    for alpha_pertbs in torch.split(sample_alpha_pertbs(alpha, radius, num_samples), batch_size):
        x_pertbs = torch.stack([mask_image(x, a) for a in alpha_pertbs])
        y_pertbs = f(x_pertbs)
        all_y_pertbs.append(y_pertbs)

    all_y_pertbs = torch.cat(all_y_pertbs, dim=0)
    stab_rate = (y.argmax(dim=-1) == all_y_pertbs.argmax(dim=-1)).float().mean()
    return stab_rate

def main(args):
    PROJECT_ROOT = os.getenv('PROJECTDIR')
    PREFIX = os.getenv('PREFIX')
    dataset = args.dataset
    model_name = args.model
    method = args.method
    in_channels = 3
    baseline = args.baseline
    attribute = args.attribute
    seed = args.seed

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
    
    checkpoint_name = f'MODEL_{model_name}_{in_channels}_' + f"SEED2D_{seed}_BASELINE_{baseline}_{attribute}_LOSS.ckpt" 
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

    biased_model_checkpoint = os.path.join(PREFIX, dataset, CKPT_DIR, f'MODEL_{model_name}_{in_channels}_' + f"SEED2D_{seed}_BASELINE_None_None_LOSS.ckpt")

    biased_model = Classifier2D.load_from_checkpoint(checkpoint_path=biased_model_checkpoint,
                            model_alias=model_alias,
                            num_classes=2,
                            lr=1e-4,
                            img_size=img_size
                        )

    if model_name == 'resnet':
        target_layer = model.model.layer4[-1]
    elif model_name == 'mobilenet':
        target_layer = model.model.blocks[-1]
    elif model_name == 'vgg':
        target_layer = model.model.features[29]
    elif 'swin2d' in model_name:
        # norm of last layer and block will be chosen: https://jacobgil.github.io/pytorch-gradcam-book/vision_transformers.html
        blk = model.model.layers[-2].blocks[-1]
        orig = blk.norm1

        class BHWC2BCHW(torch.nn.Module):
            def forward(self, x): return x.permute(0, 3, 1, 2).contiguous()
        class BCHW2BHWC(torch.nn.Module):
            def forward(self, x): return x.permute(0, 2, 3, 1).contiguous()

        blk.norm1 = torch.nn.Sequential(orig, BHWC2BCHW(), BCHW2BHWC())
        target_layer = blk.norm1[1]  # the BCHW node for Captum
    else:
        raise ValueError("Model not supported yet!")

    print(f"Using target layer: {target_layer}")


    json_file = os.path.join(PREFIX, dataset, f"{model_name}_{baseline}_{attribute}_{method}_seed_{seed}_{args.json}")

    with open(json_file, 'w') as f:
        json.dump({}, f)
    
    with open(json_file, 'r') as f:
        results = json.load(f)

    model.eval()
    biased_model.eval()
    OUTPUT_DIR = os.path.join(PREFIX, dataset, EXPLAIN_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for idx, batch in tqdm(enumerate(test_dataloader)):
        model.to(device)
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        assert len(batch) == 2
        image_id = idx

        with torch.no_grad():
            outputs = model(inputs).argmax(dim=1).item()
        
        if method == 'GradCAM':
            attribution_handle = LayerGradCam(model, target_layer)
            attribution_map = attribution_handle.attribute(inputs, target=outputs, relu_attributions=True)
        elif method == 'Saliency':
            attribution_handle = Saliency(model)
            attribution_map = attribution_handle.attribute(inputs, target=outputs)
        else:
            raise ValueError('Method not supported yet!')
        

        # time to compute stability!
        attr = attribution_map.flatten()
        k = int(0.5 * attr.numel()) # keep top 50%
        _, idx = torch.topk(attr, k)

        alpha = torch.zeros_like(attr, dtype=torch.bool)
        alpha[idx] = True

        # Reshape back if necessary
        alpha = alpha.view_as(attribution_map)
        # print(alpha.shape)
        m = (alpha.numel() - alpha.sum()).item()
        print("Number of available features to add:", m)
        start = 0
        end = m+1
        radii = list(range(start, 30, 2))
        radii += list(range((end//2)+1, end, 20))
        results[f"{image_id}_{model_name}_{dataset}_{baseline}_{attribute}"] = {}
        for r in radii:
            stab_rate_r = estimate_stability_rate(model, inputs[0], alpha, radius=r)
            # print(f"Estimated stability for perturbations of size <= {r}: {stab_rate_r.item():.4f}")
            results[f"{image_id}_{model_name}_{dataset}_{baseline}_{attribute}"][r] = stab_rate_r.item()

    with open(json_file, 'w') as f:
        json.dump(results, f)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="This is an stability computation script for 2D")
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    parser.add_argument("--model", type=str, default="efficientnetb0")
    parser.add_argument("--mode", type=str, default='eval')
    parser.add_argument("--method", type=str, default='GradCAM')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--baseline", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--attribute", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--json", type=str, default="explain_2D.json")
    # Parse the arguments
    args = parser.parse_args()
    main(args)