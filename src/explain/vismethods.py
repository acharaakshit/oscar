from dataset import SingleClassDataset
import create_splits
from tqdm import tqdm
import numpy as np
import torch
import yaml
import argparse
from torch.utils.data import DataLoader
import os
from models import CNN3DClassifier, SwinTransformer3DClassifier
from captum.attr import LayerAttribution, LayerGradCam
import torch.nn.functional as F
import math
import json
from skimage.filters import threshold_otsu
import lightning as L

def get_fg_mask_3d(volume):

    thresh = threshold_otsu(volume.ravel())

    mask_np = (volume > thresh).astype(np.int64)
    return torch.from_numpy(mask_np)

def get_fg_mask_flat(fg_mask_3d, patch_size = 16):
    m = fg_mask_3d.float().unsqueeze(0).unsqueeze(0)

    d, h, w = fg_mask_3d.shape
    pd, ph, pw = d // patch_size, h // patch_size, w // patch_size

    # max‐pool each patch to check if any voxel==1
    patch_mask = F.adaptive_max_pool3d(
        m,
        output_size=(pd, ph, pw)
    )

    fg_mask_flat = patch_mask.squeeze().flatten().long()
    return fg_mask_flat

def sample_alpha_pertbs(alpha: torch.LongTensor, radius: int, num_samples: int):
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

def mask_image(image, selected_features, patch_size=16):
    big_mask = torch.nn.functional.interpolate(selected_features.float().view(1,1,patch_size, patch_size, patch_size), size=(256,256,256))
    masked_image = image.view(1, 1, 256,256, 256) * big_mask
    return masked_image.view(1, 1, 256,256, 256)

@torch.no_grad()
def estimate_stability_rate(
    f,
    x: torch.FloatTensor,
    alpha: torch.LongTensor,
    radius: int,
    epsilon: float = 0.1,
    delta: float = 0.1,
):
    num_samples = int(math.log(2/delta) / (2 * (epsilon**2))) + 1
    y = f(mask_image(x, alpha)) # Reference prediction
    all_y_pertbs = []
    alpha_pertbs =  sample_alpha_pertbs(alpha, radius, num_samples)
    for a in tqdm(alpha_pertbs):
        x_pertbs = mask_image(x, a)
        # np.savez_compressed("masked.npz", array=x_pertbs.detach().cpu().numpy())
        y_pertbs = f(x_pertbs)
        all_y_pertbs.append(y_pertbs)

    all_y_pertbs = torch.cat(all_y_pertbs, dim=0)
    stab_rate = (y.argmax(dim=-1) == all_y_pertbs.argmax(dim=-1)).float().mean()
    return stab_rate

def main(args):
    PROJECT_ROOT = os.getenv('PROJECTDIR')
    PREFIX = os.getenv('PREFIX')
    dataset = args.dataset
    in_channels = args.in_channels
    model_name = args.model
    check_stability = args.stability
    baseline = args.baseline
    attribute = args.attribute
    bias_samples_percent = args.bias_samples_percent
    seed = args.seed
    L.seed_everything(seed=seed, workers=True)

    if bias_samples_percent:
        biased_string = f'_biased_{bias_samples_percent}'
        assert bias_samples_percent in [10, 20, 30, 40]
        bias_samples_percent = bias_samples_percent/100.0
    else:
        biased_string = ''

    # read folders yaml file
    with open(f'{PROJECT_ROOT}/config/folder.yaml') as f:
        folders = yaml.safe_load(f)

    EXPLAIN_DIR = folders['explain']
    METADATA_DIR = folders['metadata']
    CKPT_DIR = folders['checkpoints']

    # read the tasks config
    with open(f'{PROJECT_ROOT}/config/tasks.yaml') as f:
        tasks = yaml.safe_load(f)


    JSON_DIR = os.path.join(PREFIX, dataset, 'stab')
    os.makedirs(JSON_DIR, exist_ok=True)

    json_file = os.path.join(JSON_DIR, args.json)

    if not os.path.exists(json_file) and check_stability: # create only when stability check is required
        with open(json_file, "w") as f:
            json.dump({}, f)

    with open(f'{PROJECT_ROOT}/config/models.yaml') as f:
        models = yaml.safe_load(f)

    model_alias = models[model_name]

    ckpt = f'MODEL_{model_name}_{in_channels}_' + f"SEED3D_{seed}_BASELINE_{baseline}_{attribute}_F1{biased_string}.ckpt" 

    checkpoint_name = os.path.join(PREFIX, dataset, CKPT_DIR, ckpt)

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

    test_dataset = SingleClassDataset(X_test, y_class_test, image_ids=test_data['image_id'].tolist())
    num_classes=2
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_alias == 'swintransformer':
        model = SwinTransformer3DClassifier.load_from_checkpoint(
                                            checkpoint_path=checkpoint_name,
                                            img_size=(256, 256, 256),
                                            in_channels=in_channels,
                                            num_classes=num_classes,
                                            use_checkpoint=True,
                                            )
        target_layer = model.swin_transformer.layers4c[0]
    else:
        model = CNN3DClassifier.load_from_checkpoint(checkpoint_path=checkpoint_name, 
                                                    model_alias=model_alias, 
                                                    in_channels=in_channels,
                                                    num_classes=num_classes,
                                            )
        assert 'resnet' in model_name # only works for resnet, adjust otherwise
        target_layer = model.model.layer4[-1]

    print(f"Using target layer: {target_layer}")

    model.eval()
    OUTPUT_DIR = os.path.join(PREFIX, dataset, EXPLAIN_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if check_stability:
        with open(json_file, 'r') as f:
            results = json.load(f)

    for idx, batch in tqdm(enumerate(test_dataloader)):
        model.to(device)
        inputs = batch[0].to(device)
        labels = batch[1]
        assert len(batch) == 3
        image_id = batch[2][0]
        savep = os.path.join(OUTPUT_DIR, f"{image_id}_{model_name}_{baseline}_{attribute}_GradCAM_seed_{seed}{biased_string}.npz")

        with torch.no_grad():
            outputs = model(inputs).argmax(dim=1).item()

        if check_stability:
            if f"{image_id}_{model_name}_{dataset}_GradCAM" in results.keys(): # stats already exist
                print(f"{image_id}_{model_name}_{dataset}_GradCAM exists")
                continue

            print(f"The prediction and labels are {outputs}, {labels.item()}")
            attribution_handle =LayerGradCam(model, target_layer)
            attribution_map = torch.relu(attribution_handle.attribute(inputs, target=outputs))
            saliency_down = F.adaptive_avg_pool3d(attribution_map, (16, 16, 16))
            model.to(device) # get model back to GPU for inference
            k = 512 # 512 patches -- 12.5%
            flat = saliency_down.flatten()
            topk = flat.topk(k) # 512 patches for switiching on initially
            selected_features = torch.zeros_like(flat, dtype=torch.long)
            selected_features[topk.indices] = 1

            foreground_mask = get_fg_mask_3d(volume=inputs[0][0].cpu().numpy())
            foreground_mask_flat = get_fg_mask_flat(foreground_mask)
            selected_features_bg_fixed = selected_features.clone()
            selected_features_bg_fixed[foreground_mask_flat == 0] = 1 # this should be worked with
            
            m = int((selected_features_bg_fixed.numel() - selected_features_bg_fixed.sum()).item()) # rest of the foreground patches
            results[f"{image_id}_{model_name}_{dataset}_GradCAM"] = {}
            start = 0
            end = 101 if m+1 > 100 else m+1 # 100 max radius
            
            if end < 20:
                radii = list(range(start, end, 2))
            else:
                radii = list(range(start, 20, 2))
                radii += list(range((end//2)+1, end, 20)) # higher steps sizes
            try:
                for r in tqdm(radii):
                    stab_rate_r = estimate_stability_rate(model, inputs, selected_features_bg_fixed.to(device), radius=r)
                    print(f"Estimated stability for perturbations of size <= {r}: {stab_rate_r.item():.4f}")
                    results[f"{image_id}_{model_name}_{dataset}_GradCAM"][r] = stab_rate_r.item()
            except Exception as e:
                print(e) # for outliers
                continue
        else:
            attribution_handle = LayerGradCam(model, target_layer)
            attribution_map = torch.relu(attribution_handle.attribute(inputs, target=outputs))
            attribution = LayerAttribution.interpolate(attribution_map.to(device), (256,256,256), interpolate_mode='trilinear')
            np.savez_compressed(savep, array=attribution.detach().cpu().numpy())

        if check_stability: # save every record after processing
            with open(json_file, 'w') as f:
                json.dump(results, f)
        

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="This is a script for attribution map computation")
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    parser.add_argument("--model", type=str, default="efficientnetb0")
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--stability", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--json", type=str, default="explain.json")
    parser.add_argument("--baseline", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--attribute", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bias-samples-percent", type=float, default=None)
    # Parse the arguments
    args = parser.parse_args()

    main(args)
