import os
import pickle
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import argparse
import logging

logging.basicConfig(level=logging.INFO)

# define variables, metadata ,etc
load_dotenv('.env')
PREFIX = os.getenv("PREFIX")
assert PREFIX, "Set PREFIX in .env"

results_1 = os.listdir(os.path.join(PREFIX, 'celeba_gender/results/'))
results_2 = os.listdir(os.path.join(PREFIX, 'chexpert_pleuraleffusiongender/results/'))
results = [os.path.join(os.path.join(PREFIX, 'chexpert_pleuraleffusiongender/results/'), s) for s in results_2 if s.startswith('chexpert')]
results += [os.path.join(os.path.join(PREFIX, 'celeba_gender/results/'), s) for s in results_1 if s.startswith('celeba')]


corrmap = {
    'chexpert_pleuraleffusiongender_inception_True_True': {
          'chexpert_pleuraleffusiongender_inception_True_None': 'chexpert_pleuraleffusiongender_inception_None_None'
    },
     'chexpert_pleuraleffusiongender_resnet_True_True': {
          'chexpert_pleuraleffusiongender_resnet_True_None': 'chexpert_pleuraleffusiongender_resnet_None_None'
    },
    'chexpert_pleuraleffusiongender_vgg_True_True': {
          'chexpert_pleuraleffusiongender_vgg_True_None': 'chexpert_pleuraleffusiongender_vgg_None_None'
    },
    'chexpert_pleuraleffusiongender_mobilenet_True_True': {
          'chexpert_pleuraleffusiongender_mobilenet_True_None': 'chexpert_pleuraleffusiongender_mobilenet_None_None'
    },
    'chexpert_pleuraleffusiongender_vit_True_True': {
          'chexpert_pleuraleffusiongender_vit_True_None': 'chexpert_pleuraleffusiongender_vit_None_None'
    },
    'celeba_gender_inception_True_True': {
          'celeba_gender_inception_True_None': 'celeba_gender_inception_None_None'
    },
    'celeba_gender_resnet_True_True': {
          'celeba_gender_resnet_True_None': 'celeba_gender_resnet_None_None'
    },
    'celeba_gender_vgg_True_True': {
          'celeba_gender_vgg_True_None': 'celeba_gender_vgg_None_None'
    },
    'celeba_gender_mobilenet_True_True': {
          'celeba_gender_mobilenet_True_None': 'celeba_gender_mobilenet_None_None'
    },
    'celeba_gender_vit_True_True': {
          'celeba_gender_vit_True_None': 'celeba_gender_vit_None_None'
    },
}


nums = {}
for s in results:
      nums[s.split('/')[-1].replace('.pkl','')] = s


### correlation related definitions

# non-parametric tests


# pairwise correlation
def pairwise_correlation(x, y, B=10000, stat_only=False):
    r = np.corrcoef(x, y)[0,1]
    if stat_only:
        return r
    rng = np.random.default_rng(seed=42)
    sims = np.array([np.corrcoef(rng.permutation(x), y)[0,1] for _ in range(B)])
    p = (1 + np.sum(np.abs(sims) >= np.abs(r))) / (B + 1)
    return r, p # returns p-val

# partial correlation
def _residualise(x1: np.array , x2: np.array, x3: np.array):
    # concatente along columns
    # y ~ 1 + x
    X = np.c_[np.ones(len(x3)), x3]

    # residuals after regressing out x3
    b1 = np.linalg.lstsq(X, x1, rcond=None)[0]
    b2 = np.linalg.lstsq(X, x2, rcond=None)[0]
    r1 = x1 - X @ b1
    r2 = x2 - X @ b2
    return r1, r2

def partial_correlation(x1, x2, x3, B=10000, stat_only=False):
    r1, r2 = _residualise(x1=x1, x2=x2, x3=x3)
    r = np.corrcoef(r1, r2)[0,1]
    if stat_only:
        return r
    rng = np.random.default_rng(seed=42)
    sims = np.array([np.corrcoef(rng.permutation(r1), r2)[0,1] for _ in range(B)])
    p = (1 + np.sum(np.abs(sims) >= np.abs(r))) / (B + 1) # two-sided tests
    return r, p

def semipartial_correlation(x1, x2, x3, B=10000, stat_only=False):
    X = np.c_[np.ones(len(x1)), np.asarray(x3).reshape(len(x1), -1)]
    rx = x1 - X @ np.linalg.lstsq(X, x1, rcond=None)[0]
    r = np.corrcoef(rx, x2)[0,1]
    if stat_only:
        return r
    rng = np.random.default_rng(seed=42)
    sims = np.array([np.corrcoef(rng.permutation(rx), x2)[0,1] for _ in range(B)])
    p = (1 + (np.abs(sims) >= np.abs(r)).sum()) / (B + 1) # two-sided tests
    return r, p

# can be used if the image level bootstrap is required
def image_bootstrap_ci(A_imgs, B_imgs, C_imgs, corr_func, B=2000, agg="median", seed=42):
    rng = np.random.default_rng(seed)
    m = A_imgs.shape[0]
    if agg == "median":
        agg_fn = np.median
    elif agg == "mean":
        agg_fn = np.mean
    else:
        raise ValueError("agg must be 'median' or 'mean'")

    stats = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, m, m)  # resample images with replacement
        A_agg = agg_fn(A_imgs[idx], axis=0)  # aggregate across resampled images
        B_agg = agg_fn(B_imgs[idx], axis=0)
        C_agg = agg_fn(C_imgs[idx], axis=0)
        stats[b] = corr_func(A_agg, B_agg, C_agg, stat_only=True)  # compute stat across regions
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return lo, hi


# region level statistics
def residualise(y, x):
    y = np.asarray(y).reshape(-1)
    X = np.asarray(x)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X = np.c_[np.ones(len(X)), X]
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta

def per_region_pcs(R_BI, R_BA, R_PA):
    # Partial residuals
    u = residualise(R_BI, R_BA)
    v = residualise(R_PA, R_BA)
    # Standardize
    u = (u - u.mean()) / (u.std(ddof=1) + 1e-12)
    v = (v - v.mean()) / (v.std(ddof=1) + 1e-12)
    pcs = u * v
    return pcs

def main(args):
    global corrmap
    PROJECT_ROOT = os.getenv('PROJECTDIR')
    outfile = args.outfile
    samples = args.samples
    assert samples in [True, None], "Samples is an experimental setting, true or None"
    seeds = args.seeds
    assert seeds in list(range(1, 101)), "Seeds can't be less than 1 and more than 100"

    aggrank = args.aggrank

    if aggrank:
        agg_string = '_spearman'
    else:
        agg_string = ''
    
    saliency = args.saliency

    if saliency:
        sal_string = '_saliency'
    else:
        sal_string = ''
    
    assert not (saliency and aggrank), "can't be true together"

    attenuation = args.attenuation

    assert not (attenuation and samples), "can't do subset evals for samples exp"

    if attenuation:
        assert attenuation in [1,2,3,4] # test set 1 or 2
        attenuation_string = f'_attenuation_{attenuation}'
    else:
        attenuation_string = ''

    pcs_rows = []
    corrs = []
    attribution_methods = ['GradCAM', 'LRP']
    partition_methods = ['grid', 'superpixel']
    setups = ['nshort', 'ushort']
    regions = [64, 256]
    if samples:
        bias_samples_train=[25, 500, 1000, 1500, 2000]
        bias_samples_val=[10, 40, 60, 80, 100]
    else:
        bias_samples_train=[25]
        bias_samples_val=[10]
    
    if seeds > 1:
        # only valid for celeba
        corrmap = {k:v for k,v in corrmap.items() if 'celeba' in k}
        assert samples == None, "samples have to be false here"

    for i in range(seeds):
        for attribution_method in attribution_methods:
            for setup in setups:
                for partition in partition_methods:
                    for region in regions:
                        for bsam_tr, bsam_val in zip(bias_samples_train, bias_samples_val):
                            for k, v in corrmap.items():
                                attribute_key = k + f'_{attribution_method}__partition_{partition}_regions_{region}_seed_{i}{agg_string}{sal_string}{attenuation_string}'
                                try:
                                    with open(nums[attribute_key], 'rb') as handle:
                                        SA = np.array(pickle.load(handle))
                                except Exception as e:
                                    continue
                                
                                dataset = f"{'_'.join(nums[attribute_key].split('/')[-1].split('_')[:2])}"
                                model_name = f"{nums[attribute_key].split('/')[-1].split('_')[2]}"
                                logging.info(f"model: {model_name}, partition: {partition}, dataset: {dataset}, setup: {setup}, attr: {attribution_method}, region: {region}, train: {bsam_tr}")

                                for a,b in v.items():
                                    base_key = a + f'_{attribution_method}__partition_{partition}_regions_{region}_seed_{i}{agg_string}{sal_string}{attenuation_string}'
                                    with open(nums[base_key], 'rb') as handle:
                                        BA = np.array(pickle.load(handle))
                                    
                                    if bsam_tr == 25:
                                        biased_key = b + f'_{attribution_method}__partition_{partition}_regions_{region}_seed_{i}{agg_string}{sal_string}{attenuation_string}'
                                    else:
                                        biased_key = b + f'_{attribution_method}__partition_{partition}_regions_{region}_seed_{i}_{bsam_tr}_{bsam_val}{agg_string}{sal_string}'

                                    with open(nums[biased_key], 'rb') as handle:
                                        TS = np.array(pickle.load(handle))

                                    if setup == 'nshort':
                                        R2 = np.median(BA, axis=0)
                                        R1 = np.median(TS, axis=0)
                                        R3 = np.median(SA, axis=0)
                                    else:
                                        R2 = np.median(SA, axis=0)
                                        R1 = np.median(TS, axis=0)
                                        R3 = np.median(BA, axis=0)
                                    
                                    logging.info(attribute_key)
                                    logging.info(biased_key)
                                    logging.info(base_key)

                                    assert region == len(R1) == len(R2) == len(R3)
                                    
                                    pcs_obs = per_region_pcs(R_TS=R1, R_BA=R2, R_SA=R3)
                                    for rid, pcs in zip(range(len(pcs_obs)), pcs_obs):
                                        pcs_rows.append({
                                            "region_id": rid,
                                            "attribution_method": attribution_method,
                                            "pcs": float(pcs),
                                            "dataset": dataset,
                                            "model": model_name,
                                            "partition": partition,
                                            "n_regions": int(region),
                                            "seed": int(i),
                                            "setup": setup,
                                            "train_anticorrelated": int(bsam_tr),
                                            "val_anticorrelated": int(bsam_val),
                                        })

                                    logging.info(F"partition: {partition}, setup: {setup}")
                                    
                                    rho, pval = pairwise_correlation(R1, R3)
                                    corrs.append({"Iteration": i,
                                                "attribution_method": attribution_method,
                                                "shortcut type": setup, 
                                                "Partition Method": f'{partition}_{region}', 
                                                "Method": "Pearson(TS-SA)", 
                                                "Correlation": rho, 
                                                "pval": pval, 
                                                "Model": model_name,
                                                "ci_low": np.nan,
                                                "ci_high": np.nan, 
                                                "dataset": dataset, 
                                                "train_anticorrelated": bsam_tr})
                                    
                                    rho, pval = pairwise_correlation(R1, R2)
                                    corrs.append({"Iteration": i,
                                                "attribution_method": attribution_method,
                                                "shortcut type": setup, 
                                                "Partition Method": f'{partition}_{region}', 
                                                "Method": "Pearson(TS-BA)", 
                                                "Correlation": rho, 
                                                "pval": pval, 
                                                "Model": model_name,
                                                "ci_low": np.nan,
                                                "ci_high": np.nan,
                                                "dataset": dataset, 
                                                "train_anticorrelated": bsam_tr})

                                    rho, pval = pairwise_correlation(R3, R2)
                                    corrs.append({"Iteration": i,
                                                "attribution_method": attribution_method,
                                                "shortcut type": setup, 
                                                "Partition Method": f'{partition}_{region}', 
                                                "Method": "Pearson(SA-BA)", 
                                                "Correlation": rho, 
                                                "pval": pval, 
                                                "Model": model_name,
                                                "ci_low": np.nan,
                                                "ci_high": np.nan,
                                                "dataset": dataset, 
                                                "train_anticorrelated": bsam_tr})


                                    pcorr, p_pcorr = partial_correlation(x1=R1, x2=R3, x3=R2)
                                    if setup == 'nshort':
                                        lo, hi = image_bootstrap_ci(TS, SA, BA, corr_func=partial_correlation, B=10000)
                                    else:
                                        lo, hi = image_bootstrap_ci(TS, BA, SA, corr_func=partial_correlation, B=10000)

                                    
                                    corrs.append({
                                        "Iteration": i,
                                        "attribution_method": attribution_method,
                                        "shortcut type": setup,
                                        "Partition Method": f'{partition}_{region}',
                                        "Method": "Partial(TS<->SA|BA)",
                                        "Correlation": pcorr,
                                        "pval": p_pcorr,
                                        "ci_low": lo,
                                        "ci_high": hi,
                                        "Model": model_name, 
                                        "dataset": dataset,
                                        "train_anticorrelated": bsam_tr
                                    })

                                    semi_pcorr, p_semi_pcorr = semipartial_correlation(x1=R1, x2=R3, x3=R2)
                                    if setup == 'nshort':
                                        lo, hi = image_bootstrap_ci(TS, SA, BA, corr_func=semipartial_correlation, B=10000)
                                    else:
                                        lo, hi = image_bootstrap_ci(TS, BA, SA, corr_func=semipartial_correlation, B=10000)

                                    corrs.append({"Iteration": i, 
                                                "attribution_method": attribution_method, "shortcut type": setup, "Partition Method": f'{partition}_{region}', "Method": "Permutation((BI-BA)<->(SA))",
                                                "Correlation": semi_pcorr, "pval": p_semi_pcorr, "Model": model_name, "dataset": dataset, "train_anticorrelated": bsam_tr})

    corrs = pd.DataFrame(corrs)
    corrs.round(6).to_csv(f'{PROJECT_ROOT}/results/corrs_{outfile}.csv', index=False)
    pcs_df = pd.DataFrame(pcs_rows)
    pcs_df.round(6).to_csv(f"{PROJECT_ROOT}/results/pcs_all_configs_{outfile}.csv", index=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Compute correlation metrics on 3D datasets")
    parser.add_argument("--outfile", type=str)
    parser.add_argument("--samples", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seeds", type=int, default=0)
    parser.add_argument("--aggrank", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--saliency", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--attenuation", type=int, default=0)
    # Parse the arguments
    args = parser.parse_args()
    # Call main with parsed arguments
    main(args)