import numpy as np
from scipy.ndimage import sum as ndsum

def saliency_score_percent(sal_map, atlas, threshold_prop=0.5):
    cutoff = np.percentile(sal_map[np.isfinite(sal_map)], threshold_prop * 100)
    bin_map = (sal_map >= cutoff).astype(float) # keep top 50%
    region_ids = np.unique(atlas)
    total_vox = ndsum(np.ones_like(atlas), labels=atlas, index=region_ids)
    salient_vox = ndsum(bin_map, labels=atlas, index=region_ids)
    scores = np.divide(salient_vox * 100, total_vox, out=np.zeros_like(salient_vox), where=total_vox > 0)
    return region_ids, scores