import torch
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def per_group_acc(y, a, yhat):
        # y: 0=non‑blond, 1=blond; a: 0=female, 1=male
        y, a, yhat = map(np.asarray, (y, a, yhat))
        groups = { (0,0):"y=0,a=0", (0,1):"y=0,a=1", (1,0):"y=1,a=0", (1,1):"y=1,a=1" }
        acc = {}
        for (yl, al), name in groups.items():
            idx = np.where((y==yl) & (a==al))[0]
            acc[name] = (yhat[idx]==y[idx]).mean() if len(idx) else np.nan
        wga = np.nanmin(list(acc.values()))
        return acc, wga

def build_rcs_mask_from_npy(npy_path: str):
    rcs = np.load(npy_path).astype(np.float32)
    logging.info(rcs.shape)
    assert rcs.ndim == 2, "RCS npy must be 2D grid"
    t = torch.from_numpy(rcs).unsqueeze(0).unsqueeze(0)
    return t

def split_summaries(splits, model_name="default"):
    all_tables = []
    for split_name, dataset in splits:
        labels = dataset.labels.numpy()
        attrs  = dataset.attributes.numpy()

        label_map = {0: f"No {dataset.label_name}", 1: dataset.label_name}
        attr_map  = {0: f"No {dataset.attribute_name}", 1: dataset.attribute_name}
        labels_named = pd.Series(labels).map(label_map)
        attrs_named  = pd.Series(attrs).map(attr_map)

        table = pd.crosstab(
            labels_named, attrs_named,
            rownames=[dataset.label_name], colnames=[dataset.attribute_name]
        )

        table = table.reset_index().melt(
            id_vars=[dataset.label_name],
            var_name=dataset.attribute_name,
            value_name="Count"
        )

        table["Split"] = split_name
        table["Model"] = model_name
        all_tables.append(table)

    combined = pd.concat(all_tables, ignore_index=True)
    logging.info(combined)