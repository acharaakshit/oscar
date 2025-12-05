import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import CelebA as _CelebA
import torchvision.transforms as T
import pandas as pd
from PIL import Image

def simple_take(arr, k):
    arr = np.asarray(arr)
    if k <= 0 or arr.size == 0:
        return np.empty(0, dtype=int)
    arr = np.sort(arr)
    return arr[:min(k, arr.size)].astype(int)

def remove_from_groups(pool, picked):
    if picked is None or len(picked) == 0:
        return pool
    picked = np.asarray(picked)
    for k, arr in pool.items():
        if len(arr):
            pool[k] = arr[~np.isin(arr, picked)]
    
    return pool

def sample_biased(groups, n_samples, split="train",
                  anti_per_label_train=25, anti_per_label_val=10):
    chosen = []
    per_side = n_samples // 2
    for y_val in (0, 1):
        corr_attr = 1 - y_val

        # anti quota for this split (cap by availability)
        wanted_anti = anti_per_label_train if split == "train" else anti_per_label_val
        avail_anti  = len(groups[(y_val, 1 - corr_attr)])
        if wanted_anti > avail_anti:
            print(f"number of anti-correlated samples aren't available, selecting {avail_anti}")
            n_anti = avail_anti - anti_per_label_val # to keep the val quota
        else:
            n_anti = wanted_anti

        # correlated gets the rest of the per-side budget
        avail_corr = len(groups[(y_val, corr_attr)])
        n_corr = min(max(0, per_side - n_anti), avail_corr)

        corr_pick = simple_take(groups[(y_val, corr_attr)], n_corr)
        anti_pick = simple_take(groups[(y_val, 1 - corr_attr)], n_anti)

        chosen.extend(corr_pick)
        chosen.extend(anti_pick)
    return np.array(chosen, dtype=int)

def sample_balanced_disjoint(groups, train_n, val_n, test_n, reserve_eval_first=True):
    # make a working copy and sort each group once for determinism
    pool = {k: np.sort(v) for k, v in groups.items()}

    def take(per_group):
        picked = []
        for y_val in (0, 1):
            for a_val in (0, 1):
                arr = pool[(y_val, a_val)]
                n = min(per_group, len(arr))
                sel = arr[:n]
                picked.extend(sel)
                pool[(y_val, a_val)] = arr[n:]  # remove taken deterministically
        return np.asarray(picked, dtype=int)

    pt, pv, pte = train_n // 4, val_n // 4, test_n // 4
    if reserve_eval_first:
        val_sel   = take(pv)
        test_sel  = take(pte)
        train_sel = take(pt)
    else:
        train_sel = take(pt)
        val_sel   = take(pv)
        test_sel  = take(pte)

    return train_sel, val_sel, test_sel

def sample_balanced(groups, n_samples):
    chosen = []
    per_group = n_samples // 4
    for y_val in (0, 1):
        for a_val in (0, 1):
            pick = simple_take(groups[(y_val, a_val)], per_group)
            chosen.extend(pick)
    return np.array(chosen, dtype=int)

def group_indices(y, a):
    idxs = np.arange(len(y))
    # split based on subgroups
    return {
        (0,0): idxs[(y==0) & (a==0)],
        (0,1): idxs[(y==0) & (a==1)],
        (1,0): idxs[(y==1) & (a==0)],
        (1,1): idxs[(y==1) & (a==1)],
    }

class BiasedCelebADataset(Dataset):
    def __init__(self,
                 base_dataset: _CelebA,
                 indices: np.ndarray,
                 task_labels: np.ndarray,
                 attribute_labels: np.ndarray,
                 attr_labs: bool,
                 attribute_name: str,
                 label_name: str,
                ):
        self.base = base_dataset # celeba is the main dataset
        self.indices = indices # indices to be used from celeba
        self.labels = torch.from_numpy(task_labels).long()
        self.attr_labs = attr_labs # True for attributes and False for labels
        self.attribute_name = attribute_name
        self.label_name = label_name
        self.attributes = torch.from_numpy(attribute_labels).long()
        self.pre_transform = T.Compose([
            T.CenterCrop(178), # use the agreed standard
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

        self.transform = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        img, _ = self.base[idx]
        img = self.pre_transform(img)
        img = self.transform(img)
        
        return (img, self.attributes[i]) if self.attr_labs else (img, self.labels[i])

    def get_num_classes(self):
        return 2

def get_biased_celeba_splits(
        root: str,
        label: str,
        attribute: str,
        balanced: bool,
        attr_labs: bool, # whether to return attributes or labels
        train_samples: int = 5000,
        test_samples: int = 5000,
        val_samples: int = 1000,
        bias_samples_train: int = None,
        bias_samples_val: int = None,
    ) -> Dataset:

    # check if the directory exists
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset not found at  {root!r} !")

    # get the train split
    train_full = _CelebA(root, split="train", target_type="attr", download=True)
    val_full   = _CelebA(root, split="valid", target_type="attr", download=True)
    test_full = _CelebA(root, split="test", target_type="attr", download=True)

    full_dataset = ConcatDataset([train_full, val_full, test_full])

    all_attributes = np.vstack([
        (train_full.attr.numpy() + 1) // 2,
        (val_full.attr.numpy()   + 1) // 2,
        (test_full.attr.numpy()  + 1) // 2,
    ])

    name2i = {n:i for i,n in enumerate(train_full.attr_names)}
    if label not in name2i or attribute not in name2i:
        raise ValueError(f"Label {label!r} or attribute {attribute!r} not in CelebA attributes!")

    full_labels = all_attributes[:, name2i[label]]
    full_attributes = all_attributes[:, name2i[attribute]]
    groups = group_indices(y=full_labels, a=full_attributes)

    _, _, test_idxs = sample_balanced_disjoint(groups, 0, 0, test_samples, reserve_eval_first=True)

    groups = remove_from_groups(groups, test_idxs)



    if balanced:
        train_idxs, val_idxs, _ = sample_balanced_disjoint(
            groups, train_samples, val_samples, 0, reserve_eval_first=True
        ) # not passing the test samples here
    else:
        #  if bias samples are provided, then should be used
        if bias_samples_train and bias_samples_val:
            assert bias_samples_train < train_samples, "bias samples can't be larger than train samples"
            assert bias_samples_val < val_samples, "bias samples can't be larger than val samples"
        else:
            bias_samples_train = 25
            bias_samples_val = 10

        train_idxs = sample_biased(groups, train_samples, split="train", 
                                    anti_per_label_train=bias_samples_train, anti_per_label_val=bias_samples_val)
        groups = remove_from_groups(groups, train_idxs)
        val_idxs = sample_biased(groups, val_samples, split="validation", 
                                anti_per_label_train=bias_samples_train, anti_per_label_val=bias_samples_val)
        groups = remove_from_groups(groups, val_idxs)

        # test ids are already computed so no need to do that here
    
    assert len(set(train_idxs) & set(val_idxs)) == 0
    assert len(set(train_idxs) & set(test_idxs)) == 0
    assert len(set(val_idxs) & set(test_idxs)) == 0
    
    train_dataset = BiasedCelebADataset(
        base_dataset=full_dataset,
        indices=train_idxs,
        task_labels=full_labels[train_idxs],
        attribute_labels=full_attributes[train_idxs],
        attr_labs=attr_labs,
        attribute_name=attribute,
        label_name=label,
    )

    val_dataset = BiasedCelebADataset(
        base_dataset=full_dataset,
        indices=val_idxs,
        task_labels=full_labels[val_idxs],
        attribute_labels=full_attributes[val_idxs],
        attr_labs=attr_labs,
        attribute_name=attribute,
        label_name=label,
    )

    test_dataset = BiasedCelebADataset(
        base_dataset=full_dataset,
        indices=test_idxs,
        task_labels=full_labels[test_idxs],
        attribute_labels=full_attributes[test_idxs],
        attr_labs=attr_labs,
        attribute_name=attribute,
        label_name=label,
    )
    
    return train_dataset, val_dataset, test_dataset

class CheXpertShortcutDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            root: str,
            label: str,
            attribute: str,
            attr_labs: bool,
            attribute_name: str,
            label_name: str,
        ):
        self.df = df
        self.root = root
        self.attr_labs = attr_labs

        self.label = label
        self.attribute = attribute
        self.labels = torch.from_numpy(self.df[label].to_numpy()).long() # uint8
        self.attributes = torch.from_numpy(self.df[attribute].to_numpy()).long() # uint8
        self.attribute_name = attribute_name
        self.label_name = label_name
        self.pre_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root, row["Path"])
        image = Image.open(img_path).convert("L")
        image = self.pre_transform(image) 
        image = (image - image.mean()) / image.std() # per image z-score
        image = image.repeat(3, 1, 1)
        
        label = torch.tensor(int(row[self.label])).long()
        attr  = torch.tensor(int(row[self.attribute])).long()

        if self.attr_labs:
            return image, attr 
        
        return image, label

def get_biased_chexpert_splits(
        root: str,
        label: str,
        attribute: str,
        balanced: bool,
        train_samples: int = 5000,
        val_samples: int = 1000,
        test_samples: int = 2000,
        bias_samples_train: int = None,
        bias_samples_val: int = None,
        attr_labs: bool = False,
    ):

    # load and clean dataframe
    df = pd.read_csv(os.path.join(root, 'CheXpert-v1.0/train.csv')).dropna(subset=[label, attribute, "Path", "Age", "Sex"]) # selecting train for all the sets
    # remove uncertain columns
    df = df[(df[label] != -1) & (df[attribute] != -1)]
    df = df[(df['AP/PA'] == 'AP') & (df['Frontal/Lateral']  == 'Frontal')]

    # subject level leakage should be avoided
    df["patient_id"] = df["Path"].str.extract(r"(patient\d+)")
    pat = df.groupby("patient_id")[["Sex", "Age"]].first().reset_index()
    pat["age_group"] = np.where(pat["Age"] < 45, "young", "old")

    # stratify only by AGE if the chosen attribute is Sex, else by Sex x age_group
    if attribute == "Sex":
        df[attribute] = df[attribute].map({"Female": 0, "Male": 1}).astype(int) # binarise as it will be used as attribute
        pat["strat_key"] = pat["age_group"]
    else:
        pat["strat_key"] = pat["Sex"].astype(str) + "_" + pat["age_group"].astype(str)

    total_req = int(train_samples + val_samples + test_samples)
    p_train = train_samples / total_req
    p_val   = val_samples   / total_req
    p_test  = test_samples  / total_req

    def split_patients(pat_df, p_train, p_val, p_test, reserve_eval_first=True):
        tr, va, te = [], [], []
        for _, grp in pat_df.sort_values("patient_id").groupby("strat_key"):
            ids = grp["patient_id"].values
            n = len(ids)
            nv = int(np.floor(n * p_val))
            nt = int(np.floor(n * p_test))
            if reserve_eval_first:
                va.extend(ids[:nv])
                te.extend(ids[nv:nv+nt])
                tr.extend(ids[nv+nt:])
            else:
                ntr = int(np.floor(n * p_train))
                tr.extend(ids[:ntr])
                va.extend(ids[ntr:ntr+nv])
                te.extend(ids[ntr+nv:])
        return np.array(tr), np.array(va), np.array(te)


    train_p, val_p, test_p = split_patients(
        pat, p_train, p_val, p_test, reserve_eval_first=True
    )

    def subset_by_patients(patients):
        return df[df["patient_id"].isin(patients)]

    def make_groups(df_sub):
            y_sub = df_sub[label].astype(int).to_numpy()
            a_sub = df_sub[attribute].astype(int).to_numpy()
            return group_indices(y_sub, a_sub)

    df_train, df_val, df_test = subset_by_patients(train_p), subset_by_patients(val_p), subset_by_patients(test_p)

    # make subgroups
    train_groups = make_groups(df_train)
    val_groups   = make_groups(df_val)
    test_groups  = make_groups(df_test)

    if balanced:
        train_idxs = sample_balanced(train_groups, train_samples)
        val_idxs   = sample_balanced(val_groups,   val_samples)
        test_idxs  = sample_balanced(test_groups,  test_samples)
    else:
        #  if bias samples are provided, then should be used
        if bias_samples_train and bias_samples_val:
            assert bias_samples_train < train_samples, "bias samples can't be larger than train samples"
            assert bias_samples_val < val_samples, "bias samples can't be larger than val samples"
        else:
            bias_samples_train = 25
            bias_samples_val = 10

        train_idxs = sample_biased(train_groups, train_samples, split="train", 
                                anti_per_label_train= bias_samples_train,
                                anti_per_label_val=bias_samples_val)
        val_idxs   = sample_biased(val_groups,   val_samples,   split="validation",
                                anti_per_label_train= bias_samples_train,
                                anti_per_label_val=bias_samples_val)

        test_idxs  = sample_balanced(test_groups, test_samples)
    
    train_paths = set(df_train.iloc[train_idxs]["Path"])
    val_paths   = set(df_val.iloc[val_idxs]["Path"])
    test_paths  = set(df_test.iloc[test_idxs]["Path"])

    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)

    train_pats = set(df_train["patient_id"])
    val_pats   = set(df_val["patient_id"])
    test_pats  = set(df_test["patient_id"])
    assert train_pats.isdisjoint(val_pats)
    assert train_pats.isdisjoint(test_pats)
    assert val_pats.isdisjoint(test_pats)

    # subset
    train_df = df_train.iloc[train_idxs]
    val_df   = df_val.iloc[val_idxs]
    test_df  = df_test.iloc[test_idxs]

    # wrap into datasets
    train_set = CheXpertShortcutDataset(train_df, root=root, label=label, attribute=attribute, attr_labs=attr_labs, attribute_name=attribute, label_name=label,)
    val_set   = CheXpertShortcutDataset(val_df, root=root, label=label, attribute=attribute, attr_labs=attr_labs, attribute_name=attribute, label_name=label,)
    test_set  = CheXpertShortcutDataset(test_df, root=root, label=label, attribute=attribute, attr_labs=attr_labs, attribute_name=attribute, label_name=label,)

    return train_set, val_set, test_set