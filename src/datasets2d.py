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


class WaterbirdsShortcutDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            root: str,
            label: str,
            attribute: str,
            path_column: str,
            attr_labs: bool,
            attribute_name: str,
            label_name: str,
        ):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.label = label
        self.attribute = attribute
        self.path_column = path_column
        self.attr_labs = attr_labs
        self.attribute_name = attribute_name
        self.label_name = label_name
        self.labels = torch.from_numpy(self.df[label].to_numpy()).long()
        self.attributes = torch.from_numpy(self.df[attribute].to_numpy()).long()
        self.pre_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])
        self.transform = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def _image_path(self, relative_path):
        return os.path.join(self.root, relative_path)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self._image_path(row[self.path_column])).convert("RGB")
        image = self.pre_transform(image)
        image = self.transform(image)
        label = torch.tensor(int(row[self.label])).long()
        attr = torch.tensor(int(row[self.attribute])).long()
        return (image, attr) if self.attr_labs else (image, label)

    def get_num_classes(self):
        return 2

def get_biased_waterbirds_splits(
        root: str,
        label: str = "y",
        attribute: str = "place",
        balanced: bool = False,
        train_samples: int = 5000,
        val_samples: int = 1000,
        test_samples: int = 2000,
        bias_samples_train: int = None,
        bias_samples_val: int = None,
        attr_labs: bool = False,
    ):
    if not os.path.isfile(os.path.join(root, "metadata.csv")):
        raise FileNotFoundError(f"Waterbirds metadata.csv not found at {root!r}")
    df = pd.read_csv(os.path.join(root, "metadata.csv"))
    path_column = "img_filename"

    df = df.dropna(subset=[path_column, label, attribute]).copy()
    df[label] = df[label].astype(int)
    df[attribute] = df[attribute].astype(int)

    full_labels = df[label].to_numpy()
    full_attributes = df[attribute].to_numpy()
    groups = group_indices(full_labels, full_attributes)

    _, _, test_idxs = sample_balanced_disjoint(groups, 0, 0, test_samples, reserve_eval_first=True)
    groups = remove_from_groups(groups, test_idxs)

    if balanced:
        train_idxs, val_idxs, _ = sample_balanced_disjoint(
            groups, train_samples, val_samples, 0, reserve_eval_first=True
        )
    else:
        if bias_samples_train and bias_samples_val:
            assert bias_samples_train < train_samples, "bias samples can't be larger than train samples"
            assert bias_samples_val < val_samples, "bias samples can't be larger than val samples"
        else:
            bias_samples_train = 25
            bias_samples_val = 10

        sampling_groups = group_indices(full_labels, 1 - full_attributes)
        sampling_groups = remove_from_groups(sampling_groups, test_idxs)
        train_idxs = sample_biased(
            sampling_groups,
            train_samples,
            split="train",
            anti_per_label_train=bias_samples_train,
            anti_per_label_val=bias_samples_val,
        )
        sampling_groups = remove_from_groups(sampling_groups, train_idxs)
        val_idxs = sample_biased(
            sampling_groups,
            val_samples,
            split="validation",
            anti_per_label_train=bias_samples_train,
            anti_per_label_val=bias_samples_val,
        )

    train_df = df.iloc[train_idxs].copy()
    val_df = df.iloc[val_idxs].copy()
    test_df = df.iloc[test_idxs].copy()

    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise ValueError("Waterbirds metadata did not produce non-empty train/val/test splits.")

    return (
        WaterbirdsShortcutDataset(
            train_df, root=root, label=label, attribute=attribute, path_column=path_column,
            attr_labs=attr_labs, attribute_name="Water background", label_name="Waterbird",
        ),
        WaterbirdsShortcutDataset(
            val_df, root=root, label=label, attribute=attribute, path_column=path_column,
            attr_labs=attr_labs, attribute_name="Water background", label_name="Waterbird",
        ),
        WaterbirdsShortcutDataset(
            test_df, root=root, label=label, attribute=attribute, path_column=path_column,
            attr_labs=attr_labs, attribute_name="Water background", label_name="Waterbird",
        ),
    )

class Camelyon17ShortcutDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            root: str,
            label: str,
            attribute: str,
            path_column: str,
            attr_labs: bool,
            attribute_name: str,
            label_name: str,
            augment: bool = False,
        ):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.label = label
        self.attribute = attribute
        self.path_column = path_column
        self.attr_labs = attr_labs
        self.attribute_name = attribute_name
        self.label_name = label_name
        self.labels = torch.from_numpy(self.df[label].to_numpy()).long()
        self.attributes = torch.from_numpy(self.df[attribute].to_numpy()).long()
        # should be always true during training and false for validation and testing
        if augment:
            self.pre_transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomApply([
                    T.RandomRotation(
                        degrees=15,
                        interpolation=T.InterpolationMode.BILINEAR,
                        fill=255,
                    ),
                ], p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.RandomGrayscale(p=0.1),
                T.Resize((224, 224)),
                T.ToTensor(),
            ])
        else:
            self.pre_transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
            ])
        self.transform = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def _image_path(self, relative_path):
        return os.path.join(self.root, relative_path)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self._image_path(row[self.path_column])).convert("RGB")
        image = self.pre_transform(image)
        image = self.transform(image)
        label = torch.tensor(int(row[self.label])).long()
        attr = torch.tensor(int(row[self.attribute])).long()
        return (image, attr) if self.attr_labs else (image, label)

    def get_num_classes(self):
        return 2

def _camelyon_patch_quality(path):
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    intensity = image.mean(axis=2)
    gray = intensity.astype(np.float32)
    laplacian = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return {
        "blur_score": float(np.var(laplacian)),
    }


def get_camelyon_splits(
        root: str,
        label: str = "tumor",
        attribute: str = "center",
        balanced: bool = False,
        train_samples: int = 5000,
        val_samples: int = 1000,
        test_samples: int = 2000,
        bias_samples_train: int = None,
        bias_samples_val: int = None,
        attr_labs: bool = False,
        train_augment: bool = False,
    ):
    """Build Camelyon17 splits while skipping obvious low-information patches."""
    metadata_path = os.path.join(root, "metadata.csv")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Camelyon17 metadata.csv not found at {root!r}")

    path_column = "Path"
    required_columns = [label, attribute, "patient", "node", "x_coord", "y_coord"]
    df = pd.read_csv(metadata_path).dropna(subset=required_columns).copy()

    center_to_attr = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1}
    df = df[df[label].isin([0, 1]) & df[attribute].isin(center_to_attr)].copy()
    df[label] = df[label].astype(int)
    df[attribute] = df[attribute].map(center_to_attr).astype(int)
    for column in ["patient", "node", "x_coord", "y_coord"]:
        df[column] = df[column].astype(int)

    df[path_column] = df.apply(
        lambda row: os.path.join(
            "patches",
            f"patient_{row['patient']:03d}_node_{row['node']}",
            f"patch_patient_{row['patient']:03d}_node_{row['node']}_x_{row['x_coord']}_y_{row['y_coord']}.png",
        ),
        axis=1,
    )

    patient_summary = (
        df.groupby("patient")
        .agg(
            center=(attribute, "first"),
            unique_centers=(attribute, "nunique"),
            y0=(label, lambda s: int((s == 0).sum())),
            y1=(label, lambda s: int((s == 1).sum())),
        )
        .reset_index()
    )
    if (patient_summary["unique_centers"] != 1).any():
        raise ValueError("Camelyon17 patient appears in more than one selected center.")

    def split_patients_by_center(pat_df):
        train_patients, val_patients, test_patients = [], [], []
        for _, center_patients in pat_df.sort_values("patient").groupby("center"):
            if len(center_patients) < 5:
                raise ValueError("Camelyon17 needs at least five patients per selected center group.")
            ordered = center_patients.assign(
                min_label_count=center_patients[["y0", "y1"]].min(axis=1)
            ).sort_values(["min_label_count", "patient"], ascending=[False, True])
            n_test = max(1, int(round(len(ordered) * 0.2)))
            n_val = max(1, int(round(len(ordered) * 0.1)))
            test_patients.extend(ordered.iloc[:n_test]["patient"].astype(int).tolist())
            val_patients.extend(ordered.iloc[n_test:n_test + n_val]["patient"].astype(int).tolist())
            train_patients.extend(ordered.iloc[n_test + n_val:]["patient"].astype(int).tolist())
        return (
            np.asarray(train_patients, dtype=int),
            np.asarray(val_patients, dtype=int),
            np.asarray(test_patients, dtype=int),
        )

    train_patients, val_patients, test_patients = split_patients_by_center(patient_summary)

    df_train = df[df["patient"].isin(train_patients)].reset_index(drop=True)
    df_val = df[df["patient"].isin(val_patients)].reset_index(drop=True)
    df_test = df[df["patient"].isin(test_patients)].reset_index(drop=True)
    quality_cache = {}

    CAMELYON_MIN_BLUR_SCORE = 7.5e-05

    def passes_quality(df_sub, idx):
        rel_path = df_sub.at[int(idx), path_column]
        if rel_path not in quality_cache:
            quality_cache[rel_path] = _camelyon_patch_quality(os.path.join(root, rel_path))
        metrics = quality_cache[rel_path]
        if metrics["blur_score"] < CAMELYON_MIN_BLUR_SCORE:
            return False
        return True

    def take_patient_round_robin(df_sub, y_val, a_val, n):
        group_df = df_sub[(df_sub[label] == y_val) & (df_sub[attribute] == a_val)]
        patient_buckets = [
            patient_df.sort_values(["node", "x_coord", "y_coord"]).index.to_numpy()
            for _, patient_df in group_df.groupby("patient", sort=True)
        ]
        picked = []
        offset = 0
        while len(picked) < n:
            progressed = False
            for bucket in patient_buckets:
                if offset < len(bucket):
                    progressed = True
                    idx = int(bucket[offset])
                    if passes_quality(df_sub, idx):
                        picked.append(idx)
                        if len(picked) == n:
                            break
            if not progressed:
                break
            offset += 1
        return np.asarray(picked, dtype=int)

    def sample_camelyon_balanced(df_sub, n_samples):
        picked = []
        per_group = n_samples // 4
        for y_val in (0, 1):
            for a_val in (0, 1):
                picked.extend(take_patient_round_robin(df_sub, y_val, a_val, per_group))
        return np.asarray(picked, dtype=int)

    def sample_camelyon_biased(df_sub, n_samples, split):
        picked = []
        per_side = n_samples // 2
        anti_per_label = bias_samples_train if split == "train" else bias_samples_val
        for y_val in (0, 1):
            corr_attr = 1 - y_val
            anti_attr = 1 - corr_attr
            n_anti = min(
                anti_per_label,
                len(df_sub[(df_sub[label] == y_val) & (df_sub[attribute] == anti_attr)]),
            )
            n_corr = max(0, per_side - n_anti)
            picked.extend(take_patient_round_robin(df_sub, y_val, corr_attr, n_corr))
            picked.extend(take_patient_round_robin(df_sub, y_val, anti_attr, n_anti))
        return np.asarray(picked, dtype=int)

    if balanced:
        train_idxs = sample_camelyon_balanced(df_train, train_samples)
        val_idxs = sample_camelyon_balanced(df_val, val_samples)
        test_idxs = sample_camelyon_balanced(df_test, test_samples)
    else:
        if bias_samples_train and bias_samples_val:
            assert bias_samples_train < train_samples, "bias samples can't be larger than train samples"
            assert bias_samples_val < val_samples, "bias samples can't be larger than val samples"
        else:
            bias_samples_train = 25
            bias_samples_val = 10
        train_idxs = sample_camelyon_biased(df_train, train_samples, split="train")
        val_idxs = sample_camelyon_biased(df_val, val_samples, split="validation")
        test_idxs = sample_camelyon_balanced(df_test, test_samples)

    train_df = df_train.iloc[train_idxs].copy()
    val_df = df_val.iloc[val_idxs].copy()
    test_df = df_test.iloc[test_idxs].copy()

    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise ValueError("Camelyon17 metadata did not produce non-empty train/val/test splits.")

    train_split_patients = set(train_df["patient"])
    val_split_patients = set(val_df["patient"])
    test_split_patients = set(test_df["patient"])
    assert train_split_patients.isdisjoint(val_split_patients)
    assert train_split_patients.isdisjoint(test_split_patients)
    assert val_split_patients.isdisjoint(test_split_patients)

    return (
        Camelyon17ShortcutDataset(
            train_df, root=root, label=label, attribute=attribute, path_column=path_column,
            attr_labs=attr_labs, attribute_name="Centers 2-4", label_name="Tumor",
            augment=train_augment,
        ),
        Camelyon17ShortcutDataset(
            val_df, root=root, label=label, attribute=attribute, path_column=path_column,
            attr_labs=attr_labs, attribute_name="Centers 2-4", label_name="Tumor",
        ),
        Camelyon17ShortcutDataset(
            test_df, root=root, label=label, attribute=attribute, path_column=path_column,
            attr_labs=attr_labs, attribute_name="Centers 2-4", label_name="Tumor",
        ),
    )

class ISIC2019ShortcutDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            root: str,
            label: str,
            attribute: str,
            path_column: str,
            attr_labs: bool,
            attribute_name: str,
            label_name: str,
            augment: bool = False,
        ):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.label = label
        self.attribute = attribute
        self.path_column = path_column
        self.attr_labs = attr_labs
        self.attribute_name = attribute_name
        self.label_name = label_name
        self.labels = torch.from_numpy(self.df[label].to_numpy()).long()
        self.attributes = torch.from_numpy(self.df[attribute].to_numpy()).long()
        if augment:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(20),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.df)

    def _image_path(self, relative_path):
        return os.path.join(self.root, relative_path)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self._image_path(row[self.path_column])).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(int(row[self.label])).long()
        attr = torch.tensor(int(row[self.attribute])).long()
        return (image, attr) if self.attr_labs else (image, label)

    def get_num_classes(self):
        return 2


def get_isic_biased_splits(root: str,
        label: str = "malignant",
        attribute: str = "source",
        balanced: bool = False,
        train_samples: int = 5000,
        val_samples: int = 1000,
        test_samples: int = 2000,
        bias_samples_train: int = None,
        bias_samples_val: int = None,
        attr_labs: bool = False,
        train_augment: bool = False,
    ):
    if not os.path.isdir(root):
        raise FileNotFoundError(f"ISIC dataset not found at {root!r}")

    train_df = pd.read_csv(os.path.join(root, "ISIC_2019_Training_GroundTruth.csv"))
    train_df = train_df.merge(
        pd.read_csv(os.path.join(root, "ISIC_2019_Training_Metadata.csv")),
        on="image",
        how="left",
    )
    train_df["_isic_partition"] = "Training"

    test_df = pd.read_csv(os.path.join(root, "ISIC_2019_Test_GroundTruth.csv"))
    test_df = test_df.merge(
        pd.read_csv(os.path.join(root, "ISIC_2019_Test_Metadata.csv")),
        on="image",
        how="left",
    )
    test_df["_isic_partition"] = "Test"

    df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    df["Path"] = df.apply(
        lambda row: os.path.join(
            f"ISIC_2019_{row['_isic_partition']}_Input",
            f"{row['image']}.jpg",
        ),
        axis=1,
    )

    if label == "malignant":
        malignant_cols = ["MEL"]
        benign_cols = ["NV"]
        df[malignant_cols + benign_cols] = df[malignant_cols + benign_cols].fillna(0).astype(float)
        malignant = df[malignant_cols].sum(axis=1) > 0
        benign = df[benign_cols].sum(axis=1) > 0
        df[label] = np.where(malignant, 1, np.where(benign, 0, np.nan))
    elif label not in df.columns:
        raise ValueError(f"ISIC label column {label!r} not found.")

    if attribute == "source":
        def image_size(relative_path):
            with Image.open(os.path.join(root, relative_path)) as image:
                return image.size

        df["_image_size"] = df["Path"].map(image_size)
        df = df[df["_image_size"].isin([(600, 450), (1024, 1024)])].copy()
        df[attribute] = (df["_image_size"] == (1024, 1024)).astype(int)
    else:
        raise ValueError(f"Unsupported ISIC attribute {attribute!r}. Use 'source'.")

    df = df.dropna(subset=["image", label, attribute]).copy()
    df[label] = df[label].astype(int)
    df[attribute] = df[attribute].astype(int)
    df = df[df[label].isin([0, 1]) & df[attribute].isin([0, 1])].copy()

    split_column = "lesion_id" if "lesion_id" in df.columns else "image"
    df["_split_id"] = df[split_column].where(df[split_column].notna(), df["image"]).astype(str)
    if df.empty:
        raise ValueError("ISIC metadata did not produce any labeled binary samples.")

    if not balanced:
        if bias_samples_train and bias_samples_val:
            assert bias_samples_train < train_samples, "bias samples can't be larger than train samples"
            assert bias_samples_val < val_samples, "bias samples can't be larger than val samples"
        else:
            bias_samples_train = 25
            bias_samples_val = 10

    group_keys = [(0, 0), (0, 1), (1, 0), (1, 1)]
    shuffled_ids = pd.Series(sorted(df["_split_id"].unique())).sample(frac=1.0, random_state=42).tolist()
    used_ids = set()
    split_group_counts = {}
    for split_id, rows in df.groupby("_split_id"):
        split_group_counts[split_id] = {
            key: int(((rows[label] == key[0]) & (rows[attribute] == key[1])).sum())
            for key in group_keys
        }

    test_targets = {key: test_samples // 4 for key in group_keys}
    if balanced:
        val_targets = {key: val_samples // 4 for key in group_keys}
    else:
        val_targets = {}
        val_per_side = val_samples // 2
        for y_val, a_val in group_keys:
            corr_attr = y_val if attribute == "source" else 1 - y_val
            val_targets[(y_val, a_val)] = (
                max(0, val_per_side - bias_samples_val)
                if a_val == corr_attr
                else bias_samples_val
            )

    def reserve_ids(targets):
        reserved_ids = []
        counts = {key: 0 for key in targets}
        for split_id in shuffled_ids:
            if split_id in used_ids:
                continue
            row_counts = split_group_counts[split_id]
            helps = any(counts[key] < targets[key] and row_counts[key] > 0 for key in targets)
            if not helps:
                continue
            reserved_ids.append(split_id)
            used_ids.add(split_id)
            for key in targets:
                counts[key] += row_counts[key]
            if all(counts[key] >= targets[key] for key in targets):
                break

        if any(counts[key] < targets[key] for key in targets):
            raise ValueError(f"ISIC split could not satisfy requested group counts: {counts} vs {targets}")
        return reserved_ids

    test_ids = reserve_ids(test_targets)
    val_ids = reserve_ids(val_targets)
    train_ids = [split_id for split_id in shuffled_ids if split_id not in used_ids]

    def subset_by_ids(ids):
        return df[df["_split_id"].isin(ids)].reset_index(drop=True)

    df_train = subset_by_ids(train_ids)
    df_val = subset_by_ids(val_ids)
    df_test = subset_by_ids(test_ids)

    def make_groups(df_sub):
        y_sub = df_sub[label].astype(int).to_numpy()
        a_sub = df_sub[attribute].astype(int).to_numpy()
        return group_indices(y_sub, a_sub)

    train_groups = make_groups(df_train)
    val_groups = make_groups(df_val)
    test_groups = make_groups(df_test)

    if balanced:
        train_idxs = sample_balanced(train_groups, train_samples)
        val_idxs = sample_balanced(val_groups, val_samples)
        test_idxs = sample_balanced(test_groups, test_samples)
    else:
        if attribute == "source":
            train_groups = group_indices(
                df_train[label].astype(int).to_numpy(),
                1 - df_train[attribute].astype(int).to_numpy(),
            )
            val_groups = group_indices(
                df_val[label].astype(int).to_numpy(),
                1 - df_val[attribute].astype(int).to_numpy(),
            )
        train_idxs = sample_biased(
            train_groups,
            train_samples,
            split="train",
            anti_per_label_train=bias_samples_train,
            anti_per_label_val=bias_samples_val,
        )
        val_idxs = sample_biased(
            val_groups,
            val_samples,
            split="validation",
            anti_per_label_train=bias_samples_train,
            anti_per_label_val=bias_samples_val,
        )
        test_idxs = sample_balanced(test_groups, test_samples)

    train_df = df_train.iloc[train_idxs].copy()
    val_df = df_val.iloc[val_idxs].copy()
    test_df = df_test.iloc[test_idxs].copy()

    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise ValueError("ISIC metadata did not produce non-empty train/val/test splits.")

    train_split_ids = set(train_df["_split_id"])
    val_split_ids = set(val_df["_split_id"])
    test_split_ids = set(test_df["_split_id"])
    assert train_split_ids.isdisjoint(val_split_ids)
    assert train_split_ids.isdisjoint(test_split_ids)
    assert val_split_ids.isdisjoint(test_split_ids)

    attribute_name = "BCN" if attribute == "source" else attribute
    label_name = "Malignant" if label == "malignant" else label

    return (
        ISIC2019ShortcutDataset(
            train_df, root=root, label=label, attribute=attribute, path_column="Path",
            attr_labs=attr_labs, attribute_name=attribute_name, label_name=label_name,
            augment=train_augment,
        ),
        ISIC2019ShortcutDataset(
            val_df, root=root, label=label, attribute=attribute, path_column="Path",
            attr_labs=attr_labs, attribute_name=attribute_name, label_name=label_name,
        ),
        ISIC2019ShortcutDataset(
            test_df, root=root, label=label, attribute=attribute, path_column="Path",
            attr_labs=attr_labs, attribute_name=attribute_name, label_name=label_name,
        ),
    )
