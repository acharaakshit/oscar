import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib

def z_score(sample):
    # z-score normalisation
    sample = (sample - sample.mean()) / sample.std()
    return sample

class SingleClassDataset(Dataset):
    def __init__(self, X, y_class, image_ids):
        self.X = X
        self.y_class = y_class
        self.ids = image_ids
        print(len(self.X), len(self.y_class), len(self.ids))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]
        # load the data
        sample = nib.load(sample).get_fdata()
        sample = z_score(sample)
        sample = np.expand_dims(sample, axis=0)

        # Convert to PyTorch tensor
        sample = torch.tensor(sample, dtype=torch.float32)

        return sample, self.y_class[idx], self.ids[idx]