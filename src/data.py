from torch.utils.data import Dataset
import monai.transforms as monaitransforms
import numpy as np
import nibabel as nib

transforms = monaitransforms.Compose([
                        monaitransforms.RandFlip(prob=0.3, spatial_axis=(0)),  # Random flipping along x-axis
                        monaitransforms.RandAffine(prob=0.5, rotate_range=(0.1, 0.1, 0.1), translate_range=(10, 10, 10)),  # Random rotations,
                        monaitransforms.ToTensor()]
                    )


def hnorm():
    return monaitransforms.Compose([
                        monaitransforms.HistogramNormalize(min=0, max=1), # histogram in same value range
                ])

def z_score(sample):
    # z-score normalisation
    sample = (sample - sample.mean()) / sample.std()
    return sample

def histogram(sample):
    # clip lowest percentile values for cleaning
    sample = np.clip(sample, np.percentile(sample, 0.001), np.percentile(sample, 99.999))
    sample = hnorm()(sample).numpy()
    return sample

def minmax(sample):
    sample = np.clip(sample, np.percentile(sample, 0.001), np.percentile(sample, 99.999))
    sample = (sample - sample.mean())/(sample.max() - sample.mean()) # min-max scaling
    return sample

def normalise(sample, channels):
    if channels == 1:
        sample = z_score(sample) # deterministic
        sample = np.expand_dims(sample, axis=0)
    elif channels == 2:
        # no randomness as validation set should be consistent
        sample = np.concatenate([
                    np.expand_dims(z_score(sample), axis=0),
                    np.expand_dims(z_score(histogram(sample)), axis=0)
                ], 
                axis=0)
    else:
        # all channels with specific goal
        sample = np.concatenate([
                    np.expand_dims(z_score(sample), axis=0),
                    np.expand_dims(z_score(minmax(sample)), axis=0),
                    np.expand_dims(z_score(histogram(sample)), axis=0)
                ], 
                axis=0)
    return sample

class SingleClassDataset(Dataset):
    def __init__(self, X, y_class, augment = False, channels = 1):
        self.X = X
        self.y_class = y_class
        self.channels = channels
        assert self.channels in [1,2,3] # can't have any other number of channels
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]
        sample = nib.load(sample).get_fdata()
        sample = normalise(sample=sample, channels=self.channels)

        if self.augment:
            sample = transforms(sample)
        
        return sample, self.y_class[idx]