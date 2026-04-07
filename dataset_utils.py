import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

class FeatureDataset(Dataset):
    def __init__(self, features, labels=None, sets=None, mean=None, std=None, apply_sigmoid=True):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = None if labels is None else torch.as_tensor(labels, dtype=torch.long)
        self.sets = None if sets is None else np.asarray(sets)

        self.mean = None if mean is None else torch.as_tensor(mean, dtype=torch.float32)
        self.std = None if std is None else torch.as_tensor(std, dtype=torch.float32)
        self.apply_sigmoid = apply_sigmoid

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]

        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + 1e-8)

        if self.apply_sigmoid:
            x = torch.sigmoid(x)

        if self.labels is None:
            return x
        return x, self.labels[idx]

def load_encoder_imdb(npz_path):
    data = np.load(npz_path)
    features = data["data"].astype("float32")
    labels = data["labels"].astype("int64") if "labels" in data else None
    sets = data["set"].astype("int64")
    return features, labels, sets

def load_norm_stats(mean_path, std_path):
    mean = np.load(mean_path).astype("float32")
    std = np.load(std_path).astype("float32")
    return mean, std

def build_dataset(npz_path, mean_path, std_path, apply_sigmoid=True):
    x, y, s = load_encoder_imdb(npz_path)
    mean, std = load_norm_stats(mean_path, std_path)
    return FeatureDataset(x, y, s, mean=mean, std=std, apply_sigmoid=apply_sigmoid)

def split_indices_from_sets(sets):
    train_idx = np.where(sets == 1)[0]
    val_idx = np.where(sets == 2)[0]
    test_idx = np.where(sets == 3)[0]
    return train_idx, val_idx, test_idx

def build_loaders_from_dataset(dataset, batch_size=12, shuffle_train=True):
    if dataset.sets is None:
        raise ValueError("Dataset must contain set split information.")

    train_idx, val_idx, test_idx = split_indices_from_sets(dataset.sets)

    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())
    test_ds = Subset(dataset, test_idx.tolist())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
