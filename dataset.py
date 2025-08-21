import torch
from torch.utils.data import Dataset
import numpy as np
import os

class TimeSeriesDataset(Dataset):
    def __init__(self, folder, file_list=None, transform=None, labeled=False):
        self.folder = folder
        self.transform = transform
        self.labeled = labeled
        
        if file_list is None:
            self.files = [f for f in os.listdir(folder) if f.endswith('.npy')]
        else:
            with open(file_list) as f:
                self.files = [line.strip() for line in f]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        x = np.load(os.path.join(self.folder, self.files[idx]))
        x = torch.tensor(x, dtype=torch.float)
        
        if x.ndim == 2:
            x = x.permute(1, 0)  # (channels, length)
        elif x.ndim == 1:
            x = x.unsqueeze(0)  # (1, length)
        
        if self.transform:
            x = self.transform(x)
        
        if self.labeled:
            # For downstream task, assume labels are in filename or separate file
            # This is a placeholder - you'll need to adapt based on your data
            label = np.random.randint(0, 2)  # Placeholder
            return x, label
        
        return x

class SyntheticTimeSeriesDataset(Dataset):
    def __init__(self, n_samples=1000, length=300, n_channels=3, n_classes=2, labeled=False):
        self.n_samples = n_samples
        self.length = length
        self.n_channels = n_channels
        self.labeled = labeled
        self.n_classes = n_classes
        
        # Generate synthetic accelerometer-like data
        self.data = np.random.randn(n_samples, n_channels, length).astype(np.float32)
        
        if labeled:
            self.labels = np.random.randint(0, n_classes, n_samples)
        else:
            self.labels = None
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float)
        
        if self.labeled:
            return x, self.labels[idx]
        return x
