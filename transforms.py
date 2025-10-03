import torch
import numpy as np

def random_jitter(x, sigma=0.03):
    return x + sigma * torch.randn_like(x)

def scaling(x, sigma=0.1):
    factor = torch.normal(1.0, sigma, size=(x.size(0), 1, 1), device=x.device)
    return x * factor

def permutation(x, max_segments=5):
    orig = x.clone()
    batch, channels, length = x.size()
    for i in range(batch):
        n_segs = np.random.randint(1, max_segments)
        splits = np.array_split(np.arange(length), n_segs)
        order = np.random.permutation(n_segs)
        x[i] = torch.cat([orig[i,:,splits[j]] for j in order], dim=-1)
    return x

def random_transform(x):
    if np.random.rand() < 0.5:
        x = random_jitter(x)
    if np.random.rand() < 0.5:
        x = scaling(x)
    if np.random.rand() < 0.5:
        x = permutation(x)
    return x
