import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
import json
import random
from scipy.interpolate import CubicSpline

def flip_time_series(x, choice):
    if choice == 1:
        return torch.flip(x, dims=[-1])
    return x

def permute_time_series(x, choice, max_segments=4, min_segment_length=10):
    if choice == 0:
        return x
    
    batch_size, channels, length = x.shape
    result = x.clone()
    
    for i in range(batch_size):
        n_segs = np.random.randint(2, max_segments + 1)
        splits = np.array_split(np.arange(length), n_segs)
        order = np.random.permutation(n_segs)
        
        new_x = torch.cat([x[i, :, splits[j]] for j in order], dim=-1)
        result[i] = new_x
    
    return result

def scale_time_series(x, choice, scale_range=0.1):
    if choice == 0:
        return x
    
    batch_size, channels, length = x.shape
    scaling_factors = torch.normal(1.0, scale_range, size=(batch_size, channels, 1), device=x.device)
    return x * scaling_factors

def time_warp_time_series(x, choice, sigma=0.2):
    if choice == 0:
        return x
    
    batch_size, channels, length = x.shape
    result = x.clone()
    
    for i in range(batch_size):
        for c in range(channels):
            knots = 4
            xx = np.linspace(0, length-1, knots + 2)
            yy = np.random.normal(loc=1.0, scale=sigma, size=knots + 2)
            cs = CubicSpline(xx, yy)
            
            time_warp = cs(np.arange(length))
            time_warp = np.cumsum(time_warp)
            time_warp = time_warp / time_warp[-1] * (length - 1)
            
            original_signal = x[i, c, :].cpu().numpy()
            warped_signal = np.interp(np.arange(length), time_warp, original_signal)
            result[i, c, :] = torch.tensor(warped_signal, device=x.device)
    
    return result

# Implementation based on ssl-wearables generate_labels function
# Reference: ssl-wearables/sslearning/data/data_loader.py lines 95-130
# Each pretext task makes independent random decisions per signal
def generate_pretext_labels_and_transform(x, cfg):
    batch_size = x.shape[0]
    labels = torch.zeros(batch_size, 4, dtype=torch.long)
    
    transformed_x = x.clone()
    
    if cfg.get('time_reversal', False):
        choices = torch.bernoulli(torch.full((batch_size,), cfg.get('positive_ratio', 0.5))).long()
        labels[:, 0] = choices
        for i, choice in enumerate(choices):
            transformed_x[i] = flip_time_series(transformed_x[i:i+1], choice.item())[0]
    
    if cfg.get('scale', False):
        choices = torch.bernoulli(torch.full((batch_size,), cfg.get('positive_ratio', 0.5))).long()
        labels[:, 1] = choices
        for i, choice in enumerate(choices):
            transformed_x[i] = scale_time_series(transformed_x[i:i+1], choice.item())[0]
    
    if cfg.get('permutation', False):
        choices = torch.bernoulli(torch.full((batch_size,), cfg.get('positive_ratio', 0.5))).long()
        labels[:, 2] = choices
        for i, choice in enumerate(choices):
            transformed_x[i] = permute_time_series(transformed_x[i:i+1], choice.item())[0]
    
    if cfg.get('time_warped', False):
        choices = torch.bernoulli(torch.full((batch_size,), cfg.get('positive_ratio', 0.5))).long()
        labels[:, 3] = choices
        for i, choice in enumerate(choices):
            transformed_x[i] = time_warp_time_series(transformed_x[i:i+1], choice.item())[0]
    
    return transformed_x, labels

class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class ResNetSSL(nn.Module):
    def __init__(self, in_channels=3, n_classes_per_task=2, tasks=['time_reversal', 'scale', 'permutation', 'time_warped']):
        super().__init__()
        self.tasks = tasks
        self.n_tasks = len(tasks)
        
        # Feature extractor based on ResNet1D
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            self._make_layer(64, 64, 2),
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2),
            self._make_layer(256, 512, 2, stride=2),
            
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict({
            task: nn.Linear(512, n_classes_per_task) for task in tasks
        })
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        
        # First block might need stride
        layers.append(self._basic_block(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(self._basic_block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _basic_block(self, in_channels, out_channels, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        
        return BasicBlock1D(in_channels, out_channels, stride, downsample)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.squeeze(-1)
        
        outputs = {}
        for task in self.tasks:
            outputs[task] = self.task_heads[task](features)
        
        return outputs

def compute_multitask_loss(outputs, labels, task_config):
    total_loss = 0
    total_acc = 0
    n_active_tasks = 0
    
    criterion = nn.CrossEntropyLoss()
    
    task_names = ['time_reversal', 'scale', 'permutation', 'time_warped']
    
    for i, task in enumerate(task_names):
        if task_config.get(task, False):
            pred = outputs[task]
            target = labels[:, i]
            
            loss = criterion(pred, target)
            total_loss += loss
            
            acc = (pred.argmax(1) == target).float().mean()
            total_acc += acc
            
            n_active_tasks += 1
    
    if n_active_tasks > 0:
        total_loss /= n_active_tasks
        total_acc /= n_active_tasks
    
    return total_loss, total_acc

def train_ssl_multitask(model, dataloader, task_config, epochs=50, lr=1e-4):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        
        for batch_idx, x in enumerate(dataloader):
            if isinstance(x, list):
                x = x[0]
            
            x_transformed, labels = generate_pretext_labels_and_transform(x, task_config)
            
            outputs = model(x_transformed)
            
            loss, acc = compute_multitask_loss(outputs, labels, task_config)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_acc += acc.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)
        
        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, Acc = {avg_acc:.4f}")

class DownstreamClassifier(nn.Module):
    def __init__(self, feature_extractor, n_classes):
        super().__init__()
        self.feature_extractor = feature_extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
            features = features.squeeze(-1)
        return self.classifier(features)

def train_downstream_classifier(model, train_loader, test_loader, epochs=50, lr=1e-3):
    model.train()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        
        model.train()
        for x, y in train_loader:
            outputs = model(x)
            loss = criterion(outputs, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += (outputs.argmax(1) == y).float().mean().item()
        
        model.eval()
        test_acc = 0
        test_loss = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                outputs = model(x)
                loss = criterion(outputs, y)
                test_loss += loss.item()
                test_acc += (outputs.argmax(1) == y).float().mean().item()
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")
