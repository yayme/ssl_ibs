# SSL Time Series Pipeline

Internal SSL pipeline for **1D time series data** (SpO2 signals), based on ssl-wearables architecture.

## Overview

Multi-task SSL with four pretext tasks:
- **Time Reversal**: Predict if signal is temporally reversed
- **Scale**: Predict if signal is temporally scaled 
- **Permutation**: Predict if signal segments are permuted
- **Time Warp**: Predict if signal is time-warped

Uses ResNet1D backbone with task-specific heads.

## Internal Architecture

### ResNet1D Backbone
- **Base Model**: ResNet18 adapted for 1D signals
- **Input**: (batch_size, 1, sequence_length) 
- **Features**: 512-dimensional embeddings
- **Activation**: ReLU throughout
- **Normalization**: BatchNorm1d after each conv layer

### Multi-Task Heads
Each pretext task has dedicated binary classification head:
```
backbone_features (512) -> Linear(512, 256) -> ReLU -> Dropout(0.1) -> Linear(256, 1) -> Sigmoid
```

### SSL Transformations
1. **Time Reversal**: `torch.flip(signal, dims=[2])`
2. **Scale**: Temporal interpolation with factors [0.8, 1.2]
3. **Permutation**: Swap random 25% segments
4. **Time Warp**: Cubic spline interpolation with random warping

## Training Pipeline

### Phase 1: SSL Pre-training
- **Objective**: Multi-task binary classification on pretext tasks
- **Loss**: Binary cross-entropy per task, summed
- **Optimizer**: Adam with weight decay 1e-5
- **Schedule**: ReduceLROnPlateau (patience=10, factor=0.5)
- **Batch Size**: 32
- **Data Augmentation**: Random noise (σ=0.01)

### Phase 2: Downstream Fine-tuning  
- **Freeze**: Backbone frozen, only classifier trained
- **Architecture**: backbone -> Linear(512, num_classes)
- **Optimizer**: Adam with higher LR (1e-3)
- **Early Stopping**: Validation loss patience=10





## Quick Usage

```python
from ssl_utils import train_ssl, test_ssl

# Your signals as list of numpy arrays, shape (1, length)
signals = [signal.reshape(1, -1) for signal in your_signal_list]

# Train SSL
model_path = train_ssl(signals=signals, signal_length=300, epochs=50)

# Test downstream
results = test_ssl(test_signals, 300, targets, model_path, epochs=20)
```

## Configuration

Edit `config.json`:
```json
{
  "n_channels": 1,
  "sequence_length": 300,
  "time_reversal": true,
  "scale": true,
  "permutation": true,
  "time_warped": true,
  "ssl_epochs": 50,
  "ssl_lr": 1e-4
}
```

## Data Loading

We load data directly from CSV files (see `Testing_pipeline.ipynb`):
- Read SpO2 signals from CSV files
- Extract signal arrays
- Reshape to (1, length) format
- Pass to training functions

<!-- we are not using any .npy data. read testing_pipeline.ipynb to understand. -->

## Implementation Details

### File Structure
```
ssl_new/
├── ssl_utils.py           # High-level train/test functions
├── ssl_new_pipeline.py    # Core SSL implementation
├── models/
│   └── resnet1d.py       # 1D ResNet architecture
├── transforms.py         # SSL transformations
├── dataset.py           # Data loading utilities
├── config.json          # Configuration parameters
├── Testing_pipeline.ipynb # Usage examples
└── README.md            # This file
```

### Dependencies
- PyTorch 1.7+
- NumPy, Pandas
- Scikit-learn (metrics)
- tqdm (progress bars)
- Matplotlib (plotting)
- SciPy (signal processing)

### Performance Monitoring
- SSL training curves saved as `ssl_training_curves.png`
- Progress tracking with tqdm bars
- Timestamped model checkpoints
- Validation metrics logged during training

## Files

- `ssl_utils.py` - Main training/testing functions with progress tracking
- `ssl_new_pipeline.py` - Core SSL implementation with multi-task learning
- `models/resnet1d.py` - 1D ResNet backbone architecture
- `transforms.py` - SSL pretext task transformations
- `dataset.py` - PyTorch dataset classes for SSL training
- `Testing_pipeline.ipynb` - Complete example with SpO2 data and experimental documentation
- `config.json` - Configuration parameters for all experiments




