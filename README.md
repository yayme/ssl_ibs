# SSL Time Series Pipeline

Internal SSL pipeline for **1D time series data** (SpO2 signals), based on ssl-wearables architecture.

## Overview

Multi-task SSL with four pretext tasks:
- **Time Reversal**: Predict if signal is temporally reversed
- **Scale**: Predict if signal is temporally scaled 
- **Permutation**: Predict if signal segments are permuted
- **Time Warp**: Predict if signal is time-warped

Uses ResNet1D backbone with task-specific heads.

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

## Files

- `ssl_utils.py` - Main training/testing functions
- `ssl_new_pipeline.py` - Core SSL implementation  
- `Testing_pipeline.ipynb` - Example usage with SpO2 data
- `config.json` - Configuration parameters




