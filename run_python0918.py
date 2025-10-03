import os
signal_list=[]
for filename in os.listdir("SMC_WatchSpO2_deliverable_241226/SMC_WatchSpO2_deliverable_241226"):
    print(filename)
directory="SMC_WatchSpO2_deliverable_241226/SMC_WatchSpO2_deliverable_241226"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df_dict = {}
device_names = ['GW4','GW5','GW6','GW7']
for device_name in device_names:
    path = os.path.join(directory, 'watch' + device_name[-1])
    GW = os.listdir(path)
    GW.sort()
    for file_name in GW:
        file_path = os.path.join(path, file_name)
        df = pd.read_csv(file_path)
        if device_name == 'GW4':
            try:
                patient = 'GW4' + file_name.split('SUB_SMC_')[1].split('_')[0]
            except Exception:
                continue
        else:
            try:
                patient_id = file_name.split('_Id')[1].split('_')[0]
                patient = 'GW' + device_name[-1] + patient_id[-3:]
            except Exception:
                continue
        signal = df['SpO2'].values
        zero_count = np.sum(signal == 0)
        if patient not in df_dict or zero_count < df_dict[patient][1]:
            df_dict[patient] = (signal, zero_count)
df_list = [(signal, patient) for patient, (signal, _) in df_dict.items()]
print(f"length of df_list {len(df_list)}")

new_df_list = []
for signal, patient in df_list:
    signal_length = len(signal)
    num_segments = signal_length // 500  # Changed from 1000 to 500
    for i in range(num_segments):
        start_idx = i * 500  # Changed from 1000 to 500
        end_idx = start_idx + 500  # Changed from 1000 to 500
        segment = signal[start_idx:end_idx]
        new_df_list.append((segment, f"{patient}_seg{i}"))

print(len(new_df_list))

signals_only = [signal for signal, patient in new_df_list]

filtered_signals_only = []
for signal in signals_only:
    zero_count = np.sum(signal == 0)
    zero_percentage = zero_count / len(signal)
    if zero_percentage <= 0.5:
        filtered_signals_only.append(signal)

print(f"Original signals: {len(signals_only)}")
print(f"After filtering (<=50% zeros): {len(filtered_signals_only)}")


def filter_and_interpolate(signal):
    result = signal.copy()
    zero_indices = np.where(signal == 0)[0]
    
    for idx in zero_indices:
        left_val = None
        right_val = None
        
        # Find nearest non-zero value to the left
        for i in range(idx-1, -1, -1):
            if signal[i] != 0:
                left_val = signal[i]
                left_idx = i
                break
        
        # Find nearest non-zero value to the right
        for i in range(idx+1, len(signal)):
            if signal[i] != 0:
                right_val = signal[i]
                right_idx = i
                break
        
        # Interpolate
        if left_val is not None and right_val is not None:
            # Linear interpolation
            weight = (idx - left_idx) / (right_idx - left_idx)
            result[idx] = left_val + weight * (right_val - left_val)
        elif left_val is not None:
            result[idx] = left_val
        elif right_val is not None:
            result[idx] = right_val
    
    return result
signals = [filter_and_interpolate(signal) for signal in filtered_signals_only]
# Find minimum length across all signals
min_length = min(len(signal) for signal in signals)
print(f"Minimum signal length: {min_length}")

# Truncate all signals to minimum length
truncated_signals = []
for signal in signals:
    truncated = signal[:min_length]
    # Reshape to (1, length) for 1D time series
    truncated_signals.append(truncated.reshape(1, -1))


print(f"Prepared {len(truncated_signals)} signals of length {min_length} for SSL training")
# Analyze signal statistics
print("Signal Statistics:")
print(f"Number of signals: {len(truncated_signals)}")
print(f"Signal length: {min_length}")
print(f"Signal shape: {truncated_signals[0].shape}")

# Check for data quality
sample_signal = truncated_signals[0][0]
print(f"\nSample signal stats:")
print(f"Mean: {np.mean(sample_signal):.2f}")
print(f"Std: {np.std(sample_signal):.2f}")
print(f"Min: {np.min(sample_signal):.2f}")
print(f"Max: {np.max(sample_signal):.2f}")
print(f"Zeros: {np.sum(sample_signal == 0)}")

# Check all signals for zeros
zero_counts = [np.sum(signal[0] == 0) for signal in truncated_signals]
print(f"\nZero statistics across all signals:")
print(f"Average zeros per signal: {np.mean(zero_counts):.1f}")
print(f"Max zeros in a signal: {np.max(zero_counts)}")
print(f"Signals with >50% zeros: {sum(1 for zc in zero_counts if zc > min_length/2)}")
# Import SSL utility functions
import sys
sys.path.append('.')  # Add current directory to path
from ssl_utils import train_ssl

print("Starting SSL training on SpO2 data with 500-length segments...")

# Visualize signal transformations to verify pretext tasks
import torch
from ssl_new_pipeline import (flip_time_series, scale_time_series, 
                             permute_time_series, time_warp_time_series,
                             generate_pretext_labels_and_transform)

# Take a sample signal for transformation demo
sample_signal = torch.tensor(truncated_signals[0], dtype=torch.float32).unsqueeze(0)  # Add batch dimension
print(f"Original signal shape: {sample_signal.shape}")

# Create SSL config for transformations
ssl_config = {
    'time_reversal': True,
    'scale': True, 
    'permutation': True,
    'time_warped': True,
    'positive_ratio': 0.5
}

# Generate transformations
x_transformed, labels = generate_pretext_labels_and_transform(sample_signal, ssl_config)
print(f"Labels for transformations: {labels[0]}")  # [time_reversal, scale, permutation, time_warped]

# Plot original vs transformed
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Original signal
axes[0, 0].plot(sample_signal[0, 0].numpy())
axes[0, 0].set_title('Original Signal (500 length)')
axes[0, 0].grid(True)

# Individual transformations for visualization
test_signal = sample_signal.clone()

# Time reversal
flipped = flip_time_series(test_signal, 1)
axes[0, 1].plot(flipped[0, 0].numpy())
axes[0, 1].set_title('Time Reversal (Flipped)')
axes[0, 1].grid(True)

# Scaling
scaled = scale_time_series(test_signal, 1)
axes[0, 2].plot(scaled[0, 0].numpy())
axes[0, 2].set_title('Scaled Signal')
axes[0, 2].grid(True)

# Permutation
permuted = permute_time_series(test_signal, 1)
axes[1, 0].plot(permuted[0, 0].numpy())
axes[1, 0].set_title('Permuted Segments')
axes[1, 0].grid(True)

# Time warping
warped = time_warp_time_series(test_signal, 1)
axes[1, 1].plot(warped[0, 0].numpy())
axes[1, 1].set_title('Time Warped')
axes[1, 1].grid(True)

# Combined transformation (what actually gets fed to model)
axes[1, 2].plot(x_transformed[0, 0].numpy())
axes[1, 2].set_title('Final Transformed Signal')
axes[1, 2].grid(True)

plt.tight_layout()
plt.show()

print("✅ Signal transformations are working for 500-length segments!")
print(f"Original signal range: [{sample_signal.min():.2f}, {sample_signal.max():.2f}]")
print(f"Transformed signal range: [{x_transformed.min():.2f}, {x_transformed.max():.2f}]")

# Train SSL model on SpO2 signals with enhanced progress tracking
print("🚀 Starting Enhanced SSL Training with 500-Length Segments")
print("="*60)

# Reload the updated ssl_utils module
import importlib
importlib.reload(sys.modules['ssl_utils'])
from ssl_utils import train_ssl

# Train with enhanced logging and progress bars
model_path = train_ssl(
    signals=truncated_signals,
    signal_length=min_length,
    epochs=20,  
    learning_rate=1e-4,
    batch_size=16,  # Smaller batch size for limited data
    model_save_path="spo2_ssl_model_500length.pth"
)

print(f"\n✅ SSL training completed! Model saved to: {model_path}")
print("🎯 Check the training progress plots above!")

import pandas as pd
from sklearn.model_selection import train_test_split

print("🎯 Starting Downstream Classification Task with 500-Length Segments")
print("="*60)

# Load AHI data
ahi_df = pd.read_csv('patient_ahi_isi.csv')
print(f"AHI data loaded: {len(ahi_df)} patients")
print("Sample data:")
print(ahi_df.head())

# Create binary target: AHI >= 15
ahi_df['AHI_binary'] = (ahi_df['AHI'] >= 15).astype(int)
print(f"\nTarget distribution:")
print(f"AHI < 15: {(ahi_df['AHI_binary'] == 0).sum()} patients")
print(f"AHI >= 15: {(ahi_df['AHI_binary'] == 1).sum()} patients")

# Match signals with AHI labels
downstream_signals = []
downstream_targets = []
matched_patients = []

for signal, patient_name in new_df_list:
    # Extract patient ID from segment name (e.g., "GW4001_seg0" -> "GW4001")
    if '_seg' in patient_name:
        patient_id = patient_name.split('_seg')[0]
    else:
        patient_id = patient_name
    
    # Check if this patient exists in AHI data
    patient_row = ahi_df[ahi_df['ID'] == patient_id]
    if not patient_row.empty:
        # Filter signals with <=50% zeros
        zero_percentage = np.sum(signal == 0) / len(signal)
        if zero_percentage <= 0.5:
            # Apply same interpolation as training data
            filtered_signal = filter_and_interpolate(signal)
            downstream_signals.append(filtered_signal)
            downstream_targets.append(patient_row['AHI_binary'].iloc[0])
            matched_patients.append(patient_id)

print(f"\nMatched {len(downstream_signals)} signal segments with AHI labels")
print(f"Target distribution in matched data:")
print(f"AHI < 15: {np.sum(np.array(downstream_targets) == 0)} segments")
print(f"AHI >= 15: {np.sum(np.array(downstream_targets) == 1)} segments")

# Prepare signals (same preprocessing as SSL training)
min_downstream_length = min(len(signal) for signal in downstream_signals)
print(f"Minimum signal length for downstream: {min_downstream_length}")

downstream_processed = []
for signal in downstream_signals:
    truncated = signal[:min_downstream_length]
    downstream_processed.append(truncated.reshape(1, -1))

print(f"Prepared {len(downstream_processed)} signals for downstream task")
print(f"Signal shape: {downstream_processed[0].shape}")

# Test downstream task using SSL pretrained model
print("🚀 Testing Downstream Classification with SSL Features (500-Length)")
print("="*60)

# Import test_ssl function
from ssl_utils import test_ssl

# Use the test_ssl function from ssl_utils with the new model
results = test_ssl(
    test_signals=downstream_processed,
    signal_length=min_downstream_length,
    targets=downstream_targets,
    model_path=model_path,  # Use the newly trained model path
    epochs=20,
    learning_rate=1e-3,
    batch_size=16,
    results_save_path="ahi_classification_results_500length.json"
)

print("\n✅ Downstream classification with 500-length segments completed!")
print(f"📊 Results saved and plots generated")
print(f"🎯 Final Performance:")
print(f"   • Accuracy: {results['accuracy']:.4f}")
print(f"   • F1 Score: {results['f1_score']:.4f}")
print(f"   • AUC: {results['auc']:.4f}")
print(f"📈 Comparing 500-length vs 1000-length performance:")
print(f"   This model uses shorter time series (500 vs 1000 samples)")
print(f"   which may capture different temporal patterns in SpO2 data")
