import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def filter_and_interpolate(signal):
    result = signal.copy()
    zero_indices = np.where(signal == 0)[0]
    for idx in zero_indices:
        left_val = None
        right_val = None
        for i in range(idx-1, -1, -1):
            if signal[i] != 0:
                left_val = signal[i]
                left_idx = i
                break
        for i in range(idx+1, len(signal)):
            if signal[i] != 0:
                right_val = signal[i]
                right_idx = i
                break
        if left_val is not None and right_val is not None:
            weight = (idx - left_idx) / (right_idx - left_idx)
            result[idx] = left_val + weight * (right_val - left_val)
        elif left_val is not None:
            result[idx] = left_val
        elif right_val is not None:
            result[idx] = right_val
    return result

directory = "SMC_WatchSpO2_deliverable_241226/SMC_WatchSpO2_deliverable_241226"
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
    num_segments = signal_length // 1000
    for i in range(num_segments):
        start_idx = i * 1000
        end_idx = start_idx + 1000
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

signals = [filter_and_interpolate(signal) for signal in filtered_signals_only]
min_length = min(len(signal) for signal in signals)
print(f"Minimum signal length: {min_length}")

truncated_signals = []
for signal in signals:
    truncated = signal[:min_length]
    truncated_signals.append(truncated.reshape(1, -1))

print(f"Prepared {len(truncated_signals)} signals of length {min_length} for downstream task")

ahi_df = pd.read_csv('patient_ahi_isi.csv')
ahi_df['AHI_binary'] = (ahi_df['AHI'] >= 15).astype(int)
print(f"AHI data loaded: {len(ahi_df)} patients")
print("Sample data:")
print(ahi_df.head())

# Match signals with AHI labels
downstream_signals = []
downstream_targets = []
matched_patients = []
for signal, patient_name in new_df_list:
    if '_seg' in patient_name:
        patient_id = patient_name.split('_seg')[0]
    else:
        patient_id = patient_name
    patient_row = ahi_df[ahi_df['ID'] == patient_id]
    if not patient_row.empty:
        zero_percentage = np.sum(signal == 0) / len(signal)
        if zero_percentage <= 0.5:
            filtered_signal = filter_and_interpolate(signal)
            downstream_signals.append(filtered_signal)
            downstream_targets.append(patient_row['AHI_binary'].iloc[0])
            matched_patients.append(patient_id)

print(f"\nMatched {len(downstream_signals)} signal segments with AHI labels")
print(f"Target distribution in matched data:")
print(f"AHI < 15: {np.sum(np.array(downstream_targets) == 0)} segments")
print(f"AHI >= 15: {np.sum(np.array(downstream_targets) == 1)} segments")

min_downstream_length = min(len(signal) for signal in downstream_signals)
print(f"Minimum signal length for downstream: {min_downstream_length}")

downstream_processed = []
for signal in downstream_signals:
    truncated = signal[:min_downstream_length]
    downstream_processed.append(truncated.reshape(1, -1))

print(f"Prepared {len(downstream_processed)} signals for downstream task")
print(f"Signal shape: {downstream_processed[0].shape}")

print("\n🚀 Training and Testing Transfer Learning Classifier")
print("="*80)
import sys
sys.path.append('.')
from ssl_utils import tl_activity

results = tl_activity(
    test_signals=downstream_processed,
    signal_length=min_downstream_length,
    targets=downstream_targets,
    pretrained_model_path="model_check_point/mtl_30_best.mdl",
    epochs=40,
    learning_rate=1e-3,
    batch_size=16,
    results_save_path="tl_ahi_classification_results.json"
)

print("\n✅ Transfer learning training and evaluation completed!")
print(f"📊 Final Performance:")
print(f"   • Accuracy: {results['accuracy']:.4f}")
print(f"   • F1 Score: {results['f1_score']:.4f}")
print(f"   • AUC: {results['auc']:.4f}")