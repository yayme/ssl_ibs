import os
import pandas as pd
import numpy as np
import torch
from ssl_utils import test_mtl_ssl,set_random_seed
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
set_random_seed()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
# option: cuda:0 cuda:1 cuda:2
print(f'total GPU count: {torch.cuda.device_count()}')
print(f'current working GPU index: {device.index}')


class SMCDataProcessor:
    def __init__(self, base_directory):
        self.base_directory = base_directory
        
    def extract_patient_id(self, file_name, ring_type):
        if ring_type in ['rk4', 'rw3']:
            return file_name.split('_')[2][-3:]
        elif ring_type in ['rs4', 'rw4', 'w4', 'w5', 'w7', 'w56']:
            return file_name.split('_')[1][-3:]
        elif ring_type == 'rw2':
            return file_name.split('_')[0][-3:]
        else:
            return file_name.split('_')[2][-3:]
    
    def load_ring_data(self, ring_types):
        add_folder = os.path.join(self.base_directory, 'add')
        
        ring_data = []
        
        for ring_type in ring_types:
            ring_path = os.path.join(add_folder, f'apnea_algo_input_{ring_type}')
            print(ring_path)
            if not os.path.exists(ring_path):
                continue
                
            csv_files = [f for f in os.listdir(ring_path) if f.endswith('.csv')]
            
            for file_name in csv_files:
                
                try:
                    df = pd.read_csv(os.path.join(ring_path, file_name))
                    patient_id = self.extract_patient_id(file_name, ring_type)
                    
                    features = {}
                    if 'SpO2' in df.columns:
                        features['SpO2'] = df['SpO2'].values
                    if 'HR' in df.columns:
                        features['HR'] = df['HR'].values
                    if 'acc_power' in df.columns:
                        features['acc_power'] = df['acc_power'].values
                    if 'DC_R' in df.columns:
                        features['DC_R'] = df['DC_R'].values
                        
                    ring_data.append((features, f"{ring_type}_{patient_id}"))
                except Exception as e:
                    continue
                    
        return ring_data
    
    def load_watch_unlabeled_data(self, watch_types):
        add_folder = os.path.join(self.base_directory, 'add')
        watch_data = []
        
        for watch_type in watch_types:
            watch_path = os.path.join(add_folder, f'apnea_algo_input_{watch_type}')
            if not os.path.exists(watch_path):
                continue
                
            csv_files = [f for f in os.listdir(watch_path) if f.endswith('.csv')]
            
            for file_name in csv_files:
                try:
                    df = pd.read_csv(os.path.join(watch_path, file_name))
                    patient_id = self.extract_patient_id(file_name, watch_type)
                    
                    features = {}
                    if 'SpO2' in df.columns:
                        features['SpO2'] = df['SpO2'].values
                    if 'HR' in df.columns:
                        features['HR'] = df['HR'].values
                    if 'acc_power' in df.columns:
                        features['acc_power'] = df['acc_power'].values
                    if 'DC_R' in df.columns:
                        features['DC_R'] = df['DC_R'].values
                        
                    watch_data.append((features, f"{watch_type}_{patient_id}"))
                except Exception as e:
                    continue
                    
        return watch_data
    
    def load_watch_labeled_data(self):
        watch_data = []
        device_names = ['GW4', 'GW5', 'GW6', 'GW7']
        
        for device_name in device_names:
            path = os.path.join(self.base_directory, f'watch{device_name[-1]}', 'apnea_algo_input')
            if not os.path.exists(path):
                continue
                
            csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
            
            for file_name in csv_files:
                try:
                    df = pd.read_csv(os.path.join(path, file_name))
                    
                    if device_name == 'GW4':
                        patient = 'GW4' + file_name.split('SUB_SMC_')[1].split('_')[0]
                    else:
                        patient_id = file_name.split('_Id')[1].split('_')[0]
                        patient = 'GW' + device_name[-1] + patient_id[-3:]
                    
                    features = {}
                    if 'SpO2' in df.columns:
                        features['SpO2'] = df['SpO2'].values
                    if 'HR' in df.columns:
                        features['HR'] = df['HR'].values
                    if 'acc_power' in df.columns:
                        features['acc_power'] = df['acc_power'].values
                    if 'DC_R' in df.columns:
                        features['DC_R'] = df['DC_R'].values
                        
                    watch_data.append((features, patient))
                except Exception as e:
                    continue
                    
        return watch_data
    
    def interpolate_missing_values(self, signal):
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
    
    def normalize_signal(self, signal):
        scaler = MinMaxScaler()
        return scaler.fit_transform(signal.reshape(-1, 1)).flatten()
    
    def preprocess_data(self, data_list, feature_names, segment_length=1000):
        processed_segments = []
        
        for features_dict, patient_id in data_list:
            feature_arrays = []
            min_length = float('inf')
            
            for feature_name in feature_names:
                if feature_name in features_dict:
                    feature_arrays.append(features_dict[feature_name])
                    min_length = min(min_length, len(features_dict[feature_name]))
                else:
                    print(f"Warning: {feature_name} not found for {patient_id}")
                    continue
            
            if len(feature_arrays) != len(feature_names):
                continue
                
            feature_arrays = [arr[:min_length] for arr in feature_arrays]
            
            processed_features = []
            for arr in feature_arrays:
                interpolated = self.interpolate_missing_values(arr)
                normalized = self.normalize_signal(interpolated)
                processed_features.append(normalized)
            
            num_segments = min_length // segment_length
            for i in range(num_segments):
                start_idx = i * segment_length
                end_idx = start_idx + segment_length
                
                segment_features = []
                zero_percentage_total = 0
                
                for feature_arr in feature_arrays:
                    segment = feature_arr[start_idx:end_idx]
                    zero_count = np.sum(segment == 0)
                    zero_percentage_total += zero_count / len(segment)
                    
                    processed_segment = processed_features[len(segment_features)][start_idx:end_idx]
                    segment_features.append(processed_segment)
                
                avg_zero_percentage = zero_percentage_total / len(feature_arrays)
                if avg_zero_percentage <= 0.5:
                    multi_channel_segment = np.stack(segment_features, axis=0)
                    processed_segments.append((multi_channel_segment, f"{patient_id}_seg{i}"))
        
        return processed_segments
def main():
    # base_directory = r"C:\Users\PC\ssl\ssl_new\SMC_WatchSpO2_deliverable_250918\SMC_WatchSpO2_deliverable_241226"
    base_directory=r"SMC_WatchSpO2_deliverable_250918/SMC_WatchSpO2_deliverable_241226"

    processor = SMCDataProcessor(base_directory)
    
    print("=== SMC SSL Training Pipeline ===")
    
    print("\n1. Loading data...")
    ring_data = processor.load_ring_data(['rk4', 'rs4', 'rw2', 'rw3', 'rw4'])
    watch_unlabeled = processor.load_watch_unlabeled_data(['w4', 'w5', 'w56', 'w7'])
    watch_labeled = processor.load_watch_labeled_data()
    
    print(f"Ring data: {len(ring_data)} samples")
    print(f"Watch unlabeled: {len(watch_unlabeled)} samples")
    print(f"Watch labeled: {len(watch_labeled)} samples")
    
    config = {'features': ['SpO2','HR','DC_R', 'acc_power']}
    processed_data = processor.preprocess_data(watch_labeled, config['features'])
        
    if len(processed_data) == 0:
        print("No valid data found")
        return
        
    print(f"Processed segments: {len(processed_data)}")
    ahi_df = pd.read_csv('patient_ahi_isi.csv')
    print(f"AHI data loaded: {len(ahi_df)} patients")

    ahi_df['OSA'] = (ahi_df['AHI'] >= 15).astype(int)
    ahi_df['Insomnia'] = (ahi_df['ISI'] >= 15).astype(int)
    # ahi_df['COMISA'] = ((ahi_df['AHI'] >= 15) & (ahi_df['ISI'] >= 15)).astype(int)

 

    conditions = ['OSA', 'Insomnia']
    model_path = "TaskC_20251109_184902.pth"
    
    all_results = {}
    
    for condition in conditions:
        print(f"\nTesting {condition}...")
        
        downstream_signals = []
        downstream_targets = []
        
        for signal, patient_name in processed_data:
            if '_seg' in patient_name:
                patient_id = patient_name.split('_seg')[0]
            else:
                patient_id = patient_name
            
            patient_row = ahi_df[ahi_df['ID'] == patient_id]
            if not patient_row.empty:
                downstream_signals.append(signal)
                downstream_targets.append(patient_row[condition].iloc[0])
        
        if len(downstream_signals) == 0:
            print(f"No data for {condition}")
            continue
        
        import time
        unique_id = int(time.time())
        results = test_mtl_ssl(
            test_signals=downstream_signals,
            signal_length=1000,
            targets=downstream_targets,
            model_path=model_path,
            epochs=1000,
            learning_rate=1e-3,
            batch_size=16,
            results_save_path=f"TaskC_epoch1000_down1000__Total{condition.lower()}_results_{unique_id}.json",
            device=device
        )
        
        all_results[condition] = results
        print(f"{condition} - Acc: {results['accuracy']:.3f}, AUC: {results['auc']:.3f}")
    
    print("\nFinal Results:")
    for condition, results in all_results.items():
        print(f"{condition}: Acc={results['accuracy']:.3f}, F1={results['f1_score']:.3f}, AUC={results['auc']:.3f}")
        

if __name__ == "__main__":
    main()