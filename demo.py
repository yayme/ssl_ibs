"""
Demo script showing how to use train_ssl and test_ssl functions
"""

import numpy as np
import matplotlib.pyplot as plt
from ssl_utils import train_ssl, test_ssl

def generate_synthetic_signals(n_signals=1000, signal_length=300, n_channels=1):
    """Generate synthetic 1D time series signals for demonstration"""
    signals = []
    
    for i in range(n_signals):
        # Generate synthetic 1D time series data
        t = np.linspace(0, 10, signal_length)
        
        if n_channels == 1:
            # Single channel 1D signal
            signal = np.sin(2 * np.pi * 0.5 * t) + 0.5 * np.sin(2 * np.pi * 2 * t)
            signal += 0.3 * np.cos(2 * np.pi * 0.7 * t) + 0.2 * np.sin(2 * np.pi * 1.5 * t)
            
            # Add noise
            signal += np.random.normal(0, 0.1, signal_length)
            
            # Reshape to (1, signal_length) for single channel
            signal = signal.reshape(1, -1)
        else:
            # Multi-channel version (keeping original for compatibility)
            x = np.sin(2 * np.pi * 0.5 * t) + 0.5 * np.sin(2 * np.pi * 2 * t)
            y = np.cos(2 * np.pi * 0.7 * t) + 0.3 * np.cos(2 * np.pi * 1.5 * t)
            z = np.sin(2 * np.pi * 0.3 * t) + 0.4 * np.sin(2 * np.pi * 1.8 * t)
            
            # Add noise
            x += np.random.normal(0, 0.1, signal_length)
            y += np.random.normal(0, 0.1, signal_length)
            z += np.random.normal(0, 0.1, signal_length)
            
            # Stack channels
            signal = np.stack([x, y, z])  # Shape: (3, signal_length)
        
        signals.append(signal)
    
    return signals

def generate_downstream_data(n_signals=500, signal_length=300, n_channels=1):
    """Generate labeled 1D data for downstream classification"""
    signals = []
    targets = []
    
    for i in range(n_signals):
        t = np.linspace(0, 10, signal_length)
        
        if n_channels == 1:
            # Generate two different types of 1D signals
            if i < n_signals // 2:
                # Class 0: Lower frequency, more periodic
                signal = np.sin(2 * np.pi * 0.2 * t) + 0.3 * np.sin(2 * np.pi * 0.8 * t)
                signal += 0.2 * np.cos(2 * np.pi * 0.4 * t)
                targets.append(0)
            else:
                # Class 1: Higher frequency, more chaotic
                signal = np.sin(2 * np.pi * 1.5 * t) + 0.8 * np.sin(2 * np.pi * 3.2 * t)
                signal += 0.6 * np.cos(2 * np.pi * 2.1 * t) + 0.4 * np.sin(2 * np.pi * 4.5 * t)
                targets.append(1)
            
            # Add noise
            signal += np.random.normal(0, 0.2, signal_length)
            signal = signal.reshape(1, -1)
            
        else:
            # Multi-channel version
            if i < n_signals // 2:
                # Class 0: More periodic, lower frequency
                x = np.sin(2 * np.pi * 0.2 * t) + 0.3 * np.sin(2 * np.pi * 0.8 * t)
                y = np.cos(2 * np.pi * 0.3 * t) + 0.2 * np.cos(2 * np.pi * 1.0 * t)
                z = np.sin(2 * np.pi * 0.1 * t) + 0.4 * np.sin(2 * np.pi * 0.6 * t)
                targets.append(0)
            else:
                # Class 1: More chaotic, higher frequency
                x = np.sin(2 * np.pi * 1.5 * t) + 0.8 * np.sin(2 * np.pi * 3.2 * t)
                y = np.cos(2 * np.pi * 2.1 * t) + 0.6 * np.cos(2 * np.pi * 4.5 * t)
                z = np.sin(2 * np.pi * 1.8 * t) + 0.7 * np.sin(2 * np.pi * 3.8 * t)
                targets.append(1)
            
            # Add noise
            x += np.random.normal(0, 0.2, signal_length)
            y += np.random.normal(0, 0.2, signal_length)
            z += np.random.normal(0, 0.2, signal_length)
            
            signal = np.stack([x, y, z])
        
        signals.append(signal)
    
    return signals, targets

def demo_pipeline():
    """Demonstrate the complete SSL pipeline"""
    
    print("=== SSL Time Series Pipeline Demo ===\n")
    
    # Parameters
    signal_length = 300
    n_channels = 1  # Changed to 1 for true 1D time series
    
    # Step 1: Generate training data for SSL
    print("1. Generating synthetic 1D time series data for SSL training...")
    ssl_signals = generate_synthetic_signals(n_signals=800, signal_length=signal_length, n_channels=n_channels)
    print(f"Generated {len(ssl_signals)} 1D signals for SSL training")
    
    # Visualize some 1D signals
    plt.figure(figsize=(12, 6))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.plot(ssl_signals[i][0])  # Single channel signal
        plt.title(f'SSL Signal {i+1}')
        plt.grid(True)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
    
    # Step 2: Train SSL model
    print("\n2. Training SSL model...")
    model_path = train_ssl(
        signals=ssl_signals,
        signal_length=signal_length,
        epochs=20,  # Reduced for demo
        learning_rate=1e-4,
        batch_size=32,
        model_save_path="demo_ssl_model.pth"
    )
    
    # Step 3: Generate downstream data
    print("\n3. Generating labeled 1D data for downstream task...")
    test_signals, test_targets = generate_downstream_data(n_signals=400, signal_length=signal_length, n_channels=n_channels)
    print(f"Generated {len(test_signals)} labeled 1D signals for downstream task")
    
    # Visualize downstream 1D signals by class
    plt.figure(figsize=(12, 8))
    class_0_indices = [i for i, t in enumerate(test_targets) if t == 0]
    class_1_indices = [i for i, t in enumerate(test_targets) if t == 1]
    
    for i in range(3):
        plt.subplot(2, 3, i+1)
        idx = class_0_indices[i]
        plt.plot(test_signals[idx][0])  # Single channel
        plt.title(f'Class 0 Signal {i+1}')
        plt.grid(True)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        
        plt.subplot(2, 3, i+4)
        idx = class_1_indices[i]
        plt.plot(test_signals[idx][0])  # Single channel
        plt.title(f'Class 1 Signal {i+1}')
        plt.grid(True)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.savefig('demo_signals.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Step 4: Test SSL model on downstream task
    print("\n4. Testing SSL model on downstream classification...")
    results = test_ssl(
        test_signals=test_signals,
        signal_length=signal_length,
        targets=test_targets,
        model_path=model_path,
        epochs=15,  # Reduced for demo
        learning_rate=1e-3,
        batch_size=32,
        results_save_path="demo_results.json"
    )
    
    print(f"\n=== Final Demo Results ===")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"AUC: {results['auc']:.4f}")
    
    return results

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run demo
    results = demo_pipeline()
    
    print(f"\nDemo completed! Check the generated plots and saved models.")
