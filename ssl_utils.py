import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix, roc_curve
import os
import json
from tqdm import tqdm
import time
from ssl_new_pipeline import ResNetSSL, train_ssl_multitask, DownstreamClassifier, generate_pretext_labels_and_transform, compute_multitask_loss
from sklearn.metrics import average_precision_score
import random
import os

def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    print(f"[Seed set to {seed}] Random states synchronized.")
class SignalDataset(Dataset):
    def __init__(self, signals, signal_length, targets=None):
        """
        Dataset for list of signals
        Args:
            signals: List of numpy arrays or single numpy array
            signal_length: Length each signal should be (will pad/truncate)
            targets: List of target labels (for downstream task)
        """
        self.signals = signals
        self.signal_length = signal_length
        self.targets = targets
        
        # Process signals to ensure consistent format
        self.processed_signals = []
        for signal in signals:
            signal = np.array(signal)
            
            # Handle different input shapes
            if signal.ndim == 1:
                signal = signal.reshape(1, -1)  # (1, length)
            elif signal.ndim == 2:
                if signal.shape[0] > signal.shape[1]:
                    signal = signal.T  # Transpose to (channels, length)
            
            # Pad or truncate to target length
            current_length = signal.shape[1]
            if current_length < signal_length:
                # Pad with zeros
                pad_length = signal_length - current_length
                signal = np.pad(signal, ((0, 0), (0, pad_length)), mode='constant')
            elif current_length > signal_length:
                # Truncate
                signal = signal[:, :signal_length]
            
            self.processed_signals.append(signal.astype(np.float32))
    
    def __len__(self):
        return len(self.processed_signals)
    
    def __getitem__(self, idx):
        signal = torch.tensor(self.processed_signals[idx], dtype=torch.float)
        
        if self.targets is not None:
            return signal, self.targets[idx]
        return signal

def train_ssl(signals, signal_length, epochs=50, learning_rate=1e-4, batch_size=32, 
              model_name="ssl_model", config_path="config.json", device=None):
    """
    Train SSL model on list of time series signals
    
    Args:
        signals: List of numpy arrays (time series signals)
        signal_length: Target length for all signals
        epochs: Number of training epochs
        learning_rate: Learning rate for training
        batch_size: Batch size for training
        model_save_path: Path to save the trained model
        config_path: Path to config file for task settings
    
    Returns:
        str: Path to saved model
    """
    
    # Load configuration
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    
    # Extract SSL configuration directly from config (not from ssl_config sub-section)
    ssl_config = {
        'time_reversal': config.get('time_reversal', True),
        'scale': config.get('scale', True), 
        'permutation': config.get('permutation', True),
        'time_warped': config.get('time_warped', True),
        'positive_ratio': config.get('positive_ratio', 0.5)  # Probability of applying transformation (0.5 = 50% chance)
    }
    
    # EXPLANATION: positive_ratio controls the probability of applying each pretext task transformation
    # e.g., 0.5 means 50% of signals will be time-reversed, 50% won't be (for time_reversal task)
    # This creates balanced binary classification problems for each pretext task
    print(f"SSL Config: {ssl_config}")
    
    # Determine number of channels from first signal
    first_signal = np.array(signals[0])
    if first_signal.ndim == 1:
        n_channels = 1
    else:
        n_channels = min(first_signal.shape)  # Assume channels is the smaller dimension
    
    print(f"Detected {n_channels} channels")
    print(f"Target signal length: {signal_length}")
    print(f"Number of signals: {len(signals)}")
    
    # Create dataset and dataloader
    dataset = SignalDataset(signals, signal_length)
    # PyTorch DataLoader is used to load data in batches, allowing for efficient training
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Create model
    # Extract only the task names that are enabled (excluding positive_ratio which is not a task)
    task_names = ['time_reversal', 'scale', 'permutation', 'time_warped']
    active_tasks = [task for task in task_names if ssl_config.get(task, False)]
    
    print(f"Active SSL tasks: {active_tasks}")
    
    model = ResNetSSL(
        in_channels=n_channels,
        n_classes_per_task=2,
        tasks=active_tasks
    ).to(device)
    
    print(f"Training SSL model with tasks: {active_tasks}")
    print(f"Dataset: {len(signals)} signals, {n_channels} channels, length {signal_length}")
    print(f"Training: {epochs} epochs, batch size {batch_size}, lr {learning_rate}")
    
    # Train model
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    training_losses = []
    training_accs = []
    task_accuracies = {task: [] for task in active_tasks}
    
    print("\nStarting SSL Training...")
    start_time = time.time()
    
    # Main training loop with tqdm
    for epoch in tqdm(range(epochs), desc="SSL Epochs", unit="epoch"):
        epoch_start = time.time()
        total_loss = 0
        total_acc = 0
        task_losses = {task: 0 for task in active_tasks}
        task_accs = {task: 0 for task in active_tasks}
        
        # Batch loop with tqdm
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", 
                         leave=False, unit="batch")
        
        for batch_idx, x in enumerate(batch_pbar):
            if isinstance(x, list):
                x = x[0]
            
            # Move data to device
            x = x.to(device)
            
            # Generate pretext labels and transform data
            x_transformed, labels = generate_pretext_labels_and_transform(x, ssl_config)
            x_transformed = x_transformed.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(x_transformed)
            
            # Compute loss (enhanced with task-specific tracking)
            loss, acc, task_metrics = compute_multitask_loss_verbose(outputs, labels, ssl_config, active_tasks)
            
            # Update task-specific metrics
            for task, (task_loss, task_acc) in task_metrics.items():
                task_losses[task] += task_loss
                task_accs[task] += task_acc
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_acc += acc.item()
            
            # Update progress bar
            batch_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{acc.item():.4f}'
            })
        
        # Calculate epoch averages
        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)
        epoch_time = time.time() - epoch_start
        
        # Calculate task-specific averages
        for task in active_tasks:
            task_accuracies[task].append(task_accs[task] / len(dataloader))
        
        training_losses.append(avg_loss)
        training_accs.append(avg_acc)
        
        # Verbose epoch summary
        task_acc_str = " | ".join([f"{task}: {task_accs[task]/len(dataloader):.3f}" 
                                  for task in active_tasks])
        
        tqdm.write(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"Loss={avg_loss:.4f} | Acc={avg_acc:.4f} | "
                  f"{task_acc_str} | Time={epoch_time:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\nSSL Training Complete! Total time: {total_time:.1f}s")
    print(f"Final metrics: Loss={training_losses[-1]:.4f}, Acc={training_accs[-1]:.4f}")
    
    # Generate timestamp for file naming
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot training progress
    plot_ssl_training_progress(training_losses, training_accs, task_accuracies, active_tasks, model_name, timestamp)
    
    # Save model with timestamp
    timestamped_path = f"{model_name}_{timestamp}.pth"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'in_channels': n_channels,
            'tasks': active_tasks,
            'signal_length': signal_length
        },
        'training_config': {
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size
        },
        'timestamp': timestamp,
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }, timestamped_path)
    
    print(f"SSL model saved to: {timestamped_path}")
    print(f"Training completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Return the timestamped path
    final_model_path = timestamped_path
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(training_losses)
    plt.title('SSL Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(training_accs)
    plt.title('SSL Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    ssl_training_curves_path = f'{model_name}_ssl_training_curves_{timestamp}.png'
    plt.savefig(ssl_training_curves_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return final_model_path

def test_ssl(test_signals, signal_length, targets, model_path, 
             epochs=30, learning_rate=1e-3, batch_size=32,
             results_save_path="downstream_results.json", device=None):
    """
    Test SSL model on downstream classification task
    
    Args:
        test_signals: List of numpy arrays (test signals)
        signal_length: Length of each signal 
        targets: List of binary labels (0 or 1)
        model_path: Path to saved SSL model
        epochs: Number of fine-tuning epochs
        learning_rate: Learning rate for downstream training
        batch_size: Batch size
        results_save_path: Path to save results
    
    Returns:
        dict: Results containing metrics and plots
    """
    
    print(f"Loading SSL model from: {model_path}")
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Load pre-trained model
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    
    # Recreate SSL model
    ssl_model = ResNetSSL(
        in_channels=model_config['in_channels'],
        n_classes_per_task=2,
        tasks=model_config['tasks']
    )
    ssl_model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model with {model_config['in_channels']} channels and tasks: {model_config['tasks']}")
    print(f"Dataset: {len(test_signals)} signals, target length: {signal_length}")
    
    # Create dataset
    train_size = int(0.8 * len(test_signals))
    train_signals = test_signals[:train_size]
    train_targets = targets[:train_size]
    val_signals = test_signals[train_size:]
    val_targets = targets[train_size:]
    
    train_dataset = SignalDataset(train_signals, signal_length, train_targets)
    val_dataset = SignalDataset(val_signals, signal_length, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create downstream classifier
    downstream_model = DownstreamClassifier(ssl_model.feature_extractor, n_classes=2).to(device)
    
    # Training
    optimizer = optim.Adam(downstream_model.classifier.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    print(f"\nStarting downstream fine-tuning...")
    print(f"Train: {len(train_signals)} samples, Val: {len(val_signals)} samples")
    print(f"Training: {epochs} epochs, lr {learning_rate}")
    
    start_time = time.time()
    
    for epoch in tqdm(range(epochs), desc="Fine-tuning", unit="epoch"):
        epoch_start = time.time()
        
        # Training phase
        downstream_model.train()
        train_loss = 0
        train_acc = 0
        
        train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}", 
                         leave=False, unit="batch")
        
        for x, y in train_pbar:
            x, y = x.to(device), y.to(device)
            outputs = downstream_model(x)
            loss = criterion(outputs, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += (outputs.argmax(1) == y).float().mean().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{(outputs.argmax(1) == y).float().mean().item():.4f}'
            })
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # Validation phase
        downstream_model.eval()
        val_loss = 0
        val_acc = 0
        
        val_pbar = tqdm(val_loader, desc=f"Val Epoch {epoch+1}", 
                       leave=False, unit="batch")
        
        with torch.no_grad():
            for x, y in val_pbar:
                x, y = x.to(device), y.to(device)
                outputs = downstream_model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                val_acc += (outputs.argmax(1) == y).float().mean().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{(outputs.argmax(1) == y).float().mean().item():.4f}'
                })
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        epoch_time = time.time() - epoch_start
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Verbose epoch summary
        tqdm.write(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"Train Loss={train_loss:.4f} | Train Acc={train_acc:.4f} | "
                  f"Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f} | "
                  f"Time={epoch_time:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\nFine-tuning Complete! Total time: {total_time:.1f}s")
    
    # Final evaluation
    downstream_model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = downstream_model(x)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1
            all_targets.extend(y.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    auc = roc_auc_score(all_targets, all_probs)
    auprc = average_precision_score(all_targets, all_probs)
    
    print(f"\n=== Final Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training curves
    axes[0, 0].plot(train_losses, label='Train')
    axes[0, 0].plot(val_losses, label='Validation')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(train_accs, label='Train')
    axes[0, 1].plot(val_accs, label='Validation')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    # axes[1, 0].plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    axes[1, 0].plot(fpr, tpr, label=f'AUC = {auc:.3f}, accuracy= {accuracy:.3f}, f1score= {f1: .3f}, AUPRC={auprc:.3f}')
    axes[1, 0].plot([0, 1], [0, 1], 'k--')
    axes[1, 0].set_title('ROC Curve')
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    im = axes[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 1].set_title('Confusion Matrix')
    axes[1, 1].set_xlabel('Predicted Label')
    axes[1, 1].set_ylabel('True Label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1, 1].text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    unique_id = time.strftime("%Y%m%d_%H%M%S")
    downstream_results_path = f'{results_save_path[:-5]}_{unique_id}.png'
    plt.savefig(downstream_results_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save results
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'auc': auc,
        'auprc': auprc,
        'confusion_matrix': cm.tolist(),
        'training_config': {
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size
        }
    }
    
    unique_id = time.strftime("%Y%m%d_%H%M%S")
    results_save_path_unique = results_save_path.replace('.json', f'_{unique_id}.json')
    with open(results_save_path_unique, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_save_path_unique}")
    return results

def compute_multitask_loss_verbose(outputs, labels, task_config, active_tasks):
    """Enhanced loss computation with task-specific tracking"""
    import torch.nn as nn
    
    total_loss = 0
    total_acc = 0
    n_active_tasks = 0
    task_metrics = {}
    
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
            
            # Store task-specific metrics
            task_metrics[task] = (loss.item(), acc.item())
            
            n_active_tasks += 1
    
    if n_active_tasks > 0:
        total_loss /= n_active_tasks
        total_acc /= n_active_tasks
    
    return total_loss, total_acc, task_metrics

def plot_ssl_training_progress(training_losses, training_accs, task_accuracies, active_tasks, model_name="ssl_model", timestamp=None):
    """Plot comprehensive SSL training progress"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(training_losses) + 1)
    
    # Overall loss
    axes[0, 0].plot(epochs, training_losses, 'b-', linewidth=2)
    axes[0, 0].set_title('SSL Training Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(bottom=0)
    
    # Overall accuracy
    axes[0, 1].plot(epochs, training_accs, 'g-', linewidth=2)
    axes[0, 1].set_title('SSL Training Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    
    # Task-specific accuracies
    colors = ['red', 'blue', 'green', 'orange']
    for i, task in enumerate(active_tasks):
        if task in task_accuracies:
            axes[1, 0].plot(epochs, task_accuracies[task], 
                           color=colors[i % len(colors)], 
                           label=task.replace('_', ' ').title(),
                           linewidth=2)
    
    axes[1, 0].set_title('Task-Specific Accuracies', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    
    # Final metrics summary
    axes[1, 1].axis('off')
    summary_text = "Training Summary\n\n"
    summary_text += f"Final Loss: {training_losses[-1]:.4f}\n"
    summary_text += f"Final Accuracy: {training_accs[-1]:.4f}\n\n"
    summary_text += "Task Accuracies:\n"
    
    for task in active_tasks:
        if task in task_accuracies:
            final_acc = task_accuracies[task][-1]
            summary_text += f"• {task.replace('_', ' ').title()}: {final_acc:.3f}\n"
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, 
                   verticalalignment='center', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    
    plt.tight_layout()
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
    ssl_training_progress_path = f'{model_name}_ssl_training_progress_{timestamp}.png'
    plt.savefig(ssl_training_progress_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Training progress plot saved as '{ssl_training_progress_path}'")


def test_scratch_classifier(test_signals, signal_length, targets, epochs=50, 
                           learning_rate=1e-3, batch_size=16, 
                           results_save_path="scratch_classification_results.json", device=None):
    """
    Train and test a classifier from scratch (without SSL pretraining) for downstream task
    
    Args:
        test_signals: List of signals for classification
        signal_length: Length of each signal
        targets: List of target labels (0/1 for binary classification)
        epochs: Number of training epochs
        learning_rate: Learning rate for training
        batch_size: Batch size for training
        results_save_path: Path to save results
    
    Returns:
        dict: Dictionary containing accuracy, f1_score, auc and other metrics
    """
    print("🏗️ Training Classifier from Scratch (No SSL Pretraining)")
    print("="*70)
    
    # Determine number of channels from first signal
    first_signal = np.array(test_signals[0])
    if first_signal.ndim == 1:
        n_channels = 1
    else:
        n_channels = min(first_signal.shape)
    
    print(f"Dataset: {len(test_signals)} signals, {n_channels} channels, length {signal_length}")
    print(f"Training: {epochs} epochs, batch size {batch_size}, lr {learning_rate}")
    
    # Create dataset
    dataset = SignalDataset(test_signals, signal_length, targets)
    
    # Split into train/validation (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create a ResNet1D model from scratch for direct classification
    from models.resnet1d import resnet18_1d
    # Use the resnet18_1d factory to create the model with correct arguments
    scratch_model = resnet18_1d(in_channels=n_channels, num_classes=2)
    
    print(f"Created ResNet1D model with {sum(p.numel() for p in scratch_model.parameters()):,} parameters")
    print("All parameters are TRAINABLE 🔥")
    
    # Setup training
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scratch_model = scratch_model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(scratch_model.parameters(), lr=learning_rate)
    
    # Training metrics tracking
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    print(f"\nTraining on device: {device}")
    print("Starting training from scratch...")
    
    # Training loop
    for epoch in tqdm(range(epochs), desc="Training Epochs", unit="epoch"):
        # Training phase
        scratch_model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = scratch_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        
        # Validation phase
        scratch_model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = scratch_model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        
        # Calculate epoch metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = train_correct / train_total
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = val_correct / val_total
        
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
    
    # Final evaluation on validation set
    scratch_model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = scratch_model(batch_x)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1
            all_targets.extend(batch_y.cpu().numpy())
    
    # Calculate final metrics
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    auc = roc_auc_score(all_targets, all_probs)
    
    print(f"\n=== SCRATCH CLASSIFIER RESULTS ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training curves
    epochs_range = range(1, len(train_losses) + 1)
    
    axes[0, 0].plot(epochs_range, train_losses, label='Train', linewidth=2)
    axes[0, 0].plot(epochs_range, val_losses, label='Validation', linewidth=2)
    axes[0, 0].set_title('Loss Curves (Scratch Training)', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs_range, train_accs, label='Train', linewidth=2)
    axes[0, 1].plot(epochs_range, val_accs, label='Validation', linewidth=2)
    axes[0, 1].set_title('Accuracy Curves (Scratch Training)', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    axes[1, 0].plot(fpr, tpr, label=f'AUC = {auc:.3f}', linewidth=2)
    axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1, 0].set_title('ROC Curve (Scratch Training)', fontweight='bold')
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    im = axes[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 1].set_title('Confusion Matrix (Scratch Training)', fontweight='bold')
    axes[1, 1].set_xlabel('Predicted Label')
    axes[1, 1].set_ylabel('True Label')
    
    # Add text annotations to confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1, 1].text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black",
                           fontweight='bold')
    
    # Add colorbar
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    unique_id = time.strftime("%Y%m%d_%H%M%S")
    scratch_classifier_results_path = f'scratch_classifier_results_{unique_id}.png'
    plt.savefig(scratch_classifier_results_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save results
    results = {
        'method': 'scratch_classifier',
        'accuracy': accuracy,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist(),
        'training_config': {
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset)
        },
        'training_curves': {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs
        }
    }
    
    unique_id = time.strftime("%Y%m%d_%H%M%S")
    results_save_path_unique = results_save_path.replace('.json', f'_{unique_id}.json')
    with open(results_save_path_unique, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_save_path_unique}")
    print(f"Plot saved as '{scratch_classifier_results_path}'")
    return results


def tl_activity(test_signals, signal_length, targets, pretrained_model_path,
                epochs=50, learning_rate=1e-3, batch_size=16, 
                results_save_path="tl_classification_results.json", device=None):
    """
    Args:
        test_signals: List of signal arrays for classification
        signal_length: Length of each signal
        targets: List of binary targets (0 or 1 for AHI classification)
        pretrained_model_path: Path to pretrained model (.mdl file)
        epochs: Number of training epochs
        learning_rate: Learning rate for fine-tuning
        batch_size: Batch size for training
        results_save_path: Path to save results JSON
    
    Returns:
        dict: Results containing accuracy, f1_score, auc, and other metrics
    """
    from torch.utils.data import TensorDataset
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    print(f"🔧 Transfer Learning Activity")
    print(f"Device: {device}")
    print(f"Loading model: {pretrained_model_path}")
    
    # Load pretrained model
    checkpoint = torch.load(pretrained_model_path, map_location=device)
    print("✅ Loaded pretrained weights")
    
    class TL(nn.Module):
        """Transfer learning with linear input conversion"""
        def __init__(self, pretrained_model, input_length, n_classes=2):
            super().__init__()
            
            # Linear layer to convert (1, input_length) -> (3*input_length,)
            self.input_converter = nn.Linear(input_length, 3 * input_length)
            
            # Pretrained feature extractor (frozen)
            self.feature_extractor = pretrained_model
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            
            # Get feature dimension by test forward pass
            with torch.no_grad():
                test_input = torch.randn(1, 3, input_length)
                features = self.feature_extractor(test_input)
                if isinstance(features, (dict, list, tuple)):
                    features = list(features.values())[0] if isinstance(features, dict) else features[0]
                feature_dim = features.view(1, -1).shape[1]
            
            # Classifier head
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(feature_dim, n_classes)
            )
            
            print(f"📐 Input: ({input_length},) -> Linear -> (3, {input_length}) -> Features: {feature_dim} -> Classes: {n_classes}")
            print(f"❄️ Frozen parameters: {sum(p.numel() for p in self.feature_extractor.parameters()):,}")
            print(f"🔥 Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(feature_dim, n_classes)
            )
            
            print(f"� Input: ({input_length},) -> Linear -> (3, {input_length}) -> Features: {feature_dim} -> Classes: {n_classes}")
            print(f"❄️ Frozen pretrained parameters: {sum(p.numel() for p in self.feature_extractor.parameters()):,}")
            print(f"🔥 Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        
        def forward(self, x):
            # x shape: (batch, 1, length)
            batch_size, channels, length = x.shape
            
            # Flatten and convert: (batch, length) -> (batch, 3*length)
            x_flat = x.squeeze(1)  # (batch, length)
            x_converted = self.input_converter(x_flat)  # (batch, 3*length)
            
            # Reshape to (batch, 3, length)
            x_reshaped = x_converted.view(batch_size, 3, length)
            
            # Extract features
            features = self.feature_extractor(x_reshaped)
            if isinstance(features, (dict, list, tuple)):
                features = list(features.values())[0] if isinstance(features, dict) else features[0]
            
            # Flatten and classify
            features = features.view(batch_size, -1)
            output = self.classifier(features)
            return output
    
    # Prepare data
    X = torch.stack([torch.FloatTensor(signal) for signal in test_signals])
    y = torch.LongTensor(targets)
    
    # Train/validation split (80/20)
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"📊 Data: {len(X)} total, {len(X_train)} train, {len(X_val)} val")
    print(f"Signal shape: {X.shape}, Target distribution: {np.bincount(targets)}")
    
    # Create model
    from models.resnet1d import resnet18_1d
    
    # Create base model and remove final classifier
    base_model = resnet18_1d(in_channels=3, num_classes=1000)
    base_model = nn.Sequential(*list(base_model.children())[:-1])  # Remove fc layer
    
    # Load pretrained weights
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
    else:
        state_dict = checkpoint
    base_model.load_state_dict(state_dict, strict=False)
    
    model = TL(base_model, signal_length, n_classes=2)
    model = model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    
    print(f"🚀 Training for {epochs} epochs...")
    
    # Training loop
    for epoch in tqdm(range(epochs), desc="Training"):
        # Train
        model.train()
        train_loss = train_correct = train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        
        # Validate
        model.eval()
        val_loss = val_correct = val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        
        # Store metrics
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_correct / train_total)
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_correct / val_total)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Acc: {train_accs[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}")
    
    # Final evaluation
    model.eval()
    all_preds, all_probs, all_targets = [], [], []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    auc = roc_auc_score(all_targets, all_probs)
    
    print(f"\n🎯 Results: Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    
    # Plot results (same structure as other functions)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training curves
    axes[0, 0].plot(train_losses, label='Train', linewidth=2)
    axes[0, 0].plot(val_losses, label='Val', linewidth=2)
    axes[0, 0].set_title('Loss (Transfer Learning)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(train_accs, label='Train', linewidth=2)
    axes[0, 1].plot(val_accs, label='Val', linewidth=2)
    axes[0, 1].set_title('Accuracy (Transfer Learning)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    axes[1, 0].plot(fpr, tpr, label=f'AUC = {auc:.3f}', linewidth=2)
    axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1, 0].set_title('ROC Curve')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    axes[1, 1].imshow(cm, cmap='Blues')
    axes[1, 1].set_title('Confusion Matrix')
    for i in range(2):
        for j in range(2):
            axes[1, 1].text(j, i, str(cm[i, j]), ha='center', va='center')
    
    plt.tight_layout()
    plot_filename = f"tl_results_{timestamp}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save results
    results = {
        'method': 'transfer_learning',
        'accuracy': accuracy,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist(),
        'model_config': {
            'pretrained_model_path': pretrained_model_path,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'signal_length': signal_length
        }
    }
    
    results_filename = results_save_path.replace('.json', f'_{timestamp}.json')
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved: {results_filename}")
    print(f"📊 Plot saved: {plot_filename}")
    
    return results
#reference ssl-wearables/mtl.py
def train_mtl_ssl(signals, signal_length, epochs=50, learning_rate=1e-4, batch_size=32,
                  model_name="mtl_ssl_model", config_path="config.json", device=None, option=1):
    import sys
    from sslearning_uk.models.accNet import Resnet
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    task_config = {
        'time_reversal': config.get('time_reversal', True),
        'scale': config.get('scale', True),
        'permutation': config.get('permutation', True),
        'time_warped': config.get('time_warped', True),
        'positive_ratio': config.get('positive_ratio', 0.5)
    }
    
    first_signal = np.array(signals[0])
    if first_signal.ndim == 1:
        n_channels = 1
    else:
        n_channels = first_signal.shape[0]
    
    print(f"Detected {n_channels} channels")
    print(f"Target signal length: {signal_length}")
    print(f"Number of signals: {len(signals)}")
    
    dataset = SignalDataset(signals, signal_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    task_names = ['time_reversal', 'scale', 'permutation', 'time_warped']
    active_tasks = [task for task in task_names if task_config.get(task, False)]
    
    print(f"Active SSL tasks: {active_tasks}")
    
    model = Resnet(
        output_size=2,
        n_channels=n_channels,
        resnet_version=18,
        epoch_len=signal_length // 30,
        is_mtl=True,
        option=1
    ).to(device)
    
    print(f"Training MTL SSL model with tasks: {active_tasks}")
    print(f"Dataset: {len(signals)} signals, {n_channels} channels, length {signal_length}")
    print(f"Training: {epochs} epochs, batch size {batch_size}, lr {learning_rate}")
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate  , amsgrad=True)
    criterion = nn.CrossEntropyLoss()
    
    training_losses = []
    training_accs = []
    task_accuracies = {task: [] for task in active_tasks}
    
    print("\nStarting MTL SSL Training...")
    start_time = time.time()
    
    for epoch in tqdm(range(epochs), desc="MTL SSL Epochs", unit="epoch"):
        epoch_loss = 0
        epoch_acc = 0
        task_accs = {task: 0 for task in active_tasks}
        n_batches = 0
        
        for batch_signals in dataloader:
            batch_signals = batch_signals.to(device)
            
            labels = []
            transformed_signals = []
            
            for i in range(batch_signals.size(0)):
                signal = batch_signals[i].cpu().numpy()
                
                task_labels = []
                for task in task_names:
                    if task_config.get(task, False):
                        choice = np.random.choice(2, p=[task_config['positive_ratio'], 1 - task_config['positive_ratio']])
                        task_labels.append(choice)
                        
                        if task == 'time_reversal' and choice == 1:
                            signal = np.flip(signal, axis=-1).copy()
                        elif task == 'scale' and choice == 1:
                            signal = signal * np.random.uniform(0.5, 2.0)
                        elif task == 'permutation' and choice == 1:
                            n_segments = 5
                            segment_len = signal.shape[-1] // n_segments
                            segments = [signal[..., i*segment_len:(i+1)*segment_len] for i in range(n_segments)]
                            np.random.shuffle(segments)
                            signal = np.concatenate(segments, axis=-1)
                        elif task == 'time_warped' and choice == 1:
                            warp_factor = np.random.uniform(0.8, 1.2)
                            old_len = signal.shape[-1]
                            new_len = int(old_len * warp_factor)
                            signal_tensor = torch.tensor(signal).unsqueeze(0)
                            warped = torch.nn.functional.interpolate(signal_tensor, size=new_len, mode='linear', align_corners=False)
                            if new_len > old_len:
                                signal = warped[0, :, :old_len].numpy()
                            else:
                                pad_len = old_len - new_len
                                signal = torch.nn.functional.pad(warped[0], (0, pad_len)).numpy()
                    else:
                        task_labels.append(0)
                
                labels.append(task_labels)
                transformed_signals.append(signal)
            
            transformed_signals = torch.tensor(np.array(transformed_signals), dtype=torch.float32).to(device)
            labels = torch.tensor(np.array(labels), dtype=torch.long).to(device)
            
            aot_y_pred, scale_y_pred, permute_y_pred, time_w_h_pred = model(transformed_signals)
            
            total_loss = 0
            total_acc = 0
            n_active = 0
            
            active_task_idx = 0
            for task in task_names:
                if task_config.get(task, False):
                    if task == 'time_reversal':
                        pred = aot_y_pred
                    elif task == 'scale':
                        pred = scale_y_pred
                    elif task == 'permutation':
                        pred = permute_y_pred
                    elif task == 'time_warped':
                        pred = time_w_h_pred
                    
                    task_label = labels[:, active_task_idx]
                    
                    loss = criterion(pred, task_label)
                    acc = (pred.argmax(dim=1) == task_label).float().mean()
                    
                    total_loss += loss
                    total_acc += acc
                    task_accs[task] += acc.item()
                    n_active += 1
                    active_task_idx += 1
            
            if n_active > 0:
                total_loss = total_loss / n_active
                total_acc = total_acc / n_active
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_acc += total_acc.item()
            n_batches += 1
        
        epoch_loss /= n_batches
        epoch_acc /= n_batches
        
        training_losses.append(epoch_loss)
        training_accs.append(epoch_acc)
        
        for task in active_tasks:
            task_accuracies[task].append(task_accs[task] / n_batches)
    
    total_time = time.time() - start_time
    print(f"\nMTL SSL Training Complete! Total time: {total_time:.1f}s")
    print(f"Final metrics: Loss={training_losses[-1]:.4f}, Acc={training_accs[-1]:.4f}")
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plot_ssl_training_progress(training_losses, training_accs, task_accuracies, active_tasks, model_name, timestamp)
    
    timestamped_path = f"{model_name}_{timestamp}.pth"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'n_channels': n_channels,
            'tasks': active_tasks,
            'signal_length': signal_length,
            'resnet_version': 18,
            'epoch_len': signal_length // 30
        },
        'training_config': {
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size
        },
        'timestamp': timestamp,
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }, timestamped_path)
    
    print(f"MTL SSL model saved to: {timestamped_path}")
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(training_losses)
    plt.title('MTL SSL Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(training_accs)
    plt.title('MTL SSL Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    ssl_training_curves_path = f'{model_name}_mtl_ssl_training_curves_{timestamp}.png'
    plt.savefig(ssl_training_curves_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return timestamped_path
