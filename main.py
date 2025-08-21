import json
import os
import torch
from torch.utils.data import DataLoader
from ssl_new_pipeline import ResNetSSL, train_ssl_multitask, DownstreamClassifier, train_downstream_classifier
from dataset import TimeSeriesDataset, SyntheticTimeSeriesDataset

def load_config(config_path):
    with open(config_path) as f:
        return json.load(f)

def create_ssl_dataloader(config, use_synthetic=True):
    if use_synthetic:
        dataset = SyntheticTimeSeriesDataset(
            n_samples=config.get('ssl_samples', 1000),
            length=config.get('sequence_length', 300),
            n_channels=config.get('n_channels', 3),
            labeled=False
        )
    else:
        dataset = TimeSeriesDataset(
            config["train_data_dir"],
            file_list=config.get("train_file_list", None)
        )
    
    return DataLoader(
        dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=True,
        num_workers=config.get('num_workers', 0)
    )

def create_downstream_dataloaders(config, use_synthetic=True):
    if use_synthetic:
        train_dataset = SyntheticTimeSeriesDataset(
            n_samples=config.get('downstream_train_samples', 500),
            length=config.get('sequence_length', 300),
            n_channels=config.get('n_channels', 3),
            n_classes=config.get('n_classes', 2),
            labeled=True
        )
        test_dataset = SyntheticTimeSeriesDataset(
            n_samples=config.get('downstream_test_samples', 200),
            length=config.get('sequence_length', 300),
            n_channels=config.get('n_channels', 3),
            n_classes=config.get('n_classes', 2),
            labeled=True
        )
    else:
        train_dataset = TimeSeriesDataset(
            config["test_data_dir"],
            labeled=True
        )
        test_dataset = TimeSeriesDataset(
            config["test_data_dir"],
            labeled=True
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=True,
        num_workers=config.get('num_workers', 0)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        num_workers=config.get('num_workers', 0)
    )
    
    return train_loader, test_loader

def main():
    config = load_config("config.json")
    
    # Task configuration for pretext tasks
    task_config = {
        'time_reversal': config.get('time_reversal', True),
        'scale': config.get('scale', True),
        'permutation': config.get('permutation', True),
        'time_warped': config.get('time_warped', True),
        'positive_ratio': config.get('positive_ratio', 0.5)
    }
    
    # Active tasks
    active_tasks = [task for task, active in task_config.items() 
                   if active and task != 'positive_ratio']
    
    print(f"Training SSL model with tasks: {active_tasks}")
    
    # Create SSL model
    model = ResNetSSL(
        in_channels=config.get('n_channels', 3),
        n_classes_per_task=2,
        tasks=active_tasks
    )
    
    # Create SSL dataloader
    ssl_loader = create_ssl_dataloader(config, use_synthetic=True)
    
    # Train SSL model
    print("=== SSL Training ===")
    train_ssl_multitask(
        model, 
        ssl_loader, 
        task_config,
        epochs=config.get('ssl_epochs', 50),
        lr=config.get('ssl_lr', 1e-4)
    )
    
    # Save SSL model
    ssl_model_path = config.get('ssl_model_path', 'ssl_model.pth')
    torch.save(model.state_dict(), ssl_model_path)
    print(f"SSL model saved to {ssl_model_path}")
    
    # Create downstream classifier
    downstream_model = DownstreamClassifier(
        model.feature_extractor,
        n_classes=config.get('n_classes', 2)
    )
    
    # Create downstream dataloaders
    train_loader, test_loader = create_downstream_dataloaders(config, use_synthetic=True)
    
    # Train downstream classifier
    print("\n=== Downstream Training ===")
    train_downstream_classifier(
        downstream_model,
        train_loader,
        test_loader,
        epochs=config.get('downstream_epochs', 30),
        lr=config.get('downstream_lr', 1e-3)
    )
    
    # Save downstream model
    downstream_model_path = config.get('downstream_model_path', 'downstream_model.pth')
    torch.save(downstream_model.state_dict(), downstream_model_path)
    print(f"Downstream model saved to {downstream_model_path}")
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()
