"""
Main training script for few-shot learning with Prototypical Networks.
"""
import os
import sys
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.feature_extractor import create_feature_extractor
from src.models.prototypical_network import (
    PrototypicalNetwork,
    prototypical_loss,
    compute_accuracy
)
from src.data.dataset import (
    MiniImageNetDataset,
    FewShotSampler,
    get_transforms,
    create_dummy_dataset
)
from src.utils.mlflow_utils import MLflowTracker, load_config


def train_episode(
    model: PrototypicalNetwork,
    sampler: FewShotSampler,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_way: int
) -> tuple:
    """
    Train on a single episode.
    
    Args:
        model: Prototypical network
        sampler: Few-shot episode sampler
        optimizer: Optimizer
        device: Device to use
        n_way: Number of classes per episode
    
    Returns:
        loss, accuracy
    """
    model.train()
    
    # Sample episode
    support_images, support_labels, query_images, query_labels = sampler.sample_episode()
    
    # Move to device
    support_images = support_images.to(device)
    support_labels = support_labels.to(device)
    query_images = query_images.to(device)
    query_labels = query_labels.to(device)
    
    # Forward pass
    logits, _ = model(support_images, support_labels, query_images, n_way)
    
    # Compute loss
    loss = prototypical_loss(logits, query_labels)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Compute accuracy
    accuracy = compute_accuracy(logits, query_labels)
    
    return loss.item(), accuracy


def evaluate(
    model: PrototypicalNetwork,
    sampler: FewShotSampler,
    device: torch.device,
    n_way: int,
    num_episodes: int = 100
) -> tuple:
    """
    Evaluate the model on multiple episodes.
    
    Args:
        model: Prototypical network
        sampler: Few-shot episode sampler
        device: Device to use
        n_way: Number of classes per episode
        num_episodes: Number of episodes to evaluate
    
    Returns:
        mean_loss, mean_accuracy, std_accuracy
    """
    model.eval()
    
    losses = []
    accuracies = []
    
    with torch.no_grad():
        for _ in tqdm(range(num_episodes), desc="Evaluating"):
            # Sample episode
            support_images, support_labels, query_images, query_labels = sampler.sample_episode()
            
            # Move to device
            support_images = support_images.to(device)
            support_labels = support_labels.to(device)
            query_images = query_images.to(device)
            query_labels = query_labels.to(device)
            
            # Forward pass
            logits, _ = model(support_images, support_labels, query_images, n_way)
            
            # Compute loss and accuracy
            loss = prototypical_loss(logits, query_labels)
            accuracy = compute_accuracy(logits, query_labels)
            
            losses.append(loss.item())
            accuracies.append(accuracy)
    
    mean_loss = np.mean(losses)
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    return mean_loss, mean_accuracy, std_accuracy


def main():
    parser = argparse.ArgumentParser(description='Train Prototypical Network for Few-Shot Learning')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--create-dummy-data', action='store_true',
                        help='Create dummy dataset for testing')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print("Configuration loaded:")
    print(f"  Model: {config['model']['feature_extractor']}")
    print(f"  Training: {config['training']['num_episodes']} episodes")
    print(f"  N-way: {config['training']['n_way']}, K-shot: {config['training']['k_shot']}")
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create dummy dataset if requested
    data_root = config['dataset']['root']
    if args.create_dummy_data:
        print("\nCreating dummy dataset...")
        create_dummy_dataset(data_root, num_classes=50, images_per_class=100)
    
    # Load datasets
    print("\nLoading datasets...")
    train_transform = get_transforms(config, is_training=True)
    val_transform = get_transforms(config, is_training=False)
    
    train_dataset = MiniImageNetDataset(data_root, split='train', transform=train_transform)
    val_dataset = MiniImageNetDataset(data_root, split='val', transform=val_transform)
    
    if len(train_dataset.classes) == 0:
        print("\nError: No training data found!")
        print("Please either:")
        print("  1. Download mini-ImageNet dataset and place it in the correct directory")
        print("  2. Use --create-dummy-data flag to create a dummy dataset for testing")
        return
    
    # Create samplers
    train_sampler = FewShotSampler(
        train_dataset,
        n_way=config['training']['n_way'],
        k_shot=config['training']['k_shot'],
        n_query=config['training']['n_query'],
        transform_support=train_transform,
        transform_query=train_transform
    )
    
    val_sampler = FewShotSampler(
        val_dataset,
        n_way=config['training']['n_way'],
        k_shot=config['training']['k_shot'],
        n_query=config['training']['n_query'],
        transform_support=val_transform,
        transform_query=val_transform
    )
    
    # Create model
    print("\nCreating model...")
    feature_extractor = create_feature_extractor(
        model_name=config['model']['feature_extractor'],
        freeze=config['model']['freeze_backbone'],
        embedding_dim=config['model']['embedding_dim']
    )
    
    model = PrototypicalNetwork(feature_extractor)
    model = model.to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training']['step_size'],
        gamma=config['training']['gamma']
    )
    
    # Initialize MLflow tracker
    tracker = MLflowTracker(config)
    tracker.start_run(run_name=f"{config['training']['n_way']}-way-{config['training']['k_shot']}-shot")
    
    # Training loop
    print("\nStarting training...")
    num_episodes = config['training']['num_episodes']
    n_way = config['training']['n_way']
    
    best_val_accuracy = 0.0
    
    for episode in range(num_episodes):
        # Train on episode
        loss, accuracy = train_episode(model, train_sampler, optimizer, device, n_way)
        
        # Log metrics
        tracker.log_metrics({
            'train/loss': loss,
            'train/accuracy': accuracy,
            'learning_rate': optimizer.param_groups[0]['lr']
        }, step=episode)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes} - Loss: {loss:.4f}, Acc: {accuracy:.2f}%")
        
        # Validation
        if (episode + 1) % 500 == 0:
            print("\nRunning validation...")
            val_loss, val_accuracy, val_std = evaluate(
                model, val_sampler, device, n_way, num_episodes=100
            )
            
            print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_accuracy:.2f}% ± {val_std:.2f}%")
            
            tracker.log_metrics({
                'val/loss': val_loss,
                'val/accuracy': val_accuracy,
                'val/accuracy_std': val_std
            }, step=episode)
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                checkpoint_path = f"checkpoints/best_model_{config['training']['n_way']}way_{config['training']['k_shot']}shot.pth"
                tracker.save_model_checkpoint(
                    model, optimizer, episode,
                    {'val_accuracy': val_accuracy},
                    checkpoint_path
                )
                tracker.set_tag("best_val_accuracy", val_accuracy)
        
        # Step scheduler
        scheduler.step()
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    
    # Final evaluation
    print("\nRunning final evaluation...")
    val_loss, val_accuracy, val_std = evaluate(
        model, val_sampler, device, n_way, num_episodes=600
    )
    print(f"Final validation - Loss: {val_loss:.4f}, Acc: {val_accuracy:.2f}% ± {val_std:.2f}%")
    
    tracker.log_metrics({
        'final/val_loss': val_loss,
        'final/val_accuracy': val_accuracy,
        'final/val_accuracy_std': val_std
    })
    
    # End MLflow run
    tracker.end_run()


if __name__ == '__main__':
    main()
