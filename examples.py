"""
Example script demonstrating the few-shot learning system.
This script shows how to use the system for training and inference.
"""
import os
import sys
import torch

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.data.dataset import create_dummy_dataset, MiniImageNetDataset, FewShotSampler, get_transforms
from src.models.feature_extractor import create_feature_extractor
from src.models.prototypical_network import PrototypicalNetwork, prototypical_loss, compute_accuracy
from src.utils.mlflow_utils import load_config


def example_1_create_dummy_data():
    """Example 1: Create dummy dataset for testing."""
    print("\n" + "="*60)
    print("Example 1: Creating Dummy Dataset")
    print("="*60)
    
    data_root = './data'
    create_dummy_dataset(
        data_root=data_root,
        num_classes=30,
        images_per_class=100
    )
    
    print("\n✓ Dummy dataset created successfully!")
    print(f"  Location: {data_root}/mini-imagenet/")
    print("  Splits: train/, val/, test/")


def example_2_load_and_sample_episode():
    """Example 2: Load dataset and sample a few-shot episode."""
    print("\n" + "="*60)
    print("Example 2: Loading Dataset and Sampling Episode")
    print("="*60)
    
    # Load configuration
    config = load_config('configs/config.yaml')
    
    # Create transforms
    transform = get_transforms(config, is_training=True)
    
    # Load dataset
    dataset = MiniImageNetDataset(
        data_root='./data',
        split='train',
        transform=transform
    )
    
    print(f"\nDataset loaded:")
    print(f"  - Number of classes: {len(dataset.classes)}")
    
    if len(dataset.classes) == 0:
        print("\n⚠ No data found. Run example_1_create_dummy_data() first!")
        return None, None
    
    # Create sampler
    sampler = FewShotSampler(
        dataset=dataset,
        n_way=5,
        k_shot=5,
        n_query=15,
        transform_support=transform,
        transform_query=transform
    )
    
    # Sample an episode
    support_images, support_labels, query_images, query_labels = sampler.sample_episode()
    
    print(f"\nEpisode sampled:")
    print(f"  - Support set shape: {support_images.shape}")
    print(f"  - Support labels: {support_labels[:10]}...")
    print(f"  - Query set shape: {query_images.shape}")
    print(f"  - Query labels: {query_labels[:10]}...")
    
    return sampler, config


def example_3_create_and_train_model():
    """Example 3: Create model and train on one episode."""
    print("\n" + "="*60)
    print("Example 3: Creating and Training Model")
    print("="*60)
    
    # Get sampler from previous example
    sampler, config = example_2_load_and_sample_episode()
    
    if sampler is None:
        return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create feature extractor
    print("\nCreating feature extractor...")
    print("  Note: This will download the pre-trained ViT model (~350MB)")
    print("  Only needed once - model will be cached locally")
    
    try:
        feature_extractor = create_feature_extractor(
            model_name=config['model']['feature_extractor'],
            freeze=True,
            embedding_dim=config['model']['embedding_dim']
        )
        
        # Create prototypical network
        model = PrototypicalNetwork(feature_extractor)
        model = model.to(device)
        
        print(f"\n✓ Model created successfully!")
        print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train on a few episodes
        print("\nTraining on 5 episodes...")
        model.train()
        
        for episode in range(5):
            # Sample episode
            support_images, support_labels, query_images, query_labels = sampler.sample_episode()
            
            # Move to device
            support_images = support_images.to(device)
            support_labels = support_labels.to(device)
            query_images = query_images.to(device)
            query_labels = query_labels.to(device)
            
            # Forward pass
            logits, _ = model(support_images, support_labels, query_images, n_way=5)
            
            # Compute loss
            loss = prototypical_loss(logits, query_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Compute accuracy
            accuracy = compute_accuracy(logits, query_labels)
            
            print(f"  Episode {episode+1}/5 - Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
        
        print("\n✓ Training completed!")
        
    except Exception as e:
        print(f"\n⚠ Error: {e}")
        print("  This is expected if running without internet to download the model.")


def example_4_inference_workflow():
    """Example 4: Demonstrate inference workflow (class registration)."""
    print("\n" + "="*60)
    print("Example 4: Inference Workflow")
    print("="*60)
    
    print("\nInference workflow steps:")
    print("  1. Load trained model checkpoint")
    print("  2. Register new classes with K support images")
    print("  3. Classify query images against registered classes")
    
    print("\nExample API usage:")
    print("\n  # Register a new class")
    print('  curl -X POST "http://localhost:8000/register_class" \\')
    print('       -F "class_name=defect_type_A" \\')
    print('       -F "images=@image1.jpg" \\')
    print('       -F "images=@image2.jpg"')
    
    print("\n  # Classify an image")
    print('  curl -X POST "http://localhost:8000/classify_image" \\')
    print('       -F "image=@test_image.jpg" \\')
    print('       -F "top_k=3"')
    
    print("\n  # List registered classes")
    print('  curl "http://localhost:8000/list_classes"')


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print(" Few-Shot Learning System - Examples")
    print("="*70)
    
    print("\nThis script demonstrates the key components of the system:")
    print("  1. Creating dummy datasets")
    print("  2. Loading and sampling episodes")
    print("  3. Creating and training models")
    print("  4. Inference workflow")
    
    # Example 1: Create dummy data
    print("\n\nWould you like to create a dummy dataset? (y/n)")
    print("Note: Skip if you already have data or want to see other examples")
    response = input("> ").strip().lower()
    
    if response == 'y':
        example_1_create_dummy_data()
    
    # Example 2 and 3: Load data and show model creation
    example_2_load_and_sample_episode()
    
    # Example 4: Inference
    example_4_inference_workflow()
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70)
    
    print("\nNext steps:")
    print("  - Train a full model: python scripts/train.py --create-dummy-data")
    print("  - Evaluate model: python scripts/evaluate.py --checkpoint <path>")
    print("  - Start API server: python src/api/app.py")
    print("  - View MLflow UI: mlflow ui --backend-store-uri ./mlruns")


if __name__ == '__main__':
    # For non-interactive mode, just show examples without prompts
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        print("\nRunning in demo mode (no interactive prompts)...")
        example_2_load_and_sample_episode()
        example_4_inference_workflow()
    else:
        main()
