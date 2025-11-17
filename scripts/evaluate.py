"""
Evaluation script for few-shot learning model.
Evaluates on different N-way K-shot scenarios.
"""
import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    get_transforms
)
from src.utils.mlflow_utils import load_config
from src.utils.gradcam import (
    GradCAM,
    visualize_gradcam,
    get_target_layer_vit
)


def evaluate_n_way_k_shot(
    model: PrototypicalNetwork,
    dataset: MiniImageNetDataset,
    transform,
    n_way: int,
    k_shot: int,
    n_query: int,
    num_episodes: int,
    device: torch.device
) -> tuple:
    """
    Evaluate model on N-way K-shot classification.
    
    Returns:
        mean_accuracy, std_accuracy, all_accuracies
    """
    print(f"\nEvaluating {n_way}-way {k_shot}-shot classification...")
    
    sampler = FewShotSampler(
        dataset,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        transform_support=transform,
        transform_query=transform
    )
    
    model.eval()
    accuracies = []
    
    with torch.no_grad():
        for _ in tqdm(range(num_episodes), desc=f"{n_way}-way {k_shot}-shot"):
            # Sample episode
            support_images, support_labels, query_images, query_labels = sampler.sample_episode()
            
            # Move to device
            support_images = support_images.to(device)
            support_labels = support_labels.to(device)
            query_images = query_images.to(device)
            query_labels = query_labels.to(device)
            
            # Forward pass
            logits, _ = model(support_images, support_labels, query_images, n_way)
            
            # Compute accuracy
            accuracy = compute_accuracy(logits, query_labels)
            accuracies.append(accuracy)
    
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    # 95% confidence interval
    confidence_interval = 1.96 * std_accuracy / np.sqrt(num_episodes)
    
    print(f"Results: {mean_accuracy:.2f}% ± {confidence_interval:.2f}%")
    
    return mean_accuracy, std_accuracy, accuracies


def generate_gradcam_examples(
    model: PrototypicalNetwork,
    dataset: MiniImageNetDataset,
    transform,
    n_way: int,
    k_shot: int,
    device: torch.device,
    save_dir: str,
    num_examples: int = 5
):
    """
    Generate Grad-CAM visualizations for sample episodes.
    """
    print(f"\nGenerating {num_examples} Grad-CAM examples...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    sampler = FewShotSampler(
        dataset,
        n_way=n_way,
        k_shot=k_shot,
        n_query=1,
        transform_support=transform,
        transform_query=transform
    )
    
    # Get target layer for Grad-CAM
    try:
        target_layer = get_target_layer_vit(model.feature_extractor)
        gradcam = GradCAM(model.feature_extractor, target_layer)
    except Exception as e:
        print(f"Warning: Could not create Grad-CAM: {e}")
        print("Skipping Grad-CAM visualization")
        return
    
    model.eval()
    
    for i in range(num_examples):
        try:
            # Sample episode
            support_images, support_labels, query_images, query_labels = sampler.sample_episode()
            
            # Move to device
            support_images = support_images.to(device)
            query_images = query_images.to(device)
            
            # Get prediction
            with torch.no_grad():
                logits, _ = model(support_images, support_labels, query_images, n_way)
                pred_class = torch.argmax(logits, dim=1).item()
                true_class = query_labels[0].item()
            
            # Generate Grad-CAM
            query_image_input = query_images[0:1]  # [1, 3, H, W]
            
            # Enable gradients for Grad-CAM
            query_image_input.requires_grad = True
            
            # Extract features and get CAM
            cam = gradcam.generate_cam(query_image_input, target_class=None)
            
            # Visualize
            save_path = os.path.join(save_dir, f'gradcam_example_{i+1}.png')
            visualize_gradcam(
                query_images[0].cpu(),
                cam,
                save_path=save_path
            )
            
            print(f"  Example {i+1}: True class={true_class}, Predicted={pred_class}, "
                  f"Correct={'✓' if pred_class == true_class else '✗'}")
            
        except Exception as e:
            print(f"  Warning: Failed to generate Grad-CAM for example {i+1}: {e}")
            continue
    
    print(f"Grad-CAM examples saved to {save_dir}")


def plot_results(results: dict, save_path: str):
    """
    Plot evaluation results.
    
    Args:
        results: Dictionary with results for different scenarios
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scenarios = []
    accuracies = []
    errors = []
    
    for (n_way, k_shot), (mean_acc, std_acc) in results.items():
        scenarios.append(f"{n_way}-way\n{k_shot}-shot")
        accuracies.append(mean_acc)
        # 95% confidence interval
        errors.append(1.96 * std_acc / np.sqrt(600))
    
    x = np.arange(len(scenarios))
    ax.bar(x, accuracies, yerr=errors, capsize=5, alpha=0.7, color='steelblue')
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Few-Shot Classification Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (acc, err) in enumerate(zip(accuracies, errors)):
        ax.text(i, acc + err + 1, f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Results plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Few-Shot Learning Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--num-episodes', type=int, default=600,
                        help='Number of episodes for evaluation')
    parser.add_argument('--gradcam', action='store_true',
                        help='Generate Grad-CAM visualizations')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='Directory to save results')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_transform = get_transforms(config, is_training=False)
    test_dataset = MiniImageNetDataset(
        config['dataset']['root'],
        split='test',
        transform=test_transform
    )
    
    if len(test_dataset.classes) == 0:
        print("\nError: No test data found!")
        return
    
    print(f"Test dataset: {len(test_dataset.classes)} classes")
    
    # Create model
    print("\nCreating model...")
    feature_extractor = create_feature_extractor(
        model_name=config['model']['feature_extractor'],
        freeze=config['model']['freeze_backbone'],
        embedding_dim=config['model']['embedding_dim']
    )
    
    model = PrototypicalNetwork(feature_extractor)
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Loaded checkpoint from episode {checkpoint.get('epoch', 'unknown')}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate on different N-way K-shot scenarios
    results = {}
    
    for n_way in config['evaluation']['n_way_eval']:
        for k_shot in config['evaluation']['k_shot_eval']:
            mean_acc, std_acc, _ = evaluate_n_way_k_shot(
                model=model,
                dataset=test_dataset,
                transform=test_transform,
                n_way=n_way,
                k_shot=k_shot,
                n_query=config['training']['n_query'],
                num_episodes=args.num_episodes,
                device=device
            )
            
            results[(n_way, k_shot)] = (mean_acc, std_acc)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for (n_way, k_shot), (mean_acc, std_acc) in results.items():
        confidence = 1.96 * std_acc / np.sqrt(args.num_episodes)
        print(f"{n_way}-way {k_shot}-shot: {mean_acc:.2f}% ± {confidence:.2f}%")
    print("="*60)
    
    # Plot results
    plot_path = os.path.join(args.output_dir, 'evaluation_results.png')
    plot_results(results, plot_path)
    
    # Generate Grad-CAM examples if requested
    if args.gradcam:
        gradcam_dir = os.path.join(args.output_dir, 'gradcam')
        generate_gradcam_examples(
            model=model,
            dataset=test_dataset,
            transform=test_transform,
            n_way=config['training']['n_way'],
            k_shot=config['training']['k_shot'],
            device=device,
            save_dir=gradcam_dir,
            num_examples=10
        )
    
    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()
