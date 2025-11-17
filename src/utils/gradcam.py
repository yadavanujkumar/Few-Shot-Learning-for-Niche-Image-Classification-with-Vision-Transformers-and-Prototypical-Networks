"""
Grad-CAM implementation for model interpretability.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple
import matplotlib.pyplot as plt


class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping
    for visualizing important regions in the input image.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Args:
            model: The model to visualize
            target_layer: The target layer to compute gradients for
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Hook to save forward pass activations."""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the input image.
        
        Args:
            input_image: Input image tensor [1, C, H, W]
            target_class: Target class index. If None, uses the predicted class.
        
        Returns:
            cam: Grad-CAM heatmap as numpy array [H, W]
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_image)
        
        # Use predicted class if target_class is not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for the target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients  # [1, C, H', W']
        activations = self.activations  # [1, C, H', W']
        
        # Compute weights as global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [1, 1, H', W']
        
        # Apply ReLU to focus on positive contributions
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def generate_cam_for_prototypical(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_image: torch.Tensor,
        n_way: int
    ) -> np.ndarray:
        """
        Generate Grad-CAM for prototypical network.
        
        Args:
            support_images: Support images [n_way * k_shot, C, H, W]
            support_labels: Support labels [n_way * k_shot]
            query_image: Query image [1, C, H, W]
            n_way: Number of classes
        
        Returns:
            cam: Grad-CAM heatmap [H, W]
        """
        self.model.eval()
        
        # Forward pass
        logits, _ = self.model(support_images, support_labels, query_image, n_way)
        
        # Get predicted class
        pred_class = logits.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for predicted class
        one_hot = torch.zeros_like(logits)
        one_hot[0, pred_class] = 1
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        if self.gradients is None or self.activations is None:
            return np.zeros((224, 224))
        
        gradients = self.gradients
        activations = self.activations
        
        # Compute weights
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam


def visualize_gradcam(
    image: torch.Tensor,
    cam: np.ndarray,
    save_path: Optional[str] = None,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on the original image.
    
    Args:
        image: Original image tensor [C, H, W]
        cam: Grad-CAM heatmap [H', W']
        save_path: Path to save the visualization
        alpha: Transparency of the heatmap overlay
    
    Returns:
        visualization: RGB image with heatmap overlay
    """
    # Convert image to numpy
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    # Denormalize if needed (assuming ImageNet normalization)
    if image.max() <= 1.0:
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        image = image * std + mean
    
    # Convert from CHW to HWC
    image = np.transpose(image, (1, 2, 0))
    
    # Clip to [0, 1]
    image = np.clip(image, 0, 1)
    
    # Resize CAM to match image size
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255.0
    
    # Overlay heatmap on image
    visualization = alpha * heatmap + (1 - alpha) * image
    visualization = np.clip(visualization, 0, 1)
    
    # Save if path provided
    if save_path:
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(cam_resized, cmap='jet')
        plt.title('Grad-CAM Heatmap')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(visualization)
        plt.title('Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return visualization


def get_target_layer_vit(model):
    """
    Get the appropriate target layer for Grad-CAM from a ViT model.
    
    Args:
        model: ViT-based model
    
    Returns:
        target_layer: The layer to use for Grad-CAM
    """
    # For ViT, we typically use the last layer norm or the last encoder layer
    if hasattr(model, 'vit'):
        # If wrapped in ViTFeatureExtractor
        if hasattr(model.vit, 'layernorm'):
            return model.vit.layernorm
        elif hasattr(model.vit, 'encoder'):
            return model.vit.encoder.layer[-1]
    elif hasattr(model, 'encoder'):
        # Direct ViT model
        return model.encoder.layer[-1]
    
    # Fallback
    return list(model.modules())[-2]


def get_target_layer_resnet(model):
    """
    Get the appropriate target layer for Grad-CAM from a ResNet model.
    
    Args:
        model: ResNet-based model
    
    Returns:
        target_layer: The layer to use for Grad-CAM
    """
    # For ResNet, we typically use the last convolutional layer
    if hasattr(model, 'backbone'):
        # If wrapped in ResNetFeatureExtractor
        if hasattr(model.backbone, 'layer4'):
            return model.backbone.layer4
    elif hasattr(model, 'layer4'):
        # Direct ResNet model
        return model.layer4
    
    # Fallback
    return list(model.modules())[-2]
