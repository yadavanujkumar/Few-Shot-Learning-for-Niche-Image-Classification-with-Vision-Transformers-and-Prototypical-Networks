"""
FastAPI application for few-shot learning inference.
Provides endpoints for registering new classes and classifying images.
"""
import os
import sys
import io
import pickle
from typing import List, Optional
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.feature_extractor import create_feature_extractor
from src.models.prototypical_network import PrototypicalNetwork
from src.data.dataset import get_transforms
from src.utils.mlflow_utils import load_config


# Global variables
app = FastAPI(
    title="Few-Shot Learning API",
    description="API for few-shot image classification using Prototypical Networks",
    version="1.0.0"
)

model = None
transform = None
device = None
registered_classes = {}
prototypes = {}


def load_model(config_path: str, checkpoint_path: str):
    """Load the trained model."""
    global model, transform, device
    
    # Load configuration
    config = load_config(config_path)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    feature_extractor = create_feature_extractor(
        model_name=config['model']['feature_extractor'],
        freeze=True,  # Always freeze for inference
        embedding_dim=config['model']['embedding_dim']
    )
    
    model = PrototypicalNetwork(feature_extractor)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create transform
    transform = get_transforms(config, is_training=False)
    
    print(f"Model loaded successfully on {device}")


def load_prototypes(prototype_path: str):
    """Load previously saved prototypes."""
    global registered_classes, prototypes
    
    if os.path.exists(prototype_path):
        with open(prototype_path, 'rb') as f:
            data = pickle.load(f)
            registered_classes = data.get('classes', {})
            prototypes = data.get('prototypes', {})
        print(f"Loaded {len(registered_classes)} registered classes")
    else:
        print("No previous prototypes found, starting fresh")


def save_prototypes(prototype_path: str):
    """Save current prototypes to disk."""
    os.makedirs(os.path.dirname(prototype_path), exist_ok=True)
    with open(prototype_path, 'wb') as f:
        pickle.dump({
            'classes': registered_classes,
            'prototypes': prototypes
        }, f)
    print(f"Saved {len(registered_classes)} registered classes")


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    config_path = os.getenv('CONFIG_PATH', 'configs/config.yaml')
    checkpoint_path = os.getenv('CHECKPOINT_PATH', 'checkpoints/best_model_5way_5shot.pth')
    prototype_path = os.getenv('PROTOTYPE_PATH', 'models/prototypes.pkl')
    
    try:
        load_model(config_path, checkpoint_path)
        load_prototypes(prototype_path)
    except Exception as e:
        print(f"Warning: Failed to load model on startup: {e}")
        print("Model will need to be loaded manually")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Few-Shot Learning API",
        "endpoints": {
            "health": "/health",
            "register_class": "/register_class",
            "classify_image": "/classify_image",
            "list_classes": "/list_classes",
            "delete_class": "/delete_class"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "registered_classes": len(registered_classes),
        "device": str(device) if device else None
    }


@app.post("/register_class")
async def register_class(
    class_name: str = Form(...),
    images: List[UploadFile] = File(...)
):
    """
    Register a new class by providing K support images.
    
    Args:
        class_name: Name of the class to register
        images: List of K support images for the class
    
    Returns:
        Success message and class information
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if class_name in registered_classes:
        raise HTTPException(status_code=400, detail=f"Class '{class_name}' already registered")
    
    if len(images) == 0:
        raise HTTPException(status_code=400, detail="At least one image is required")
    
    try:
        # Load and transform images
        embeddings = []
        
        for image_file in images:
            # Read image
            image_bytes = await image_file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Transform
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Extract features
            with torch.no_grad():
                embedding = model.feature_extractor(image_tensor)
                embeddings.append(embedding)
        
        # Compute prototype as mean of embeddings
        embeddings = torch.cat(embeddings, dim=0)  # [K, embedding_dim]
        prototype = embeddings.mean(dim=0)  # [embedding_dim]
        
        # Register class
        class_id = len(registered_classes)
        registered_classes[class_name] = class_id
        prototypes[class_id] = prototype.cpu()
        
        # Save prototypes
        prototype_path = os.getenv('PROTOTYPE_PATH', 'models/prototypes.pkl')
        save_prototypes(prototype_path)
        
        return {
            "message": f"Successfully registered class '{class_name}'",
            "class_id": class_id,
            "num_support_images": len(images),
            "total_registered_classes": len(registered_classes)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")


@app.post("/classify_image")
async def classify_image(
    image: UploadFile = File(...),
    top_k: int = Form(3)
):
    """
    Classify an image against registered classes.
    
    Args:
        image: Image to classify
        top_k: Number of top predictions to return
    
    Returns:
        Top-K predictions with class names and confidence scores
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(registered_classes) == 0:
        raise HTTPException(status_code=400, detail="No classes registered. Please register classes first.")
    
    try:
        # Read and transform image
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = transform(img).unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            query_embedding = model.feature_extractor(image_tensor)  # [1, embedding_dim]
        
        # Compute distances to all prototypes
        distances = []
        class_names = []
        
        for class_name, class_id in registered_classes.items():
            prototype = prototypes[class_id].to(device).unsqueeze(0)  # [1, embedding_dim]
            
            # Euclidean distance
            dist = torch.sum((query_embedding - prototype) ** 2).item()
            distances.append(dist)
            class_names.append(class_name)
        
        # Convert distances to probabilities (softmax of negative distances)
        distances = np.array(distances)
        logits = -distances
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        
        # Get top-K predictions
        top_k = min(top_k, len(class_names))
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        
        predictions = [
            {
                "class_name": class_names[idx],
                "confidence": float(probabilities[idx]),
                "distance": float(distances[idx])
            }
            for idx in top_indices
        ]
        
        return {
            "predictions": predictions,
            "num_registered_classes": len(registered_classes)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error classifying image: {str(e)}")


@app.get("/list_classes")
async def list_classes():
    """List all registered classes."""
    return {
        "registered_classes": list(registered_classes.keys()),
        "total_classes": len(registered_classes)
    }


@app.delete("/delete_class")
async def delete_class(class_name: str):
    """
    Delete a registered class.
    
    Args:
        class_name: Name of the class to delete
    """
    if class_name not in registered_classes:
        raise HTTPException(status_code=404, detail=f"Class '{class_name}' not found")
    
    # Remove class
    class_id = registered_classes[class_name]
    del registered_classes[class_name]
    del prototypes[class_id]
    
    # Save prototypes
    prototype_path = os.getenv('PROTOTYPE_PATH', 'models/prototypes.pkl')
    save_prototypes(prototype_path)
    
    return {
        "message": f"Successfully deleted class '{class_name}'",
        "remaining_classes": len(registered_classes)
    }


if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment or use defaults
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', '8000'))
    
    uvicorn.run(app, host=host, port=port)
