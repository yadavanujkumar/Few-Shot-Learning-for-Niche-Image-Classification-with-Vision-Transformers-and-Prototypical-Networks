# Few-Shot Learning for Niche Image Classification

A robust image classification system using **Vision Transformers (ViT)** and **Prototypical Networks** for few-shot learning scenarios. This implementation addresses the challenge of classifying rare objects or anomalies with extremely limited labeled training data.

## 🎯 Key Features

- **Vision Transformer (ViT)** feature extraction with pre-trained models
- **Prototypical Networks** for efficient few-shot learning
- **Episodic training** for N-way K-shot classification
- **MLflow integration** for experiment tracking
- **Grad-CAM visualization** for model interpretability
- **FastAPI deployment** ready for production use
- Support for 1-shot, 5-shot, and custom few-shot scenarios

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference API](#inference-api)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Results](#results)
- [Requirements](#requirements)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yadavanujkumar/Few-Shot-Learning-for-Niche-Image-Classification-with-Vision-Transformers-and-Prototypical-Networks.git
cd Few-Shot-Learning-for-Niche-Image-Classification-with-Vision-Transformers-and-Prototypical-Networks
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🎬 Quick Start

### Option 1: Using Dummy Data (for testing)

```bash
# Train with automatically generated dummy dataset
python scripts/train.py --config configs/config.yaml --create-dummy-data

# Evaluate the trained model
python scripts/evaluate.py --config configs/config.yaml \
    --checkpoint checkpoints/best_model_5way_5shot.pth \
    --gradcam
```

### Option 2: Using Mini-ImageNet Dataset

1. Download the mini-ImageNet dataset and organize it as:
```
data/mini-imagenet/
├── train/
│   ├── class_001/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

2. Train the model:
```bash
python scripts/train.py --config configs/config.yaml
```

## 📁 Project Structure

```
.
├── configs/
│   └── config.yaml              # Configuration file
├── src/
│   ├── data/
│   │   └── dataset.py           # Dataset loader and few-shot sampler
│   ├── models/
│   │   ├── feature_extractor.py # ViT and ResNet feature extractors
│   │   └── prototypical_network.py  # Prototypical Network implementation
│   ├── utils/
│   │   ├── gradcam.py           # Grad-CAM visualization
│   │   └── mlflow_utils.py      # MLflow tracking utilities
│   └── api/
│       └── app.py               # FastAPI application
├── scripts/
│   ├── train.py                 # Training script
│   └── evaluate.py              # Evaluation script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 📖 Usage

### Training

Train a prototypical network for few-shot learning:

```bash
python scripts/train.py \
    --config configs/config.yaml \
    --device auto \
    --create-dummy-data  # Optional: create dummy data for testing
```

**Key training parameters (in config.yaml):**
- `n_way`: Number of classes per episode (default: 5)
- `k_shot`: Number of support examples per class (default: 5)
- `n_query`: Number of query examples per class (default: 15)
- `num_episodes`: Total training episodes (default: 10000)

### Evaluation

Evaluate the trained model on different N-way K-shot scenarios:

```bash
python scripts/evaluate.py \
    --config configs/config.yaml \
    --checkpoint checkpoints/best_model_5way_5shot.pth \
    --num-episodes 600 \
    --gradcam \
    --output-dir evaluation_results
```

This will:
- Evaluate on all configured N-way K-shot scenarios (e.g., 5-way 1-shot, 5-way 5-shot)
- Generate performance plots
- Create Grad-CAM visualizations (if `--gradcam` is specified)
- Save results to the output directory

### Inference API

#### Starting the API Server

```bash
# Set environment variables
export CONFIG_PATH=configs/config.yaml
export CHECKPOINT_PATH=checkpoints/best_model_5way_5shot.pth
export PROTOTYPE_PATH=models/prototypes.pkl

# Start the server
python src/api/app.py
```

Or using uvicorn directly:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

#### API Endpoints

1. **Register a new class** (POST `/register_class`):
```bash
curl -X POST "http://localhost:8000/register_class" \
  -F "class_name=defect_type_A" \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg" \
  -F "images=@image3.jpg"
```

2. **Classify an image** (POST `/classify_image`):
```bash
curl -X POST "http://localhost:8000/classify_image" \
  -F "image=@test_image.jpg" \
  -F "top_k=3"
```

3. **List registered classes** (GET `/list_classes`):
```bash
curl "http://localhost:8000/list_classes"
```

4. **Delete a class** (DELETE `/delete_class`):
```bash
curl -X DELETE "http://localhost:8000/delete_class?class_name=defect_type_A"
```

5. **Health check** (GET `/health`):
```bash
curl "http://localhost:8000/health"
```

## ⚙️ Configuration

Edit `configs/config.yaml` to customize the system:

```yaml
model:
  feature_extractor: "google/vit-base-patch16-224"  # or "resnet50"
  embedding_dim: 768
  freeze_backbone: true

training:
  num_episodes: 10000
  n_way: 5
  k_shot: 5
  n_query: 15
  learning_rate: 0.001

dataset:
  name: "mini-imagenet"
  image_size: 224

evaluation:
  n_way_eval: [5]
  k_shot_eval: [1, 5]
  num_eval_episodes: 600
```

## 🏗️ Architecture

### Prototypical Networks

The system uses Prototypical Networks, which:

1. **Extract Features**: Use a pre-trained Vision Transformer to extract features from support and query images
2. **Compute Prototypes**: Calculate class prototypes as the mean of support set embeddings for each class
3. **Classify by Distance**: Assign query images to the class with the nearest prototype in embedding space

```
Support Set → ViT → Embeddings → Compute Prototypes
                                        ↓
Query Image → ViT → Embedding → Distance to Prototypes → Classification
```

### Vision Transformer (ViT)

- Uses pre-trained ViT from Hugging Face Transformers
- Extracts rich semantic features using self-attention
- Can be frozen or fine-tuned based on configuration
- Alternative: ResNet or EfficientNet backbones also supported

## 📊 Results

The system is evaluated on N-way K-shot classification tasks:

- **5-way 1-shot**: Classification with only 1 example per class
- **5-way 5-shot**: Classification with 5 examples per class

Results are tracked in MLflow and include:
- Classification accuracy (mean ± confidence interval)
- Training/validation loss curves
- Grad-CAM visualizations
- Model checkpoints

Example output:
```
5-way 1-shot: 55.2% ± 1.3%
5-way 5-shot: 72.8% ± 0.9%
```

## 🔬 Model Interpretability

The system includes Grad-CAM (Gradient-weighted Class Activation Mapping) for visual explanations:

```python
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model_5way_5shot.pth \
    --gradcam
```

This generates visualizations showing which regions of the image the model focuses on for its predictions.

## 📦 Requirements

### Core Dependencies

- Python 3.8+
- PyTorch 2.0+
- Transformers (Hugging Face)
- MLflow
- FastAPI
- OpenCV
- Matplotlib

See `requirements.txt` for complete list.

### Hardware

- **Minimum**: CPU with 8GB RAM
- **Recommended**: GPU with 8GB+ VRAM (CUDA-compatible)
- Training time: ~1-2 hours on GPU for 10,000 episodes

## 🎯 Use Cases

This system is designed for:

1. **Manufacturing**: Detect subtle defects with limited examples
2. **Medical Imaging**: Classify rare conditions from few scans
3. **Biodiversity**: Identify new or rare species
4. **Quality Control**: Rapid adaptation to new defect types
5. **Anomaly Detection**: Learn from few examples of anomalies

## 🔍 MLflow Tracking

All experiments are tracked in MLflow:

```bash
# View experiments
mlflow ui --backend-store-uri ./mlruns
```

Navigate to `http://localhost:5000` to view:
- Hyperparameters
- Training metrics
- Model artifacts
- Grad-CAM visualizations

## 🚢 Deployment

### Docker Deployment (Recommended)

Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t few-shot-api .
docker run -p 8000:8000 \
  -e CONFIG_PATH=configs/config.yaml \
  -e CHECKPOINT_PATH=checkpoints/best_model_5way_5shot.pth \
  few-shot-api
```

## 📝 Technical Requirements Checklist

- [x] Python with PyTorch
- [x] Pre-trained Vision Transformer (ViT)
- [x] Prototypical Network implementation
- [x] MLflow experiment tracking
- [x] Episodic training
- [x] N-way K-shot classification accuracy reporting
- [x] Grad-CAM for interpretability
- [x] FastAPI inference pipeline with class registration

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Vision Transformer implementation from Hugging Face Transformers
- Prototypical Networks paper: [Snell et al., 2017](https://arxiv.org/abs/1703.05175)
- Mini-ImageNet dataset for few-shot learning benchmarks

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This implementation focuses on educational purposes and demonstrating few-shot learning capabilities. For production use, consider additional optimizations and security measures.