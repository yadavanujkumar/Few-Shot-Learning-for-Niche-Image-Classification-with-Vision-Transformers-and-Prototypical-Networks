# Deployment Guide

This guide provides detailed instructions for deploying the Few-Shot Learning system in various environments.

## Table of Contents

1. [Local Development Setup](#local-development-setup)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [API Usage Examples](#api-usage-examples)
5. [MLflow Dashboard](#mlflow-dashboard)
6. [Troubleshooting](#troubleshooting)

---

## Local Development Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment tool (venv or conda)
- 8GB+ RAM recommended
- GPU with CUDA support (optional, but recommended for training)

### Step 1: Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd Few-Shot-Learning-for-Niche-Image-Classification-with-Vision-Transformers-and-Prototypical-Networks

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Data

Option A - Use dummy data for testing:
```bash
python scripts/train.py --create-dummy-data
```

Option B - Use real mini-ImageNet dataset:
1. Download mini-ImageNet from: https://github.com/twitter/meta-learning-lstm
2. Organize as:
   ```
   data/mini-imagenet/
   ├── train/
   ├── val/
   └── test/
   ```

### Step 3: Train Model

```bash
# Basic training
python scripts/train.py --config configs/config.yaml

# Training with custom settings
python scripts/train.py \
    --config configs/config.yaml \
    --device cuda
```

### Step 4: Evaluate Model

```bash
python scripts/evaluate.py \
    --config configs/config.yaml \
    --checkpoint checkpoints/best_model_5way_5shot.pth \
    --gradcam \
    --output-dir evaluation_results
```

### Step 5: Start API Server

```bash
# Set environment variables
export CONFIG_PATH=configs/config.yaml
export CHECKPOINT_PATH=checkpoints/best_model_5way_5shot.pth
export PROTOTYPE_PATH=models/prototypes.pkl

# Start server
python src/api/app.py
```

Or using uvicorn:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## Docker Deployment

### Prerequisites

- Docker installed (version 20.10+)
- Docker Compose (optional, but recommended)

### Build and Run with Docker

```bash
# Build the image
docker build -t few-shot-learning:latest .

# Run the container
docker run -d \
  --name few-shot-api \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/checkpoints:/app/checkpoints \
  few-shot-learning:latest
```

### Using Docker Compose

```bash
# Start all services (API + MLflow UI)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

This starts:
- API server on port 8000
- MLflow UI on port 5000

---

## Cloud Deployment

### AWS Deployment (EC2)

1. **Launch EC2 Instance**
   - Instance type: t3.medium or larger (GPU: p3.2xlarge)
   - AMI: Ubuntu 20.04 LTS
   - Security group: Allow ports 22, 8000, 5000

2. **Setup Instance**
   ```bash
   # SSH into instance
   ssh -i your-key.pem ubuntu@your-instance-ip

   # Install Docker
   sudo apt-get update
   sudo apt-get install -y docker.io docker-compose
   sudo usermod -aG docker ubuntu

   # Clone repository
   git clone <repository-url>
   cd Few-Shot-Learning-for-Niche-Image-Classification-with-Vision-Transformers-and-Prototypical-Networks

   # Start services
   docker-compose up -d
   ```

3. **Access API**
   - API: http://your-instance-ip:8000
   - MLflow: http://your-instance-ip:5000

### Google Cloud Platform (GCP)

1. **Create Compute Engine Instance**
   ```bash
   gcloud compute instances create few-shot-learning \
     --machine-type=n1-standard-4 \
     --image-family=ubuntu-2004-lts \
     --image-project=ubuntu-os-cloud \
     --boot-disk-size=50GB
   ```

2. **Deploy Application**
   ```bash
   # SSH into instance
   gcloud compute ssh few-shot-learning

   # Follow same steps as AWS EC2
   ```

### Azure Deployment

1. **Create Virtual Machine**
   ```bash
   az vm create \
     --resource-group your-resource-group \
     --name few-shot-learning \
     --image UbuntuLTS \
     --size Standard_D4s_v3 \
     --generate-ssh-keys
   ```

2. **Deploy Application**
   - Follow same Docker deployment steps

---

## API Usage Examples

### Using cURL

#### Register a New Class

```bash
curl -X POST "http://localhost:8000/register_class" \
  -F "class_name=defect_type_scratches" \
  -F "images=@support_img1.jpg" \
  -F "images=@support_img2.jpg" \
  -F "images=@support_img3.jpg" \
  -F "images=@support_img4.jpg" \
  -F "images=@support_img5.jpg"
```

Response:
```json
{
  "message": "Successfully registered class 'defect_type_scratches'",
  "class_id": 0,
  "num_support_images": 5,
  "total_registered_classes": 1
}
```

#### Classify an Image

```bash
curl -X POST "http://localhost:8000/classify_image" \
  -F "image=@query_image.jpg" \
  -F "top_k=3"
```

Response:
```json
{
  "predictions": [
    {
      "class_name": "defect_type_scratches",
      "confidence": 0.89,
      "distance": 2.34
    },
    {
      "class_name": "defect_type_dents",
      "confidence": 0.08,
      "distance": 8.12
    }
  ],
  "num_registered_classes": 5
}
```

### Using Python Requests

```python
import requests

# Register class
files = [
    ('images', open('support_img1.jpg', 'rb')),
    ('images', open('support_img2.jpg', 'rb')),
]
data = {'class_name': 'defect_type_scratches'}
response = requests.post(
    'http://localhost:8000/register_class',
    files=files,
    data=data
)
print(response.json())

# Classify image
files = {'image': open('query_image.jpg', 'rb')}
data = {'top_k': 3}
response = requests.post(
    'http://localhost:8000/classify_image',
    files=files,
    data=data
)
print(response.json())
```

### Using JavaScript/Node.js

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

// Register class
const form = new FormData();
form.append('class_name', 'defect_type_scratches');
form.append('images', fs.createReadStream('support_img1.jpg'));
form.append('images', fs.createReadStream('support_img2.jpg'));

axios.post('http://localhost:8000/register_class', form, {
    headers: form.getHeaders()
}).then(response => {
    console.log(response.data);
});

// Classify image
const classifyForm = new FormData();
classifyForm.append('image', fs.createReadStream('query_image.jpg'));
classifyForm.append('top_k', '3');

axios.post('http://localhost:8000/classify_image', classifyForm, {
    headers: classifyForm.getHeaders()
}).then(response => {
    console.log(response.data);
});
```

---

## MLflow Dashboard

### Starting MLflow UI

```bash
# Local
mlflow ui --backend-store-uri ./mlruns --port 5000

# Docker
docker-compose up mlflow-ui
```

### Accessing the Dashboard

Navigate to: http://localhost:5000

### Features Available

1. **Experiment Tracking**
   - View all training runs
   - Compare hyperparameters
   - Visualize metrics over time

2. **Model Registry**
   - Browse saved models
   - View model artifacts
   - Download checkpoints

3. **Artifacts**
   - Grad-CAM visualizations
   - Model checkpoints
   - Configuration files

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Solution:**
- Reduce batch size (n_way * k_shot * n_query)
- Use CPU instead: `--device cpu`
- Enable gradient checkpointing

#### 2. Module Not Found Errors

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 3. API Server Not Starting

**Solution:**
```bash
# Check if port is in use
lsof -i :8000

# Use different port
export API_PORT=8080
python src/api/app.py
```

#### 4. Model Download Issues

**Solution:**
```bash
# Download model manually
python -c "from transformers import ViTModel; ViTModel.from_pretrained('google/vit-base-patch16-224')"
```

#### 5. MLflow Tracking Issues

**Solution:**
```bash
# Clear MLflow cache
rm -rf mlruns mlartifacts

# Restart with fresh tracking
python scripts/train.py --config configs/config.yaml
```

### Performance Optimization

#### For Training

1. **Use GPU**
   ```bash
   python scripts/train.py --device cuda
   ```

2. **Adjust batch size**
   - Modify `n_way`, `k_shot`, `n_query` in config.yaml

3. **Reduce model size**
   - Use smaller ViT: `vit-small-patch16-224`
   - Or use ResNet: `resnet18`

#### For Inference

1. **Freeze backbone**
   ```yaml
   model:
     freeze_backbone: true
   ```

2. **Batch processing**
   - Register multiple classes at once
   - Process multiple queries together

3. **Model quantization**
   ```python
   import torch
   model = torch.quantization.quantize_dynamic(
       model, {torch.nn.Linear}, dtype=torch.qint8
   )
   ```

### Getting Help

- Check logs: `docker-compose logs -f`
- View API docs: http://localhost:8000/docs
- Open GitHub issue with:
  - Error message
  - System info
  - Steps to reproduce

---

## Production Considerations

### Security

1. **API Authentication**
   - Add JWT tokens
   - Use API keys
   - Rate limiting

2. **HTTPS**
   - Use reverse proxy (Nginx)
   - SSL certificates

3. **Input Validation**
   - Image format checks
   - Size limits
   - Content validation

### Scalability

1. **Horizontal Scaling**
   - Multiple API instances
   - Load balancer
   - Shared model storage

2. **Caching**
   - Cache prototypes
   - Redis for sessions
   - Model caching

3. **Monitoring**
   - Prometheus metrics
   - Grafana dashboards
   - Error tracking (Sentry)

### Backup and Recovery

1. **Model Checkpoints**
   - Regular backups
   - Version control
   - S3/Cloud storage

2. **Registered Classes**
   - Backup prototypes.pkl
   - Export/import functionality
   - Database storage

---

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
