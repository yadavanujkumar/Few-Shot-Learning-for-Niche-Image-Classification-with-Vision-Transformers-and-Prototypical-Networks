FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data models checkpoints mlruns mlartifacts

# Expose port
EXPOSE 8000

# Set environment variables
ENV CONFIG_PATH=configs/config.yaml
ENV CHECKPOINT_PATH=checkpoints/best_model_5way_5shot.pth
ENV PROTOTYPE_PATH=models/prototypes.pkl
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Run the API server
CMD ["python", "src/api/app.py"]
