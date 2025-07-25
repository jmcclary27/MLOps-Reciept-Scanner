FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy project files and credentials
COPY . .
COPY vertex_dvc_key.json /app/vertex_dvc_key.json

# Install system-level dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r train_requirements.txt

# Set credentials for DVC
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/vertex_dvc_key.json"

# Default entrypoint for Vertex training
ENTRYPOINT ["bash", "-c", "dvc pull && python train_model.py --csv artifacts/data/receipt_dataset.csv"]
